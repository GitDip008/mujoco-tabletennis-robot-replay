import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from inference import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from kinematics import kinematic_summary
from feature_extractor import is_idle_window, window_energy

# ── Peak-based stroke detection parameters ──────────────────────────────────
# Each physical swing is a burst of accelerometer energy. We count one stroke
# per energy peak, with a refractory distance so the backswing + forward-swing
# sub-peaks of a SINGLE swing don't double-count, while two distinct swings
# (further apart) are still separated.
PEAK_MIN_DISTANCE = 10     # windows between peaks (~1.0 s at step=10, 100 Hz)
PEAK_MIN_HEIGHT   = 20.0   # min summed-accel-variance energy to count as a swing


def collapse_stroke_events(label_ids: np.ndarray, confidences: np.ndarray) -> list:
    """
    Collapse runs of consecutive identical non-zero predictions into single
    stroke events. A real swing spans several windows; without this every
    window is counted as a separate stroke.

    Returns a list of {label_id, peak_confidence, n_windows}.
    """
    events = []
    n = len(label_ids)
    i = 0
    while i < n:
        lab = int(label_ids[i])
        if lab == 0:
            i += 1
            continue
        j = i
        while j < n and int(label_ids[j]) == lab:
            j += 1
        run_conf = confidences[i:j]
        events.append({
            "label_id"       : lab,
            "peak_confidence": round(float(np.max(run_conf)), 4),
            "n_windows"      : int(j - i),
            "start_index"    : i,
        })
        i = j
    return events


def detect_stroke_events(label_ids: np.ndarray,
                         confidences: np.ndarray,
                         energies: np.ndarray,
                         min_distance: int = PEAK_MIN_DISTANCE,
                         min_height: float = PEAK_MIN_HEIGHT) -> list:
    """
    Count one stroke per motion-energy PEAK (not per label-run). This detects
    each individual swing even when several swings of the SAME class happen
    back-to-back — every physical swing has its own acceleration burst.

    Safeguard for rapid swings: the energy signal is first split into active
    segments separated by idle gaps (energy below min_height). find_peaks runs
    within the full signal with a refractory `min_distance`, but any two peaks
    separated by an idle window are always kept distinct (an idle gap is
    unambiguous evidence of two separate swings).

    Class for each peak = majority non-zero label in a ±2-window neighbourhood
    (robust to the exact peak window being gated to No Stroke).

    Returns a list of {label_id, peak_confidence, start_index, energy}.
    """
    energies    = np.asarray(energies, dtype=float)
    label_ids   = np.asarray(label_ids)
    confidences = np.asarray(confidences, dtype=float)
    n = len(energies)
    if n == 0:
        return []

    peaks, _ = find_peaks(energies, distance=min_distance, height=min_height)

    # Safeguard: also detect peaks that the refractory distance may have
    # swallowed but which are separated from their neighbour by an idle gap.
    # We do this by scanning active segments and ensuring at least one peak each.
    active = energies >= min_height
    seg_peaks = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        # segment [i, j): take the local argmax as a guaranteed peak
        seg_peaks.append(i + int(np.argmax(energies[i:j])))
        i = j

    all_peaks = sorted(set(peaks.tolist()) | set(seg_peaks))

    # Enforce refractory distance on the merged peak set (keep the higher peak)
    kept = []
    for p in all_peaks:
        if kept and (p - kept[-1]) < min_distance:
            # too close — keep whichever has the larger energy
            if energies[p] > energies[kept[-1]]:
                kept[-1] = p
            continue
        kept.append(p)

    events = []
    for p in kept:
        lo, hi = max(0, p - 2), min(n, p + 3)
        neigh = label_ids[lo:hi]
        nonzero = neigh[neigh != 0]
        if len(nonzero) == 0:
            continue   # peak with no stroke label nearby → ignore
        lab  = int(np.bincount(nonzero).argmax())
        conf = round(float(np.max(confidences[lo:hi])), 4)
        events.append({
            "label_id"       : lab,
            "peak_confidence": conf,
            "start_index"    : int(p),
            "energy"         : round(float(energies[p]), 2),
        })
    return events


# ######################### Constants #########################

STROKE_CLASSES   = [1, 2, 3]        # exclude class 0 (No Stroke)
MIN_CONFIDENCE   = 0.60             # below this → flag as low-confidence
WEAK_STROKE_THR  = 0.65             # avg confidence below this → weak stroke


def run_session(
    predictor: StrokePredictor,
    session_df: pd.DataFrame,
    raw_rows: list = None,
) -> dict:
    """
    Run inference over an entire session DataFrame and return
    a structured summary dict ready for the LLM prompt.

    Args:
        predictor   : StrokePredictor instance (subject-specific)
        session_df  : DataFrame with FEATURE_COLS columns
                      (one row per segmented swing event)

    Returns:
        session_summary dict — see structure below
    """
    features = session_df[FEATURE_COLS].values.astype(np.float32)
    results  = predictor.predict_batch(features)

    # ── Energy gate: force No Stroke on near-motionless windows ─────────────
    # Keeps the report consistent with the live GUI path.
    for i, feats in enumerate(features):
        if is_idle_window(feats):
            results[i] = {
                "label_id"     : 0,
                "label_name"   : CLASS_NAMES[0],
                "confidence"   : results[i]["confidence"],
                "probabilities": results[i]["probabilities"],
            }

    # ── Per-window arrays (after gating) ────────────────────────────────────
    label_ids    = np.array([r["label_id"]   for r in results])
    confidences  = np.array([r["confidence"] for r in results])
    energies      = np.array([window_energy(f) for f in features])

    total_swings = len(results)

    # ── Stroke-event detection: one event per motion-energy peak ────────────
    # Counts each individual swing, including back-to-back same-class swings.
    events = detect_stroke_events(label_ids, confidences, energies)
    total_strokes = len(events)

    # Per-class counts + average peak-confidence from EVENTS, not windows
    stroke_distribution = {}
    for cid in STROKE_CLASSES:
        name        = CLASS_NAMES[cid]
        cls_events  = [e for e in events if e["label_id"] == cid]
        count       = len(cls_events)
        pct         = round(count / total_strokes * 100, 1) if total_strokes > 0 else 0.0
        avg_conf    = (round(float(np.mean([e["peak_confidence"] for e in cls_events])), 4)
                       if cls_events else 0.0)
        stroke_distribution[name] = {
            "count"           : count,
            "percentage"      : pct,
            "avg_confidence"  : avg_conf,
            "is_weak"         : avg_conf < WEAK_STROKE_THR and count > 0,
        }

    # ######################### Session-level stats #########################
    no_stroke_count  = int(np.sum(label_ids == 0))
    overall_avg_conf = round(float(np.mean(confidences)), 4) if total_swings else 0.0
    low_conf_count   = int(np.sum(confidences < MIN_CONFIDENCE))
    low_conf_pct     = round(low_conf_count / total_swings * 100, 1) if total_swings else 0.0

    # Tempo: gap between consecutive stroke EVENTS (window-index proxy)
    event_starts = [e["start_index"] for e in events]
    if len(event_starts) > 1:
        intervals        = np.diff(event_starts)
        tempo_mean       = round(float(np.mean(intervals)), 2)
        tempo_std        = round(float(np.std(intervals)), 2)
        tempo_cv         = round(tempo_std / tempo_mean, 4) if tempo_mean > 0 else 0.0
        tempo_label      = _tempo_label(tempo_cv)
    else:
        tempo_mean = tempo_std = tempo_cv = 0.0
        tempo_label = "insufficient data"

    # Dominant stroke
    dominant_stroke = (
        max(stroke_distribution, key=lambda k: stroke_distribution[k]["count"])
        if total_strokes > 0 else "None"
    )

    # Weak strokes list
    weak_strokes = [
        name for name, stats in stroke_distribution.items()
        if stats["is_weak"]
    ]

    # ── Kinematics (optional — requires raw IMU rows) ─────────────────────────
    kinematics = None
    if raw_rows and len(raw_rows) >= 2:
        try:
            kinematics = kinematic_summary(raw_rows)
        except Exception:
            kinematics = None

    # ######################### Assemble summary #########################
    summary = {
        "total_events"        : total_swings,
        "total_strokes"       : total_strokes,
        "no_stroke_events"    : no_stroke_count,
        "overall_avg_confidence" : overall_avg_conf,
        "low_confidence_count"   : low_conf_count,
        "low_confidence_pct"     : low_conf_pct,
        "stroke_distribution" : stroke_distribution,
        "dominant_stroke"     : dominant_stroke,
        "weak_strokes"        : weak_strokes,
        "tempo": {
            "mean_interval"   : tempo_mean,
            "std_interval"    : tempo_std,
            "coeff_variation" : tempo_cv,
            "label"           : tempo_label,
        },
        "kinematics"          : kinematics,
    }
    return summary


def format_summary_for_prompt(summary: dict) -> str:
    """
    Converts the summary dict into a clean, structured text block
    for injection into the LLM coaching prompt.
    """
    lines = [
        "=== SESSION SUMMARY ===",
        f"Total events recorded : {summary['total_events']}",
        f"Total stroke events   : {summary['total_strokes']}",
        f"No-stroke intervals   : {summary['no_stroke_events']}",
        f"Overall avg confidence: {summary['overall_avg_confidence']:.2%}",
        f"Low-confidence events : {summary['low_confidence_count']} "
        f"({summary['low_confidence_pct']}% of session)",
        "",
        "--- Stroke Distribution ---",
    ]

    for name, stats in summary["stroke_distribution"].items():
        flag = " ← LOW CONFIDENCE" if stats["is_weak"] else ""
        lines.append(
            f"  {name:<20} {stats['count']:>5} shots "
            f"({stats['percentage']:>5.1f}%)  "
            f"avg_conf={stats['avg_confidence']:.2%}{flag}"
        )

    lines += [
        "",
        "--- Tempo ---",
        f"  Pattern   : {summary['tempo']['label']}",
        f"  Mean gap  : {summary['tempo']['mean_interval']:.1f} events",
        f"  Std dev   : {summary['tempo']['std_interval']:.1f}",
        f"  CV        : {summary['tempo']['coeff_variation']:.3f}",
        "",
        f"Dominant stroke : {summary['dominant_stroke']}",
        f"Weak strokes    : {', '.join(summary['weak_strokes']) if summary['weak_strokes'] else 'None'}",
    ]

    km = summary.get("kinematics")
    if km:
        lines += [
            "",
            "--- Kinematics (joint angles, deg) ---",
            f"  Elbow peak angle      : {km['elbow_peak_angle']:.1f}°",
            f"  Elbow range           : {km['elbow_range']:.1f}°",
            f"  Elbow peak velocity   : {km['elbow_peak_velocity']:.1f} deg/s",
            f"  Time to elbow peak    : {km['time_to_elbow_peak']:.3f}s",
            f"  Forearm pronation range : {km['forearm_roll_range']:.1f}°",
            f"  Trunk sagittal flexion  : {km['torso_range']:.1f}°",
        ]

    lines.append("=== END SUMMARY ===")
    return "\n".join(lines)


def _tempo_label(cv: float) -> str:
    if cv < 0.2:  return "consistent"
    if cv < 0.4:  return "moderate variance"
    return "irregular"


# #########################  Smoke test #########################
if __name__ == "__main__":
    import pathlib

    root      = pathlib.Path(__file__).resolve().parent.parent
    SUBJECT_ID = 10

    df        = pd.read_csv(root / "data/raw/TTSWING.csv")
    session   = df[df["id"] == SUBJECT_ID].copy()

    predictor = StrokePredictor.from_subject(subject_id=SUBJECT_ID)
    summary   = run_session(predictor, session)
    prompt_block = format_summary_for_prompt(summary)

    print(prompt_block)
    print()
    print("Raw summary dict keys:", list(summary.keys()))
    print("Weak strokes detected:", summary["weak_strokes"])
    print("Dominant stroke      :", summary["dominant_stroke"])