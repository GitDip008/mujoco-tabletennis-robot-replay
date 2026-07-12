"""
plot_kinematics.py
──────────────────
Generates the thesis kinematics figure:
  Elbow flexion / Forearm pronation / Trunk sagittal flexion
  over time for one representative 1-second window per stroke class (2×2 grid).

Output:
    mujoco_sim/output/kinematics_plot.pdf   ← use this in thesis (vector)
    mujoco_sim/output/kinematics_plot.png   ← preview

Run:
    python plot_kinematics.py
"""

import csv
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE     = pathlib.Path(__file__).resolve().parent
SRC_DIR  = HERE.parent.parent / "src"
OUT_DIR  = HERE.parent / "output"
CSV_PATH = OUT_DIR / "tt_synthetic_full.csv"

sys.path.insert(0, str(SRC_DIR))
from kinematics import compute_joint_angle_series, _quat_from_row, _yaw_deg

STROKE_NAMES = {
    0: "No Stroke",
    1: "Forehand Topspin",
    2: "Backhand Drive",
    3: "Forehand Smash",
}
COLORS = {
    "elbow"      : "#F59E0B",   # amber
    "forearm"    : "#3B82F6",   # blue
    "torso"      : "#10B981",   # green
}
WINDOW   = 60    # frames per window
FPS      = 60
Y_MIN    =   0
Y_MAX    = 105   # shared y-axis across all panels


def load_one_window_per_class(csv_path: pathlib.Path) -> dict:
    """Return {label_id: [rows]} — first WINDOW-frame run for each class."""
    windows  = {}
    buffers  = {i: [] for i in range(4)}
    collected = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["stroke_label"])
            if label in collected:
                continue
            buffers[label].append(row)
            if len(buffers[label]) == WINDOW:
                windows[label] = buffers[label]
                collected.add(label)
            if len(collected) == 4:
                break

    return windows


def check_tpose_baseline(csv_path: pathlib.Path) -> float:
    """
    Return the forearm pronation (yaw) of the very first row in the CSV.
    This is the T-pose / rest baseline.  Should be ~0° if correctly zeroed,
    or a fixed anatomical offset (e.g. ~47°) that can be noted in the thesis.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        row    = next(reader)
    q_fore   = _quat_from_row(row, 8)
    baseline = _yaw_deg(q_fore)
    return baseline


def main():
    print(f"Reading {CSV_PATH} …")

    # ── Baseline check ────────────────────────────────────────────────────────
    baseline = check_tpose_baseline(CSV_PATH)
    print(f"Forearm pronation at T-pose (row 0): {baseline:.2f}°")
    if abs(baseline) > 5:
        print(f"  ↳ Non-zero baseline detected ({baseline:.1f}°). "
              f"This is the natural anatomical offset — not subtracted.")

    windows = load_one_window_per_class(CSV_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    t = np.arange(WINDOW) / FPS * 1000   # convert to milliseconds

    for idx, label_id in enumerate([0, 1, 2, 3]):
        ax    = axes[idx]
        rows  = windows.get(label_id, [])
        title = STROKE_NAMES[label_id]

        if not rows:
            ax.set_title(f"{title}\n(no data)")
            continue

        series = compute_joint_angle_series(rows)

        ax.plot(t, series["elbow"],        color=COLORS["elbow"],
                lw=2.0, label="Elbow flexion")
        ax.plot(t, series["forearm_roll"], color=COLORS["forearm"],
                lw=2.0, label="Forearm pronation")
        ax.plot(t, series["torso"],        color=COLORS["torso"],
                lw=2.0, label="Trunk sagittal flexion")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Joint Angle (°)", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, WINDOW / FPS * 1000)
        ax.set_ylim(Y_MIN, Y_MAX)

    fig.suptitle(
        "Elbow Flexion / Forearm Pronation / Trunk Flexion — "
        "One Representative Window per Stroke Class",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUT_DIR / "kinematics_plot.pdf"
    png_path = OUT_DIR / "kinematics_plot.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    print(f"Saved PDF → {pdf_path}")
    print(f"Saved PNG → {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
