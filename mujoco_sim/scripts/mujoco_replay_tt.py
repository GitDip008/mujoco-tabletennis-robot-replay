"""
mujoco_replay_tt.py
───────────────────
Capture-and-replay table-tennis demo:

  PHASE 1 (LIVE_RECORD)
    Player swings in front of the avatar. Each swing is buffered (last 1.5 s
    of IK joint angles for shoulder1/shoulder2/elbow). The classifier predicts
    the stroke type. Pressing 'C' (or auto-trigger on confident stroke)
    captures the buffered swing into a library, keyed by predicted class.

  PHASE 2 (AUTO_PLAY)
    Press 1/2/3 to choose the next stroke: FH topspin / BH drive / FH smash.
    Press SPACE: ball is served with a trajectory matched to that stroke,
    and the latest captured swing of that class is replayed so the paddle's
    peak-velocity frame lands at the predicted ball-in-contact-zone time.

Save / load library to disk with 'S' / 'O'.

Viewer keys
    L  → LIVE_RECORD     P  → AUTO_PLAY
    1  → FH topspin      2  → BH drive       3  → FH smash
    C  → capture latest swing now (LIVE_RECORD)
    SPACE → serve + replay (AUTO_PLAY)
    R  → reset ball only (no replay)
    S  → save library    O  → load library
    Esc → quit
"""
import sys
import time
import queue
import pickle
import asyncio
import threading
import pathlib
from collections import deque
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

import mujoco
import mujoco.viewer

# ── Paths ──────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent
MJ_ROOT    = HERE.parent
MODEL_PATH = MJ_ROOT / "models" / "humanoid_tt.xml"
ROOT       = MJ_ROOT.parent              # tt_coaching_pipeline
SRC_DIR    = ROOT / "src"
OUT_DIR    = MJ_ROOT / "output"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LIB_PATH   = OUT_DIR / f"swing_library_{timestamp}.pkl"

CKPT_PATH   = OUT_DIR / "model_synthetic.pt"
SCALER_PATH = OUT_DIR / "scaler_synthetic.pkl"

IMU_DIR    = pathlib.Path(r"E:\thesis_work\TT_thesis\imu_data")
TELEOP_DIR = pathlib.Path(r"E:\thesis_work\TT_thesis\SiriusCeption-UserUI")

for p in (SRC_DIR, IMU_DIR, TELEOP_DIR):
    sys.path.insert(0, str(p))

from udp_imu_server       import UDPIMUServer
from SiriusCeption_Teleop import SiriusCeptionTeleop
from inference            import StrokePredictor, CLASS_NAMES
from feature_extractor    import (
    SlidingWindowExtractor, window_energy, is_idle_window, RACKET_IMU_ID,
)

# ── Config ─────────────────────────────────────────────────────────────
UDP_PORT      = 9999
SENSOR_HAND   = 7
SENSOR_FORE   = 8
SENSOR_UPPER  = 9
L_UPPER       = 0.28
L_FORE        = 0.28
CALIB_SECS    = 3.0
TARGET_FPS    = 60
COORD_SIGN    = np.array([1.0, 1.0, 1.0])   # flip a sign if an axis is inverted (Z = up/down)

BUFFER_SECONDS    = 1.5

# Stroke-peak detection (same gate live_app_789.py uses to populate "Shot
# Predictions"). A stroke fires only at a local maximum of window-energy
# above LIVE_PEAK_MIN_HEIGHT, with a refractory period of PEAK_MIN_DISTANCE
# windows. Everything else (arm motion, return swing, idle) is ignored.
LIVE_PEAK_MIN_HEIGHT = 40.0
PEAK_MIN_DISTANCE    = 10

# Per-stroke profiles for AUTO_PLAY mode.
# serve_pos / serve_vel are only the INITIAL GUESS — when a replay is
# triggered, the solver simulates the ball with the real physics and
# iteratively adjusts serve_vel so the ball passes through the paddle's
# position at the swing's peak frame. That guarantees bat-ball contact
# regardless of where the recorded swing actually travels.
STROKE_PROFILES = {
    "fh_topspin": dict(
        serve_pos = np.array([3.00, -0.20, 1.10]),
        serve_vel = np.array([-3.5,  0.00, 1.8]),
    ),
    "bh_drive": dict(
        serve_pos = np.array([3.00,  0.20, 1.10]),
        serve_vel = np.array([-3.5,  0.00, 1.8]),
    ),
    "fh_smash": dict(
        serve_pos = np.array([3.00, -0.20, 1.40]),
        serve_vel = np.array([-3.0,  0.00, 3.0]),
    ),
}

READY_POSE       = np.array([0.0, 0.0, -0.5])   # arm stance between replays
SOLVER_TOL       = 0.02     # accept solution when ball passes within 2 cm
SOLVER_MAX_ITERS = 25
SERVE_LEAD       = 0.3      # extra seconds before the action starts

# Capture shaping: wait a beat after the stroke is detected so the buffer
# contains the follow-through (detection fires ~at the energy peak, i.e. the
# hit), then trim the recording to the active part of the swing. The trim is
# deliberately gentle — the wind-up matters for finding hittable contact
# frames, so keep generous padding around the fast segment.
CAPTURE_DELAY    = 0.35     # s between stroke detection and buffer snapshot
TRIM_SPEED_FRAC  = 0.08     # keep frames with paddle speed ≥ 8% of max
TRIM_PAD_SECONDS = 0.40     # padding kept on each side of the active segment

# After a replay finishes, hold the final pose briefly, then blend slowly
# back to the ready stance. An instant snap back sweeps the paddle through
# the ball's return path and smacks it a second time.
RETURN_HOLD_SECS  = 0.4
RETURN_BLEND_SECS = 1.0

# Map classifier output label_id → our internal stroke key.
# TTSWING/synthetic order: 0=No Stroke, 1=Topspin, 2=Backhand, 3=Smash.
CLASS_ID_TO_KEY = {
    1: "fh_topspin",
    2: "bh_drive",
    3: "fh_smash",
}

MODE_LIVE_RECORD = "LIVE_RECORD"
MODE_AUTO_PLAY   = "AUTO_PLAY"


# ── Peak-based stroke counter (lifted verbatim from live_app_789.py) ──

class OnlineStrokeCounter:
    """Fires exactly once per local energy maximum that clears min_height,
    with a refractory of `min_distance` windows. Mirrors the gate that
    populates the "Shot Predictions" panel in live_app_789.py."""

    def __init__(self, min_distance=PEAK_MIN_DISTANCE,
                 min_height=LIVE_PEAK_MIN_HEIGHT):
        self.min_distance    = min_distance
        self.min_height      = min_height
        self.e_prev2 = None; self.e_prev1 = None
        self.lab_prev1 = 0;  self.conf_prev1 = 0.0
        self.idx = 0
        self.last_stroke_idx = -10**9

    def update(self, energy: float, label_id: int, confidence: float):
        fired = None
        is_local_max = (self.e_prev2 is not None and self.e_prev1 is not None
                        and self.e_prev2 < self.e_prev1 >= energy)
        if (is_local_max
                and self.e_prev1 >= self.min_height
                and (self.idx - 1) - self.last_stroke_idx >= self.min_distance
                and self.lab_prev1 != 0):
            self.last_stroke_idx = self.idx - 1
            fired = (self.lab_prev1, self.conf_prev1)
        self.e_prev2, self.e_prev1 = self.e_prev1, energy
        self.lab_prev1, self.conf_prev1 = label_id, confidence
        self.idx += 1
        return fired


# ── Background thread: UDPIMUServer + SiriusCeptionTeleop ─────────────

class TeleopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt    = threading.Event()
        self.calibrated   = threading.Event()
        self.sensors_seen = threading.Event()
        self.teleop = None
        self.server = None

    def stop(self): self._stop_evt.set()

    def run(self):
        try:
            asyncio.run(self._async_main())
        except Exception:
            import traceback; traceback.print_exc()

    async def _async_main(self):
        self.server = UDPIMUServer(port=UDP_PORT)
        self.server.run()
        await asyncio.sleep(0.4)
        print(f"[teleop] Waiting for sensors {SENSOR_HAND}/{SENSOR_FORE}/{SENSOR_UPPER}…")
        while not self._stop_evt.is_set():
            if all(s in self.server.get_latest_data()
                   for s in (SENSOR_HAND, SENSOR_FORE, SENSOR_UPPER)):
                break
            await asyncio.sleep(0.1)
        if self._stop_evt.is_set(): return
        print("[teleop] All 3 sensors online.")
        self.sensors_seen.set()

        self.teleop = SiriusCeptionTeleop(
            imu_server=self.server,
            upper_id=SENSOR_UPPER, fore_id=SENSOR_FORE, hand_id=SENSOR_HAND,
            l_upper=L_UPPER, l_fore=L_FORE,
            init_pose="forward", earth_frame="SEU",
        )
        print(f"[teleop] STAND IN T-POSE. Calibrating for {CALIB_SECS:.0f}s…")
        await self.teleop.calibrate(duration=CALIB_SECS)
        print("[teleop] Calibration complete.")
        self.calibrated.set()

        while not self._stop_evt.is_set():
            await asyncio.sleep(0.05)
        await self.server.stop()


# ── Classifier thread (mirrors live_app_789's LiveIMUStreamer pipeline) ─
#
# Polls the UDP server at 3× the sensor rate and feeds the extractor ONLY
# when the accel+gyro signature changes — i.e. one row per NEW 100 Hz
# packet, never a duplicated hold. Without this dedup the window sees
# step-jump artifacts and fires phantom strokes (the "everything is a
# smash" bug). Detected strokes are queued for the main loop to consume.

SAMPLE_RATE = 100   # Hz — must match the sensor's udp_hz

class ClassifierThread(threading.Thread):
    def __init__(self, server, predictor, stroke_queue):
        super().__init__(daemon=True)
        self.server     = server
        self.predictor  = predictor
        self.extractor  = SlidingWindowExtractor()
        self.counter    = OnlineStrokeCounter()
        self.out_q      = stroke_queue          # queue of (stroke_key, conf, t)
        self.enabled    = threading.Event()     # only classify in LIVE_RECORD
        self._stop_evt  = threading.Event()

    def stop(self): self._stop_evt.set()

    @staticmethod
    def _build_row(d) -> dict:
        a = d.get("accel", (0.0, 0.0, 0.0))
        g = d.get("gyro",  (0.0, 0.0, 0.0))
        q = d.get("quat",  (1.0, 0.0, 0.0, 0.0))
        p = f"imu_{RACKET_IMU_ID}_"
        return {
            p + "accel_x": a[0], p + "accel_y": a[1], p + "accel_z": a[2],
            p + "gyro_x":  g[0], p + "gyro_y":  g[1], p + "gyro_z":  g[2],
            p + "quat_w":  q[0], p + "quat_x":  q[1],
            p + "quat_y":  q[2], p + "quat_z":  q[3],
        }

    def run(self):
        dt = 1.0 / (SAMPLE_RATE * 3)
        last_signature = None
        while not self._stop_evt.is_set():
            if not self.enabled.is_set():
                time.sleep(0.05)
                continue
            latest = self.server.get_latest_data()
            d = latest.get(SENSOR_HAND)
            if d is not None:
                sig = tuple(d["accel"]) + tuple(d["gyro"])
                if sig != last_signature:
                    last_signature = sig
                    feats = self.extractor.add_frame(self._build_row(d))
                    if feats is not None:
                        energy = window_energy(feats)
                        if is_idle_window(feats):
                            lid, conf = 0, 0.0
                        else:
                            pred = self.predictor.predict(feats)
                            lid, conf = pred["label_id"], pred["confidence"]
                        fired = self.counter.update(energy, lid, conf)
                        if fired is not None:
                            f_lid, f_conf = fired
                            if f_lid in CLASS_ID_TO_KEY:
                                key = CLASS_ID_TO_KEY[f_lid]
                                print(f"[stroke] {key}  conf={f_conf:.2f}  energy={energy:.0f}")
                                self.out_q.put((key, f_conf, time.time()))
            time.sleep(dt)


# ── Main ───────────────────────────────────────────────────────────────

HELP_TEXT = """
╔══════════════════════════ CONTROLS (Ctrl + key) ═════════════════════════╗
║  All commands need Ctrl held (avoids MuJoCo's own viewer shortcuts —     ║
║  plain '1' hides the skeleton, plain 'S' toggles shadows, etc.)          ║
║                                                                          ║
║  Ctrl+L  LIVE_RECORD mode — swing; detected strokes are auto-captured    ║
║  Ctrl+P  AUTO_PLAY mode  — avatar replays captured swings                ║
║  Ctrl+1  select FH topspin   Ctrl+2  select BH drive  Ctrl+3  select smash ║
║  SPACE   (AUTO_PLAY) serve ball + replay selected stroke                 ║
║  Ctrl+C  (LIVE_RECORD) capture last swing manually                       ║
║  Ctrl+R  reset ball only                                                 ║
║  Ctrl+S  save swing library to disk     Ctrl+O  load library from disk   ║
║  Esc     quit                                                            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def main():
    print(HELP_TEXT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    def jaddr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    s1_q, s1_v = jaddr("shoulder1_right")
    s2_q, s2_v = jaddr("shoulder2_right")
    el_q, el_v = jaddr("elbow_right")
    ball_q, ball_v = jaddr("ball_root")

    bid_upper = bid("upper_arm_right")
    bid_lower = bid("lower_arm_right")
    bid_hand  = bid("hand_right")

    gid_blade  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
    gid_handle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_handle")
    gid_ball   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    gid_table  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

    # Opponent court (for judging where a return lands): far side of the net.
    NET_X       = 1.70
    TABLE_X_FAR = 3.07
    TABLE_Y     = 0.7625

    # Scratch MjData for ball-trajectory rollouts (never touches the live data)
    data_sim = mujoco.MjData(model)

    # Stand pose
    data.qpos[2] = 1.282
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    stand_qpos = data.qpos.copy()

    # Park the ball below the floor until we serve
    def park_ball():
        data.qpos[ball_q:ball_q+3]   = [0.0, 0.0, -5.0]
        data.qpos[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[ball_v:ball_v+6]   = 0.0
        stand_qpos[ball_q:ball_q+3]   = [0.0, 0.0, -5.0]
        stand_qpos[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]

    park_ball()
    mujoco.mj_forward(model, data)

    # ── Load classifier ────────────────────────────────────────────────
    try:
        predictor = StrokePredictor.from_checkpoint(str(CKPT_PATH), str(SCALER_PATH))
        print(f"[classifier] loaded {CKPT_PATH.name}")
    except Exception as e:
        print(f"[classifier] DISABLED ({e})")
        predictor = None

    # ── Start IMU/teleop ───────────────────────────────────────────────
    th = TeleopThread(); th.start()
    print("[main] Waiting for sensors + calibration…")
    th.sensors_seen.wait()
    th.calibrated.wait()
    print("[main] Teleop ready. Opening viewer.")

    # ── Start classifier thread (live_app_789-style dedup pipeline) ────
    stroke_q: queue.Queue = queue.Queue()
    clf_thread = None
    if predictor is not None:
        clf_thread = ClassifierThread(th.server, predictor, stroke_q)
        clf_thread.enabled.set()        # LIVE_RECORD is the starting mode
        clf_thread.start()

    # ── IK cost ────────────────────────────────────────────────────────
    def ik_cost(qv, t_se, t_ew):
        data.qpos[s1_q] = qv[0]
        data.qpos[s2_q] = qv[1]
        data.qpos[el_q] = qv[2]
        mujoco.mj_forward(model, data)
        cur_se = data.xpos[bid_lower] - data.xpos[bid_upper]
        cur_ew = data.xpos[bid_hand]  - data.xpos[bid_lower]
        return float(np.sum((cur_se - t_se)**2) + np.sum((cur_ew - t_ew)**2))

    # ── State ──────────────────────────────────────────────────────────
    state = {
        "mode": MODE_LIVE_RECORD,
        "next_stroke": "fh_topspin",
        "library": {k: [] for k in STROKE_PROFILES},     # key → list of (q_seq, peak_idx)
        "last_pred_key": None,
        "last_pred_conf": 0.0,
        "last_capture_t": 0.0,
        "pending_capture": None,    # (key, due_time) — delayed buffer snapshot
        # replay — ALL scheduling is in render-frame counts (sim time), not
        # wall-clock seconds: the ball advances exactly dt_frame of sim time
        # per render iteration, so frame counts keep paddle and ball in sync
        # even when the render loop hiccups.
        "want_replay":  False,      # set by SPACE (viewer thread) → handled in main loop
        "auto_k":       0,          # render-frame counter while in AUTO_PLAY
        "serve_k":      None,       # frame at which to fire the serve
        "serve_vel":    None,
        "serve_key":    None,
        "replay_k0":    None,       # frame at which replay frame 0 plays
        "replay_seq":   None,
        "replay_active": False,
        "min_dist":     None,       # closest ball-blade approach this rally
        "rally_hit":    False,      # real ball-paddle contact seen this rally
        "rally_land":   None,       # where the return landed (x, y)
        "return_k":     None,       # frame the post-swing hold/blend started
        "return_from":  None,       # pose to blend back to ready from
        "ctrl_t":       0.0,        # last time a Ctrl key-down was seen
    }

    dt_frame   = 1.0 / TARGET_FPS
    dt_phys    = model.opt.timestep
    phys_steps = max(1, int(round(dt_frame / dt_phys)))
    buffer_len = int(BUFFER_SECONDS * TARGET_FPS)

    # ring buffer of joint angles (LIVE_RECORD only)
    joint_buffer: deque = deque(maxlen=buffer_len)

    q_warm     = np.array([0.0, 0.0, -0.5])
    q_prev_arm = q_warm.copy()

    def find_peak_frame(q_seq: np.ndarray) -> int:
        """Frame index of maximum forward wrist velocity (contact instant)."""
        n = len(q_seq)
        wx = np.zeros(n)
        for i, q in enumerate(q_seq):
            fk_data.qpos[:] = stand_qpos
            fk_data.qpos[s1_q] = q[0]
            fk_data.qpos[s2_q] = q[1]
            fk_data.qpos[el_q] = q[2]
            mujoco.mj_forward(model, fk_data)
            wx[i] = fk_data.xpos[bid_hand][0]
        if n < 2:
            return 0
        vel = np.diff(wx)
        return int(np.argmax(vel))

    # Hittable window: the contact frame must put the paddle somewhere a
    # served ball can physically reach (in front of the player, inside the
    # table width, above table-top height).
    HIT_X = (0.30, 1.00)
    HIT_Y = (-0.75, 0.75)
    HIT_Z = (0.85, 1.85)

    def contact_candidates(q_seq: np.ndarray):
        """Rank swing frames as contact candidates: paddle inside the
        hittable window AND moving forward (toward the opponent). Striking
        while the paddle travels +x is what sends the return over the net —
        a fast frame where the paddle moves down/sideways just clips the
        ball off its arc. Ranked by forward velocity, fastest-forward first.
        Returns (positions, indices)."""
        P = np.array([paddle_pos_at(q) for q in q_seq])
        if len(P) < 3:
            return P, []
        vel   = np.diff(P, axis=0)                    # per-frame displacement
        speed = np.linalg.norm(vel, axis=1)
        smax  = float(speed.max())
        cands = [
            i for i in range(1, len(q_seq) - 1)
            if speed[i - 1] >= 0.30 * smax
            and vel[i - 1][0] >= 0.5 * speed[i - 1]   # dominantly forward
            and HIT_X[0] <= P[i][0] <= HIT_X[1]
            and HIT_Y[0] <= P[i][1] <= HIT_Y[1]
            and HIT_Z[0] <= P[i][2] <= HIT_Z[1]
        ]
        cands.sort(key=lambda i: -vel[i - 1][0])      # most forward first
        if not cands:
            # Fallback: no dominantly-forward frame — accept any forward
            # motion at all rather than refusing the swing outright.
            cands = [
                i for i in range(1, len(q_seq) - 1)
                if speed[i - 1] >= 0.30 * smax
                and vel[i - 1][0] > 0.0
                and HIT_X[0] <= P[i][0] <= HIT_X[1]
                and HIT_Y[0] <= P[i][1] <= HIT_Y[1]
                and HIT_Z[0] <= P[i][2] <= HIT_Z[1]
            ]
            cands.sort(key=lambda i: -vel[i - 1][0])
        return P, cands

    # ── Serve solver: aim the ball at the swing's contact point ────────
    scratch = mujoco.MjData(model)
    fk_data = mujoco.MjData(model)      # FK scratch — never touch live `data`

    def paddle_pos_at(qarm) -> np.ndarray:
        """FK: world position of the paddle blade for given arm angles."""
        fk_data.qpos[:] = stand_qpos
        fk_data.qpos[s1_q], fk_data.qpos[s2_q], fk_data.qpos[el_q] = qarm
        mujoco.mj_forward(model, fk_data)
        return fk_data.geom_xpos[gid_blade].copy()

    def simulate_ball(serve_pos, serve_vel, t_max=2.5):
        """Roll the ball forward with the real physics (avatar frozen in
        ready stance, paddle contacts disabled) and return [(t, pos), …]."""
        saved = (model.geom_contype[gid_blade],  model.geom_conaffinity[gid_blade],
                 model.geom_contype[gid_handle], model.geom_conaffinity[gid_handle])
        model.geom_contype[gid_blade]  = 0; model.geom_conaffinity[gid_blade]  = 0
        model.geom_contype[gid_handle] = 0; model.geom_conaffinity[gid_handle] = 0
        try:
            d = scratch
            d.qpos[:] = stand_qpos
            d.qpos[s1_q], d.qpos[s2_q], d.qpos[el_q] = READY_POSE
            d.qpos[ball_q:ball_q+3]   = serve_pos
            d.qpos[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]
            d.qvel[:] = 0.0
            d.qvel[ball_v:ball_v+3] = serve_vel
            traj = []
            for k in range(int(t_max / dt_phys)):
                bp = d.qpos[ball_q:ball_q+7].copy()
                bv = d.qvel[ball_v:ball_v+6].copy()
                d.qpos[:] = stand_qpos
                d.qpos[s1_q], d.qpos[s2_q], d.qpos[el_q] = READY_POSE
                d.qpos[ball_q:ball_q+7] = bp
                d.qvel[:] = 0.0
                d.qvel[ball_v:ball_v+6] = bv
                mujoco.mj_step(model, d)
                traj.append(((k + 1) * dt_phys, d.qpos[ball_q:ball_q+3].copy()))
                if d.qpos[ball_q + 2] < 0.05:
                    break
            return traj
        finally:
            (model.geom_contype[gid_blade],  model.geom_conaffinity[gid_blade],
             model.geom_contype[gid_handle], model.geom_conaffinity[gid_handle]) = saved

    trial_data = mujoco.MjData(model)

    def coupled_trial(q_seq, vel, key, serve_rel, replay_rel, n_iters):
        """Replicate the live AUTO_PLAY loop exactly (swing replay + ball,
        paddle contacts ON): serve fires at frame serve_rel, replay frame 0
        plays at frame replay_rel.
        Returns (hit_frame, outgoing_vel, min_dist, landing_pos) where
        landing_pos is the ball's first table contact AFTER the hit (None if
        the return never lands on the table).

        This catches what the aiming solver can't see: the paddle's wind-up
        sweeping through the ball's incoming corridor and clipping it early.
        """
        prof = STROKE_PROFILES[key]
        d = trial_data
        mujoco.mj_resetData(model, d)
        local_stand = stand_qpos.copy()
        local_stand[ball_q:ball_q+3]   = [0.0, 0.0, -5.0]
        local_stand[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[:] = local_stand
        d.qvel[:] = 0.0
        qw = q_seq[0].copy()
        qp = qw.copy()
        served    = False
        hit_frame = None
        out_vel   = None
        min_dist  = 1e9
        land_pos  = None
        for k in range(n_iters):
            if not served and k >= serve_rel:
                d.qpos[ball_q:ball_q+3]   = prof["serve_pos"]
                d.qpos[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]
                d.qvel[ball_v:ball_v+3]   = vel
                d.qvel[ball_v+3:ball_v+6] = 0.0
                local_stand[ball_q:ball_q+3] = prof["serve_pos"]
                served = True
            fi = k - replay_rel
            if fi < 0:
                qw = q_seq[0]
            elif fi < len(q_seq):
                qw = q_seq[fi]
            else:
                qw = q_seq[-1]
            rate = (qw - qp) / dt_frame
            for kk in range(phys_steps):
                alpha = (kk + 1) / phys_steps
                qs = qp + (qw - qp) * alpha
                bp = d.qpos[ball_q:ball_q+7].copy()
                bv = d.qvel[ball_v:ball_v+6].copy()
                d.qpos[:] = local_stand
                d.qpos[s1_q], d.qpos[s2_q], d.qpos[el_q] = qs
                d.qpos[ball_q:ball_q+7] = bp
                d.qvel[:] = 0.0
                d.qvel[s1_v], d.qvel[s2_v], d.qvel[el_v] = rate
                d.qvel[ball_v:ball_v+6] = bv
                mujoco.mj_step(model, d)
                if served:
                    dist = float(np.linalg.norm(
                        d.geom_xpos[gid_ball] - d.geom_xpos[gid_blade]))
                    min_dist = min(min_dist, dist)
                if hit_frame is None:
                    for ci in range(d.ncon):
                        g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                        if gid_ball in (g1, g2) and (gid_blade in (g1, g2)
                                                     or gid_handle in (g1, g2)):
                            hit_frame = k
                            break
                else:
                    if out_vel is None and k >= hit_frame + 3:
                        out_vel = d.qvel[ball_v:ball_v+3].copy()
                    # First table contact after the hit = where the return lands
                    if land_pos is None and k >= hit_frame + 3:
                        for ci in range(d.ncon):
                            g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                            if gid_ball in (g1, g2) and gid_table in (g1, g2):
                                land_pos = d.geom_xpos[gid_ball][:2].copy()
                                break
            qp = qw.copy()
            if land_pos is not None:
                break
        if hit_frame is not None and out_vel is None:
            out_vel = d.qvel[ball_v:ball_v+3].copy()
        return hit_frame, out_vel, min_dist, land_pos

    def _plane_crossing(traj, p_star):
        """Time + position where the simulated ball crosses x = p_star.x."""
        for (t1, p1), (t2, p2) in zip(traj[:-1], traj[1:]):
            if p1[0] >= p_star[0] >= p2[0]:
                a = (p1[0] - p_star[0]) / max(p1[0] - p2[0], 1e-9)
                return t1 + a * (t2 - t1), p1 + a * (p2 - p1)
        return None, None

    def solve_serve(key: str, p_star: np.ndarray):
        """Shooting method with per-axis secant updates: adjusts the serve
        velocity until the simulated ball (real physics, incl. table bounce)
        passes through p_star. Returns (serve_vel, time_of_arrival) or
        (None, None) if it can't converge."""
        prof = STROKE_PROFILES[key]
        S = prof["serve_pos"].astype(float).copy()
        v = prof["serve_vel"].astype(float).copy()
        prev_y = prev_z = None      # (velocity, error) pairs for secant slope
        ey = ez = 99.0
        for it in range(SOLVER_MAX_ITERS):
            t_star, p_sim = _plane_crossing(simulate_ball(S, v), p_star)
            if t_star is None:
                v[0] *= 1.2        # ball never reached the plane → serve faster
                continue
            ey = p_sim[1] - p_star[1]
            ez = p_sim[2] - p_star[2]
            if abs(ey) < SOLVER_TOL and abs(ez) < SOLVER_TOL:
                print(f"[solver] converged in {it+1} iters: vel={np.round(v,2)}, "
                      f"arrives t={t_star:.2f}s, miss=({ey*100:.1f},{ez*100:.1f})cm")
                return v, t_star
            # Secant slope per axis; first iteration falls back to the
            # ballistic sensitivity (dz/dv ≈ flight time).
            if prev_y is not None and abs(v[1]-prev_y[0]) > 1e-6 and abs(ey-prev_y[1]) > 1e-9:
                slope_y = (ey - prev_y[1]) / (v[1] - prev_y[0])
            else:
                slope_y = t_star
            if prev_z is not None and abs(v[2]-prev_z[0]) > 1e-6 and abs(ez-prev_z[1]) > 1e-9:
                slope_z = (ez - prev_z[1]) / (v[2] - prev_z[0])
            else:
                slope_z = t_star
            prev_y = (v[1], ey)
            prev_z = (v[2], ez)
            v[1] -= float(np.clip(ey / slope_y, -1.5, 1.5))
            v[2] -= float(np.clip(ez / slope_z, -1.5, 1.5))
        print(f"[solver] FAILED to converge (last miss y={ey*100:.1f}cm z={ez*100:.1f}cm). "
              f"Contact point may be unreachable from the serve position.")
        return None, None

    def trim_swing(q_seq: np.ndarray) -> np.ndarray:
        """Cut the dead lead-in/lead-out: keep only the contiguous segment
        around the paddle-speed maximum where speed ≥ 15% of max, padded by
        0.2 s each side. Without this, a 1.5 s buffer whose stroke sits at
        the very end (e.g. a smash detected late) replays ~1.3 s of slow
        arm drift before the actual hit — the avatar 'swings' long before
        the ball arrives."""
        n = len(q_seq)
        if n < 5:
            return q_seq
        P = np.array([paddle_pos_at(q) for q in q_seq])
        speed = np.linalg.norm(np.diff(P, axis=0), axis=1)      # len n-1
        smax = float(speed.max())
        if smax < 1e-6:
            return q_seq
        k_peak  = int(np.argmax(speed))
        thresh  = TRIM_SPEED_FRAC * smax
        i0 = k_peak
        while i0 > 0 and speed[i0 - 1] >= thresh:
            i0 -= 1
        i1 = k_peak
        while i1 < len(speed) - 1 and speed[i1 + 1] >= thresh:
            i1 += 1
        pad = int(TRIM_PAD_SECONDS * TARGET_FPS)
        a = max(0, i0 - pad)
        b = min(n, i1 + 2 + pad)
        return q_seq[a:b]

    def capture_swing(label_key: str):
        if len(joint_buffer) < 10:
            print("[capture] buffer too short")
            return
        q_seq = np.array([row[1:] for row in joint_buffer])     # drop timestamp col
        raw_len = len(q_seq)
        q_seq = trim_swing(q_seq)
        peak = find_peak_frame(q_seq)
        state["library"][label_key].append((q_seq, peak))
        n = len(state["library"][label_key])
        print(f"[capture] saved as {label_key} "
              f"(trimmed {raw_len}→{len(q_seq)}, peak={peak}) — total {n}")

    def reset_ball(profile_key: str | None = None, vel: np.ndarray | None = None):
        if profile_key is None:
            profile_key = state["next_stroke"]
        prof = STROKE_PROFILES[profile_key]
        if vel is None:
            vel = prof["serve_vel"]
        data.qpos[ball_q:ball_q+3]   = prof["serve_pos"]
        data.qpos[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[ball_v:ball_v+3]   = vel
        data.qvel[ball_v+3:ball_v+6] = 0.0
        stand_qpos[ball_q:ball_q+3]   = prof["serve_pos"]
        stand_qpos[ball_q+3:ball_q+7] = [1.0, 0.0, 0.0, 0.0]

    def trigger_replay():
        key = state["next_stroke"]
        if not state["library"][key]:
            print(f"[play] no recordings for {key}; capture some first")
            return
        q_seq, _ = state["library"][key][-1]

        # 1. Rank hittable contact frames (paddle in front, over the table,
        #    moving FORWARD) — these are the moments the swing can actually
        #    drive the ball toward the opponent.
        P, cands = contact_candidates(q_seq)
        if not cands:
            print(f"[play] {key}: no forward-moving frame of this swing passes "
                  f"through the hittable window — capture the stroke again")
            return

        # 2.+3. Joint search over (contact candidate × timing offset).
        #    For each candidate: aim the serve at it with the shooting
        #    solver, then run the coupled validation (swing + ball, contacts
        #    ON, +90 frames so the return landing is visible) across timing
        #    offsets. Rank: return lands in opponent court > any contact;
        #    within a tier prefer the straightest forward return. Early-exit
        #    on the first candidate that produces an on-table return.
        margin = 8
        best = None   # (score, vel, serve_rel, replay_rel0, delta, hf, ov, lp)
        for idx in cands[:4]:
            p_try = P[idx]
            print(f"[play] {key}: trying contact frame {idx} at {np.round(p_try, 2)}")
            vel, t_star = solve_serve(key, p_try)
            if vel is None:
                continue

            n_flight  = int(round(t_star / dt_frame))
            serve_rel = int(round(SERVE_LEAD / dt_frame)) + margin
            replay_rel0 = serve_rel - 1 + n_flight - idx
            if replay_rel0 < margin:            # swing longer than ball flight
                extra = margin - replay_rel0
                serve_rel   += extra
                replay_rel0 += extra
            n_iters = max(serve_rel + n_flight,
                          replay_rel0 + margin + len(q_seq)) + 90

            print("[play] validating contact (coupled simulation)…")
            for delta in range(-margin, margin + 1):
                hf, ov, md, lp = coupled_trial(q_seq, vel, key, serve_rel,
                                               replay_rel0 + delta, n_iters)
                if hf is None or ov is None:
                    continue
                on_table = (lp is not None
                            and NET_X + 0.05 <= lp[0] <= TABLE_X_FAR
                            and abs(lp[1]) <= TABLE_Y)
                speed = float(np.linalg.norm(ov)) + 1e-9
                straightness = float(ov[0]) / speed   # 1 = straight forward
                score = (1 if on_table else 0, straightness, float(ov[0]))
                if best is None or score > best[0]:
                    best = (score, vel, serve_rel, replay_rel0,
                            delta, hf, ov, lp)
            if best is not None and best[0][0] == 1:
                break    # on-table return found — stop searching candidates

        if best is None:
            print("[play] validation: NO contact for any candidate/offset — "
                  "capture this stroke again (swing may be too shallow)")
            return
        score, vel, serve_rel, replay_rel0, delta, hf, ov, lp = best
        where = (f"lands at x={lp[0]:.2f} y={lp[1]:+.2f} "
                 f"({'ON opponent court' if score[0] else 'OFF table'})"
                 if lp is not None else "return never lands on table")
        print(f"[play] validation: swing offset {delta:+d} frames → hit at "
              f"frame {hf}, return v={np.round(ov, 2)} m/s, {where}")

        k_now = state["auto_k"]
        state["serve_k"]       = k_now + serve_rel
        state["serve_vel"]     = vel
        state["serve_key"]     = key
        state["replay_k0"]     = k_now + replay_rel0 + delta
        state["replay_seq"]    = q_seq
        state["replay_active"] = True
        state["min_dist"]      = None
        state["rally_hit"]     = False
        state["rally_land"]    = None
        state["return_k"]      = None
        state["return_from"]   = None
        print(f"[play] serve at frame +{serve_rel}, contact ≈ frame +{hf}")

    def save_library():
        with open(LIB_PATH, "wb") as f:
            pickle.dump(state["library"], f)
        counts = {k: len(v) for k, v in state["library"].items()}
        print(f"[save] {LIB_PATH.name}  {counts}")

    def load_library():
        # Saves are timestamped per session — load the most recent one.
        candidates = sorted(OUT_DIR.glob("swing_library_*.pkl"))
        if not candidates:
            print(f"[load] no swing_library_*.pkl found in {OUT_DIR}")
            return
        path = candidates[-1]
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        for k, lst in loaded.items():
            if k in state["library"]:
                state["library"][k].extend(lst)
        counts = {k: len(v) for k, v in state["library"].items()}
        print(f"[load] {path.name}  {counts}")

    # All commands are Ctrl-chorded: MuJoCo's viewer has its own plain-key
    # shortcuts ('1' toggles geom group 1 = the whole skeleton vanishes,
    # 'S' toggles shadows, …) and the user callback cannot suppress them.
    # The callback only gets a bare keycode (no modifier state), but GLFW
    # delivers the Ctrl key-down itself (341 left / 345 right), so we
    # remember when Ctrl was last pressed and accept an action key within
    # a short window. While Ctrl is held, the viewer's built-in plain-key
    # bindings don't fire — no more vanishing skeleton / shadow toggles.
    CTRL_KEYS   = (341, 345)
    CTRL_WINDOW = 3.0      # seconds after Ctrl press in which a key counts

    def key_callback(keycode):
        if keycode in CTRL_KEYS:
            state["ctrl_t"] = time.time()
            return
        # SPACE stays plain — it has no conflicting viewer binding here.
        if keycode == 32:      # SPACE
            if state["mode"] == MODE_AUTO_PLAY:
                # Solver runs on the MAIN thread — running it here (viewer
                # thread) races the physics loop and corrupts mjData.
                state["want_replay"] = True
            return
        if time.time() - state.get("ctrl_t", 0.0) > CTRL_WINDOW:
            return             # not chorded → leave the key to the viewer

        if keycode == ord('L'):
            state["mode"] = MODE_LIVE_RECORD
            park_ball()
            if clf_thread is not None:
                clf_thread.enabled.set()
            print("[mode] LIVE_RECORD — swing to capture; classifier labels each swing")
        elif keycode == ord('P'):
            state["mode"] = MODE_AUTO_PLAY
            park_ball()
            state["auto_k"] = 0
            state["serve_k"] = None
            state["replay_active"] = False
            state["want_replay"] = False
            if clf_thread is not None:
                clf_thread.enabled.clear()
            print(f"[mode] AUTO_PLAY — next stroke: {state['next_stroke']}; SPACE to serve")
        elif keycode == ord('1'):
            state["next_stroke"] = "fh_topspin"; print("[select] fh_topspin")
        elif keycode == ord('2'):
            state["next_stroke"] = "bh_drive";   print("[select] bh_drive")
        elif keycode == ord('3'):
            state["next_stroke"] = "fh_smash";   print("[select] fh_smash")
        elif keycode == ord('C'):
            if state["mode"] == MODE_LIVE_RECORD:
                key = state["last_pred_key"] or state["next_stroke"]
                capture_swing(key)
        elif keycode == ord('R'):
            reset_ball()
        elif keycode == ord('S'):
            save_library()
        elif keycode == ord('O'):
            load_library()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            t0 = time.time()

            # ─────────────────────────── LIVE_RECORD ─────────────────────────
            if state["mode"] == MODE_LIVE_RECORD:
                try:
                    positions, _ = th.teleop.compute()
                except Exception:
                    viewer.sync(); time.sleep(dt_frame); continue

                shoulder = np.asarray(positions[0], float)
                elbow    = np.asarray(positions[1], float)
                wrist    = np.asarray(positions[2], float)
                se = elbow - shoulder; ew = wrist - elbow
                n_se = np.linalg.norm(se); n_ew = np.linalg.norm(ew)
                if n_se < 1e-6 or n_ew < 1e-6:
                    viewer.sync(); time.sleep(dt_frame); continue

                t_se = (se / n_se) * COORD_SIGN * L_UPPER
                t_ew = (ew / n_ew) * COORD_SIGN * L_FORE
                res = minimize(ik_cost, q_warm, args=(t_se, t_ew),
                               method="Nelder-Mead",
                               options={"maxiter": 30, "xatol": 1e-3, "fatol": 1e-4})
                q_warm = res.x

                # Buffer this frame
                joint_buffer.append((t0, q_warm[0], q_warm[1], q_warm[2]))

                # Drain stroke detections from the classifier thread.
                # A capture happens ONLY when OnlineStrokeCounter fired on a
                # real energy-peak stroke — exactly the same gate that drives
                # the "Shot Predictions" panel in live_app_789. Idle motion
                # and arm-waving never reach here. The snapshot itself is
                # DELAYED by CAPTURE_DELAY so the buffer includes the
                # follow-through, not just everything up to the hit.
                while not stroke_q.empty():
                    key, conf, _t = stroke_q.get_nowait()
                    state["last_pred_key"]  = key
                    state["last_pred_conf"] = conf
                    state["pending_capture"] = (key, t0 + CAPTURE_DELAY)
                if state.get("pending_capture") is not None:
                    pkey, due = state["pending_capture"]
                    if t0 >= due:
                        state["pending_capture"] = None
                        capture_swing(pkey)
                        state["last_capture_t"] = t0

                # Avatar kinematic update only (no ball physics in LIVE)
                data.qpos[:] = stand_qpos
                data.qpos[s1_q] = q_warm[0]
                data.qpos[s2_q] = q_warm[1]
                data.qpos[el_q] = q_warm[2]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

            # ─────────────────────────── AUTO_PLAY ───────────────────────────
            else:
                # SPACE pressed → run the solver here on the main thread
                if state["want_replay"]:
                    state["want_replay"] = False
                    if not state["replay_active"]:
                        trigger_replay()

                # Fire the serve on its exact scheduled frame
                if state["serve_k"] is not None and state["auto_k"] >= state["serve_k"]:
                    reset_ball(state["serve_key"], state["serve_vel"])
                    state["serve_k"] = None
                    print("[play] ball served")

                if state["replay_active"]:
                    frame_idx = state["auto_k"] - state["replay_k0"]
                    q_seq = state["replay_seq"]
                    if frame_idx < 0:
                        q_warm = q_seq[0]
                    elif frame_idx < len(q_seq):
                        q_warm = q_seq[frame_idx]
                    else:
                        q_warm = q_seq[-1]
                        state["replay_active"] = False
                        state["return_k"]    = state["auto_k"]
                        state["return_from"] = q_seq[-1].copy()
                        if state["rally_hit"]:
                            lp = state["rally_land"]
                            if lp is not None:
                                ok = (NET_X + 0.05 <= lp[0] <= TABLE_X_FAR
                                      and abs(lp[1]) <= TABLE_Y)
                                print(f"[play] HIT — return landed at "
                                      f"x={lp[0]:.2f} y={lp[1]:+.2f} "
                                      f"({'IN' if ok else 'OUT'})")
                            else:
                                print("[play] HIT — return still in flight")
                        else:
                            md = state["min_dist"]
                            print(f"[play] MISS — closest ball-blade approach "
                                  f"{(md or 9.99)*100:.1f} cm")
                elif state["return_from"] is not None:
                    # Hold the follow-through pose, then ease back to ready —
                    # a snap-back sweeps the paddle through the ball's return
                    # path and hits it a second time.
                    kk      = state["auto_k"] - state["return_k"]
                    hold_f  = int(RETURN_HOLD_SECS  * TARGET_FPS)
                    blend_f = int(RETURN_BLEND_SECS * TARGET_FPS)
                    if kk < hold_f:
                        q_warm = state["return_from"]
                    elif kk < hold_f + blend_f:
                        a = (kk - hold_f) / blend_f
                        a = a * a * (3.0 - 2.0 * a)          # smoothstep
                        q_warm = (1.0 - a) * state["return_from"] + a * READY_POSE
                    else:
                        state["return_from"] = None
                        state["return_k"]    = None
                        q_warm = READY_POSE.copy()
                else:
                    q_warm = READY_POSE.copy()  # ready stance

                # Physics step ball, kinematic avatar. The arm SWEEPS through
                # the substeps (interpolated from last frame's pose) instead of
                # holding one pose — without this the paddle teleports ~10 cm
                # per render frame near the swing peak and the ball tunnels
                # straight through the gap. qvel is the true sweep rate
                # (per render frame, NOT per physics step).
                arm_rate = (q_warm - q_prev_arm) / dt_frame
                for k in range(phys_steps):
                    alpha = (k + 1) / phys_steps
                    q_sub = q_prev_arm + (q_warm - q_prev_arm) * alpha
                    ball_pos = data.qpos[ball_q:ball_q+7].copy()
                    ball_vel = data.qvel[ball_v:ball_v+6].copy()
                    data.qpos[:] = stand_qpos
                    data.qpos[s1_q] = q_sub[0]
                    data.qpos[s2_q] = q_sub[1]
                    data.qpos[el_q] = q_sub[2]
                    data.qpos[ball_q:ball_q+7] = ball_pos
                    data.qvel[:] = 0.0
                    data.qvel[s1_v] = arm_rate[0]
                    data.qvel[s2_v] = arm_rate[1]
                    data.qvel[el_v] = arm_rate[2]
                    data.qvel[ball_v:ball_v+6] = ball_vel
                    mujoco.mj_step(model, data)

                    # Track real contacts + closest approach for the report
                    if state["replay_active"] or state["rally_hit"]:
                        dist = float(np.linalg.norm(
                            data.geom_xpos[gid_ball] - data.geom_xpos[gid_blade]))
                        if state["min_dist"] is None or dist < state["min_dist"]:
                            state["min_dist"] = dist
                        for ci in range(data.ncon):
                            g1 = data.contact[ci].geom1
                            g2 = data.contact[ci].geom2
                            if gid_ball not in (g1, g2):
                                continue
                            if not state["rally_hit"] and (gid_blade in (g1, g2) or gid_handle in (g1, g2)):
                                state["rally_hit"] = True
                            elif (state["rally_hit"] and state["rally_land"] is None and gid_table in (g1, g2)):
                                state["rally_land"] = data.geom_xpos[gid_ball][:2].copy()

                state["auto_k"] += 1

            q_prev_arm = q_warm.copy()

            viewer.sync()
            elapsed = time.time() - t0
            if elapsed < dt_frame:
                time.sleep(dt_frame - elapsed)

    if clf_thread is not None:
        clf_thread.stop()
    th.stop()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
