"""
live_app_mujoco.py
──────────────────
Live coaching app that runs the classification GUI AND a side-by-side
MuJoCo 3-D viewer driven by the same SiriusCeption sensors. Replaces
Unity. Same single UDP server feeds both consumers (no port conflict).

Pipeline:
    SiriusCeption sensor (id=7) → WiFi UDP :9999
        → Zuyan's UDPIMUServer (async background thread)
        → poll latest packet @ SAMPLE_RATE Hz → feature row
        → SlidingWindowExtractor → energy gate → StrokePredictor
        → OnlineStrokeCounter (one count per motion-energy peak)
        → live Tkinter dashboard + on-demand LLM report

Imports only from existing modules; modifies none.

Run:
    python live_app_nounity.py
"""
import sys
import time
import queue
import asyncio
import pathlib
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

import numpy as np
import pandas as pd
import yaml

# MuJoCo teleop deps
import mujoco
import mujoco.viewer
from scipy.optimize import minimize

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR  = pathlib.Path(__file__).resolve().parent
ROOT     = SRC_DIR.parent
IMU_DATA_DIR = pathlib.Path(r"E:\thesis_work\TT_thesis\imu_data")
TELEOP_DIR   = pathlib.Path(r"E:\thesis_work\TT_thesis\SiriusCeption-UserUI")
MJ_MODEL     = ROOT / "mujoco_sim" / "models" / "humanoid.xml"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(IMU_DATA_DIR))
sys.path.insert(0, str(TELEOP_DIR))

from SiriusCeption_Teleop import SiriusCeptionTeleop

from udp_imu_server   import UDPIMUServer
from inference        import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from feature_extractor import (
    SlidingWindowExtractor, WINDOW_SIZE, STEP_SIZE,
    is_idle_window, window_energy,
)
from summarizer       import run_session, PEAK_MIN_DISTANCE, PEAK_MIN_HEIGHT
from coaching         import get_coaching_feedback

# ── Config ───────────────────────────────────────────────────────────────────
SUBJECT_ID    = 10
UDP_PORT      = 9999
RACKET_ID     = 7            # RightHand sensor
SAMPLE_RATE   = 100          # Hz — must match the sensor's udp_hz
MODEL_PATH    = ROOT / "mujoco_sim" / "output" / "model_synthetic.pt"
SCALER_PATH   = ROOT / "mujoco_sim" / "output" / "scaler_synthetic.pkl"

# ── MuJoCo teleop config ─────────────────────────────────────────────────────
SENSOR_HAND   = 7
SENSOR_FORE   = 8
SENSOR_UPPER  = 9
L_UPPER       = 0.28
L_FORE        = 0.28
MJ_CALIB_SECS = 3.0
MJ_FPS        = 30
COORD_SIGN    = np.array([1.0, 1.0, -1.0])    # phase-2 fix that worked

from ui_theme import (
    COLORS, CLASS_COLORS, SHORT_NAMES, FONTS,
    make_card, section_title, make_button,
    draw_progress_bar, style_header, style_footer,
)

# Return-swing rejection: a real swing peaks at energy 140+, while the
# "bringing arm back to ready" motion peaks at ~20-30. A modest absolute
# floor separates them without dropping genuinely soft real swings.
LIVE_PEAK_MIN_HEIGHT = 40.0
PEAK_REL_FRAC        = 0.0   # 0 = relative-to-median gate OFF
TPOSE_SECONDS        = 10    # hold-T-pose countdown before classification starts

# DEBUG: when True, every raw row received from the sensor is appended to a
# CSV in recordings/, so a live session can be re-run through the offline path
# and compared. Set False for normal use.
DEBUG_DUMP_ROWS = True


# ── Live UDP streamer (background thread + asyncio loop) ───────────────────────

class LiveIMUStreamer(threading.Thread):
    """
    Runs Zuyan's UDPIMUServer inside its own asyncio loop and polls the
    latest packet for RACKET_ID at SAMPLE_RATE Hz, pushing feature rows
    into the event queue. No Unity controller in this build.
    """

    def __init__(self, event_queue: queue.Queue,
                 racket_id: int = RACKET_ID, rate_hz: int = SAMPLE_RATE):
        super().__init__(daemon=True)
        self.q          = event_queue
        self.racket_id  = racket_id
        self.rate_hz    = rate_hz
        self._stop      = threading.Event()
        self._server    = None
        self._loop      = None       # exposed so MuJoCo thread can run teleop.calibrate()

    def stop(self):
        self._stop.set()

    @staticmethod
    def _build_row(d: dict) -> dict:
        """Map a single sensor packet → CSV-style row (slot 7 only; that's all
        the feature extractor reads)."""
        q = d["quat"]; a = d["accel"]; g = d["gyro"]
        p = f"imu_{RACKET_ID}_"
        return {
            p + "quat_w": q[0], p + "quat_x": q[1], p + "quat_y": q[2], p + "quat_z": q[3],
            p + "accel_x": a[0], p + "accel_y": a[1], p + "accel_z": a[2],
            p + "gyro_x":  g[0], p + "gyro_y":  g[1], p + "gyro_z":  g[2],
        }

    async def _main(self):
        self._server = UDPIMUServer(port=UDP_PORT)
        self._server.run()
        self.q.put(("status", f"Listening on UDP :{UDP_PORT}, waiting for sensor {self.racket_id}"))
        # Poll faster than the sensor rate (~3× target) and deduplicate.
        # The sensor delivers bursts over WiFi; polling at the target rate
        # holds duplicates, then jumps — which the model reads as impact
        # bursts (smash bias). Pushing only on real changes removes that.
        dt = 1.0 / (self.rate_hz * 3)
        last_signature = None
        seen = False
        while not self._stop.is_set():
            latest = self._server.get_latest_data()
            if self.racket_id in latest:
                d = latest[self.racket_id]
                # Signature = accel + gyro tuple; identical => duplicate packet
                sig = tuple(d["accel"]) + tuple(d["gyro"])
                if sig != last_signature:
                    last_signature = sig
                    if not seen:
                        seen = True
                        self.q.put(("status", "Streaming, swing away"))
                    self.q.put(("imu_row", self._build_row(d)))
            await asyncio.sleep(dt)
        await self._server.stop()

    def run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except Exception:
            import traceback
            self.q.put(("error", traceback.format_exc()))
        finally:
            self.q.put(("stream_done", None))


# ── MuJoCo teleop thread (shares the streamer's UDP server) ──────────────────

class MuJoCoTeleopThread(threading.Thread):
    """
    Opens a MuJoCo viewer of the humanoid and drives its right-arm joints in
    real time from sensors 7/8/9 via Zuyan's SiriusCeptionTeleop. Reuses the
    LiveIMUStreamer's UDPIMUServer + asyncio loop so there's no UDP port
    conflict and no second sensor connection.
    """

    def __init__(self, streamer: "LiveIMUStreamer", event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.streamer = streamer
        self.q        = event_queue
        self._stop    = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._run_inner()
        except Exception:
            import traceback
            self.q.put(("error", "[MuJoCo] " + traceback.format_exc()))

    def _run_inner(self):
        # 1. Wait for the streamer to have its server up
        self.q.put(("status", "MuJoCo waiting for UDP server…"))
        while self.streamer._server is None or self.streamer._loop is None:
            if self._stop.is_set():
                return
            time.sleep(0.05)
        server = self.streamer._server
        loop   = self.streamer._loop

        # 2. Wait for all 3 arm sensors
        self.q.put(("status", f"MuJoCo waiting for sensors {SENSOR_HAND}/{SENSOR_FORE}/{SENSOR_UPPER}…"))
        while not self._stop.is_set():
            latest = server.get_latest_data()
            if all(s in latest for s in (SENSOR_HAND, SENSOR_FORE, SENSOR_UPPER)):
                break
            time.sleep(0.1)
        if self._stop.is_set():
            return

        # 3. Build teleop and calibrate (calibrate is async → run in streamer's loop)
        teleop = SiriusCeptionTeleop(
            imu_server  = server,
            upper_id    = SENSOR_UPPER,
            fore_id     = SENSOR_FORE,
            hand_id     = SENSOR_HAND,
            l_upper     = L_UPPER,
            l_fore      = L_FORE,
            init_pose   = "forward",
            earth_frame = "SEU",
        )
        self.q.put(("status",
                    f"MuJoCo calibrating, hold T-pose for {MJ_CALIB_SECS:.0f}s"))
        fut = asyncio.run_coroutine_threadsafe(
            teleop.calibrate(duration=MJ_CALIB_SECS), loop)
        fut.result()
        self.q.put(("status", "MuJoCo skeleton ready"))

        # 4. Load humanoid + resolve joint/body refs
        model = mujoco.MjModel.from_xml_path(str(MJ_MODEL))
        data  = mujoco.MjData(model)
        data.qpos[2] = 1.282
        data.qpos[3] = 1.0
        mujoco.mj_forward(model, data)

        def jaddr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return model.jnt_qposadr[jid]
        def bid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        addr_s1 = jaddr("shoulder1_right")
        addr_s2 = jaddr("shoulder2_right")
        addr_el = jaddr("elbow_right")
        bid_upper = bid("upper_arm_right")
        bid_lower = bid("lower_arm_right")
        bid_hand  = bid("hand_right")

        def ik_cost(qv, t_se, t_ew):
            data.qpos[addr_s1] = qv[0]
            data.qpos[addr_s2] = qv[1]
            data.qpos[addr_el] = qv[2]
            mujoco.mj_forward(model, data)
            cur_se = data.xpos[bid_lower] - data.xpos[bid_upper]
            cur_ew = data.xpos[bid_hand]  - data.xpos[bid_lower]
            return float(np.sum((cur_se - t_se) ** 2) +
                         np.sum((cur_ew - t_ew) ** 2))

        q_warm = np.array([0.0, 0.0, -0.5])
        dt = 1.0 / MJ_FPS

        # 5. Open the viewer (this is a blocking context manager)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not self._stop.is_set():
                t0 = time.time()
                try:
                    positions, _hand_R = teleop.compute()
                except Exception:
                    viewer.sync(); time.sleep(dt); continue

                shoulder = np.asarray(positions[0], dtype=float)
                elbow    = np.asarray(positions[1], dtype=float)
                wrist    = np.asarray(positions[2], dtype=float)
                se = elbow - shoulder
                ew = wrist - elbow
                n_se = np.linalg.norm(se); n_ew = np.linalg.norm(ew)
                if n_se < 1e-6 or n_ew < 1e-6:
                    viewer.sync(); time.sleep(dt); continue

                target_se = (se / n_se) * COORD_SIGN * L_UPPER
                target_ew = (ew / n_ew) * COORD_SIGN * L_FORE

                res = minimize(
                    ik_cost, q_warm,
                    args=(target_se, target_ew),
                    method="Nelder-Mead",
                    options={"maxiter": 30, "xatol": 1e-3, "fatol": 1e-4},
                )
                q_warm = res.x
                data.qpos[addr_s1] = q_warm[0]
                data.qpos[addr_s2] = q_warm[1]
                data.qpos[addr_el] = q_warm[2]
                mujoco.mj_forward(model, data)
                viewer.sync()

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        self.q.put(("status", "MuJoCo viewer closed"))


# ── Online stroke counter (one count per motion-energy peak) ───────────────────

class OnlineStrokeCounter:
    """Streaming peak detector: one stroke per local energy maximum above
    min_height, with a refractory min_distance. One-window detection latency."""

    def __init__(self, min_distance=PEAK_MIN_DISTANCE, min_height=LIVE_PEAK_MIN_HEIGHT,
                 rel_frac=PEAK_REL_FRAC):
        from collections import deque
        self.min_distance = min_distance
        self.min_height   = min_height
        self.rel_frac     = rel_frac
        self.e_prev2 = None
        self.e_prev1 = None
        self.lab_prev1  = 0
        self.conf_prev1 = 0.0
        self.idx = 0
        self.last_stroke_idx = -10 ** 9
        self.counts = {1: 0, 2: 0, 3: 0}
        self._recent_energies = deque(maxlen=8)

    def update(self, energy: float, label_id: int, confidence: float):
        fired = None
        is_local_max = (self.e_prev2 is not None and self.e_prev1 is not None
                        and self.e_prev2 < self.e_prev1 >= energy)

        if (is_local_max
                and self.e_prev1 >= self.min_height
                and (self.idx - 1) - self.last_stroke_idx >= self.min_distance
                and self.lab_prev1 != 0):

            accept = True
            if self.rel_frac > 0 and len(self._recent_energies) >= 3:
                rel_thr = self.rel_frac * float(np.median(self._recent_energies))
                if self.e_prev1 < rel_thr:
                    accept = False

            if accept:
                self.last_stroke_idx = self.idx - 1
                self.counts[self.lab_prev1] = self.counts.get(self.lab_prev1, 0) + 1
                self._recent_energies.append(self.e_prev1)
                fired = (self.lab_prev1, self.conf_prev1)

        self.e_prev2, self.e_prev1 = self.e_prev1, energy
        self.lab_prev1, self.conf_prev1 = label_id, confidence
        self.idx += 1
        return fired

    def total(self):
        return sum(self.counts.values())


# ── Main GUI ───────────────────────────────────────────────────────────────────

class LiveApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Table Tennis Coaching Live (MuJoCo skeleton)")
        self.geometry("1100x760")
        self.configure(bg=COLORS["bg"])
        self._anim_t = 0.0

        with open(ROOT / "config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        print(f"[Model] Loading {MODEL_PATH}")
        self.predictor = StrokePredictor.from_checkpoint(
            str(MODEL_PATH), str(SCALER_PATH), cfg=self.cfg)

        self.event_queue   = queue.Queue()
        self.streamer      = None
        self._mj_thread    = None     # MuJoCo teleop thread
        self.session_active = False
        self._extractor    = SlidingWindowExtractor(WINDOW_SIZE, STEP_SIZE)
        self._counter      = OnlineStrokeCounter()
        self.session_rows  = []
        self.session_preds = []
        self.n_windows     = 0
        self._warmup_until = 0.0

        self._build_ui()
        self._poll_queue()
        self._animate_stripe()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _animate_stripe(self):
        if not self.winfo_exists():
            return
        try:
            import math, colorsys
            self._anim_t += 0.03
            h = (240 + 60 * math.sin(self._anim_t)) / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            self._stripe.config(bg=color)
        except Exception:
            pass
        self.after(60, self._animate_stripe)

    def _on_close(self):
        try:
            if self._mj_thread:
                self._mj_thread.stop()
        except Exception:
            pass
        try:
            if self.streamer:
                self.streamer.stop()
        except Exception:
            pass
        self.destroy()

    # ── Skeleton launcher ───────────────────────────────────────────────────
    def _launch_skeleton(self):
        """Spin up the MuJoCo viewer + teleop in a background thread.
        Requires the LiveIMUStreamer to already be running (press Start Live
        first)."""
        if self._mj_thread is not None and self._mj_thread.is_alive():
            self.status_var.set("◉ Skeleton already running")
            return
        if self.streamer is None or not self.streamer.is_alive():
            messagebox.showwarning(
                "Start the session first",
                "Press 'Start Live' before launching the MuJoCo skeleton.\n"
                "The skeleton shares the UDP server with the classifier.")
            return
        self._mj_thread = MuJoCoTeleopThread(self.streamer, self.event_queue)
        self._mj_thread.start()
        self.btn_skel.config(state=tk.DISABLED)
        self.status_var.set("◉ MuJoCo skeleton starting")
        self._log("◇  MuJoCo viewer launching\n", tag="muted")

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.configure(bg=COLORS["bg"])

        # Header
        hdr = style_header(self)
        tk.Label(hdr, text="◆  TABLE TENNIS COACHING",
                 font=FONTS["h1"], fg=COLORS["text"],
                 bg=COLORS["surface"]).pack(side=tk.LEFT, padx=22, pady=14)
        tk.Label(hdr, text="LIVE",
                 font=FONTS["h3"], fg=COLORS["accent"],
                 bg=COLORS["surface"]).pack(side=tk.LEFT, pady=18)
        tk.Label(hdr,
                 text="Real-Time UDP  ·  Unity Skeleton  ·  LLM review",
                 font=FONTS["body"], fg=COLORS["text_dim"],
                 bg=COLORS["surface"]).pack(side=tk.RIGHT, padx=22)
        # animated stripe (replaces the static accent strip from style_header)
        self._stripe = tk.Frame(self, bg=COLORS["accent"], height=3)
        self._stripe.pack(fill=tk.X)

        # Footer
        footer = style_footer(self)
        self.status_var = tk.StringVar(value="◉ Ready, press Start Live")
        tk.Label(footer, textvariable=self.status_var, font=FONTS["body"],
                 fg=COLORS["text_dim"], bg=COLORS["surface"]
                 ).pack(side=tk.RIGHT, padx=22)
        self.btn_skel   = make_button(footer, "◆  SKELETON 3D",  self._launch_skeleton, kind="ghost")
        self.btn_start  = make_button(footer, "▶  START LIVE",  self._start,  kind="success")
        self.btn_stop   = make_button(footer, "■  STOP",        self._stop,   kind="danger")
        self.btn_report = make_button(footer, "✦  LLM REPORT",  self._generate_report, kind="violet")
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_skel.pack(side=tk.LEFT, padx=(22, 6), pady=14)
        self.btn_start.pack(side=tk.LEFT, padx=6, pady=14)
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=14)
        self.btn_report.pack(side=tk.LEFT, padx=6, pady=14)

        # Body
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        # Left card: Stroke Counts only
        left = make_card(body); left.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        inner_l = tk.Frame(left, bg=COLORS["card"])
        inner_l.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        section_title(inner_l, "STROKE COUNTS").pack(anchor="w")
        self.count_labels = {}
        for cid in (1, 2, 3):
            row = tk.Frame(inner_l, bg=COLORS["card"])
            row.pack(fill=tk.X, pady=4)
            dot = tk.Label(row, text="●", font=FONTS["h2"],
                           fg=CLASS_COLORS[CLASS_NAMES[cid]], bg=COLORS["card"])
            dot.pack(side=tk.LEFT)
            name = tk.Label(row, text=SHORT_NAMES[cid], font=FONTS["body"],
                            fg=COLORS["text"], bg=COLORS["card"])
            name.pack(side=tk.LEFT, padx=(8, 0))
            val = tk.Label(row, text="0", font=FONTS["value"],
                           fg=CLASS_COLORS[CLASS_NAMES[cid]], bg=COLORS["card"])
            val.pack(side=tk.RIGHT)
            self.count_labels[cid] = val

        # totals row
        totals = tk.Frame(inner_l, bg=COLORS["card_alt"])
        totals.pack(fill=tk.X, pady=(16, 0), ipady=10)
        col1 = tk.Frame(totals, bg=COLORS["card_alt"]); col1.pack(side=tk.LEFT, padx=14)
        tk.Label(col1, text="TOTAL", font=FONTS["caption"],
                 fg=COLORS["text_muted"], bg=COLORS["card_alt"]).pack(anchor="w")
        self.lbl_total = tk.Label(col1, text="0", font=FONTS["value_md"],
                                  fg=COLORS["accent"], bg=COLORS["card_alt"])
        self.lbl_total.pack(anchor="w")
        col2 = tk.Frame(totals, bg=COLORS["card_alt"]); col2.pack(side=tk.RIGHT, padx=14)
        tk.Label(col2, text="WINDOWS", font=FONTS["caption"],
                 fg=COLORS["text_muted"], bg=COLORS["card_alt"]).pack(anchor="e")
        self.lbl_windows = tk.Label(col2, text="0", font=FONTS["value_md"],
                                    fg=COLORS["text"], bg=COLORS["card_alt"])
        self.lbl_windows.pack(anchor="e")

        # T-pose countdown banner (replaces the old live prediction box)
        self.lbl_countdown = tk.Label(inner_l, text="",
                                      font=FONTS["value_md"],
                                      fg=COLORS["warning"], bg=COLORS["card"])
        self.lbl_countdown.pack(anchor="w", pady=(18, 0))

        # Right card: Shot Predictions
        right = make_card(body); right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        inner_r = tk.Frame(right, bg=COLORS["card"])
        inner_r.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)
        section_title(inner_r, "SHOT PREDICTIONS").pack(anchor="w", pady=(0, 8))
        self.log = scrolledtext.ScrolledText(
            inner_r, font=("Cascadia Mono", 12),
            bg=COLORS["card_dark"], fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            selectbackground=COLORS["accent_dim"],
            relief="flat", bd=0, padx=10, pady=10)
        self.log.pack(fill=tk.BOTH, expand=True)
        # log tag colours
        self.log.tag_config("stroke",  foreground=COLORS["accent"])
        self.log.tag_config("muted",   foreground=COLORS["text_muted"])
        self.log.tag_config("warn",    foreground=COLORS["warning"])
        self.log.tag_config("danger",  foreground=COLORS["danger"])

    # ── Session control ───────────────────────────────────────────────────────
    def _start(self):
        if self.session_active:
            return
        self._extractor.reset()
        self._counter = OnlineStrokeCounter()
        self.session_rows.clear(); self.session_preds.clear(); self.n_windows = 0
        for cid in (1, 2, 3):
            self.count_labels[cid].config(text="0")
        self.lbl_total.config(text="0")
        self.lbl_windows.config(text="0")
        self.log.delete("1.0", tk.END)
        self._log("··  Live session started  ··\n", tag="muted")
        self.session_active = True
        self.btn_start.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)

        # DEBUG: open a raw-row dump so we can replay this session offline
        self._dbg_file = None
        if DEBUG_DUMP_ROWS:
            import csv as _csv
            dbg_path = IMU_DATA_DIR / "recordings" / f"_live_debug_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            self._dbg_cols = [f"imu_{RACKET_ID}_quat_w", f"imu_{RACKET_ID}_quat_x",
                              f"imu_{RACKET_ID}_quat_y", f"imu_{RACKET_ID}_quat_z",
                              f"imu_{RACKET_ID}_accel_x", f"imu_{RACKET_ID}_accel_y", f"imu_{RACKET_ID}_accel_z",
                              f"imu_{RACKET_ID}_gyro_x", f"imu_{RACKET_ID}_gyro_y", f"imu_{RACKET_ID}_gyro_z"]
            self._dbg_file = open(dbg_path, "w", newline="")
            self._dbg_writer = _csv.writer(self._dbg_file)
            self._dbg_writer.writerow(self._dbg_cols)
            self._log(f"[debug] dumping raw rows → {dbg_path.name}\n")

        self.streamer = LiveIMUStreamer(self.event_queue)
        self.streamer.start()

        self._warmup_until = time.time() + TPOSE_SECONDS
        self._update_countdown()

    def _update_countdown(self):
        if not self.session_active:
            return
        remaining = self._warmup_until - time.time()
        if remaining > 0:
            self.lbl_countdown.config(text=f"T-POSE  {int(remaining)+1}s",
                                      fg=COLORS["warning"])
            self.status_var.set("◉ Hold T-pose, arms straight out to the sides")
            self.after(200, self._update_countdown)
        else:
            self.lbl_countdown.config(text="GO  ·  swing away",
                                      fg=COLORS["success"])
            self.status_var.set("◉ Streaming")
            self.after(2500, lambda: self.lbl_countdown.config(text=""))

    def _stop(self):
        if self.streamer:
            self.streamer.stop()
        if self._mj_thread is not None:
            try: self._mj_thread.stop()
            except Exception: pass
            self._mj_thread = None
        self.session_active = False
        self.btn_start.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED)
        self.btn_skel.config(state=tk.NORMAL)
        if getattr(self, "_dbg_file", None) is not None:
            try:
                self._dbg_file.close()
            except Exception:
                pass
            self._dbg_file = None
        self._log(f"··  Session ended, {self._counter.total()} strokes  ··\n",
                  tag="muted")
        self.status_var.set("◉ Session ended")

    # ── Queue polling ───────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            for _ in range(500):
                msg_type, payload = self.event_queue.get_nowait()
                if msg_type == "imu_row":
                    self._handle_row(payload)
                elif msg_type == "status":
                    self.status_var.set(payload)
                elif msg_type == "error":
                    self._log(f"[ERROR]\n{payload}\n")
                    self.status_var.set("Streamer error, see log")
                elif msg_type == "stream_done":
                    if self.session_active:
                        self._stop()
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    # ── Per-window inference ────────────────────────────────────────────────
    def _handle_row(self, row: dict):
        # DEBUG: record every raw row exactly as received
        if DEBUG_DUMP_ROWS and getattr(self, "_dbg_file", None) is not None:
            self._dbg_writer.writerow([row.get(c, 0.0) for c in self._dbg_cols])

        features = self._extractor.add_frame(row)
        if features is None:
            return
        # T-pose warm-up: keep filling the window buffer but don't classify/count
        if time.time() < self._warmup_until:
            return
        self.n_windows += 1
        result = self.predictor.predict(features)

        energy = window_energy(features)
        if is_idle_window(features):
            result = {"label_id": 0, "label_name": CLASS_NAMES[0],
                      "confidence": result["confidence"],
                      "probabilities": result["probabilities"]}

        self.session_rows.append(dict(zip(FEATURE_COLS, features.tolist())))
        self.session_preds.append(result)

        fired = self._counter.update(energy, result["label_id"], result["confidence"])
        if fired is not None:
            lab_id, conf = fired
            self.count_labels[lab_id].config(text=str(self._counter.counts[lab_id]))
            self.lbl_total.config(text=str(self._counter.total()))
            self._log(f"  ✦  #{self._counter.total():<3} "
                      f"{SHORT_NAMES[lab_id]:<11}  conf {conf:.0%}\n", tag="stroke")

        self._update_live(result)
        self.lbl_windows.config(text=str(self.n_windows))

    def _update_live(self, result):
        # Live prediction panel removed; predictions appear in Shot Predictions log.
        return

    def _log(self, text, tag=None):
        if tag:
            self.log.insert(tk.END, text, tag)
        else:
            self.log.insert(tk.END, text)
        self.log.see(tk.END)

    # ── LLM report ────────────────────────────────────────────────────────────
    def _generate_report(self):
        if not self.session_rows:
            messagebox.showwarning("No Data", "No session data to report on.")
            return
        self.btn_report.config(state=tk.DISABLED, text="⌛  GENERATING")
        self.status_var.set("◉ Calling LLM, please wait")

        def _run():
            try:
                session_df = pd.DataFrame(self.session_rows)
                summary    = run_session(self.predictor, session_df)
                feedback   = get_coaching_feedback(summary, subject_id=SUBJECT_ID, cfg=self.cfg)
                self.after(0, lambda: self._show_report(feedback))
            except Exception as e:
                self.after(0, lambda: self._report_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_report(self, feedback):
        self.btn_report.config(state=tk.NORMAL, text="✦  LLM REPORT")
        self.status_var.set("◉ Report generated")
        self._log("\n◆◆  LLM COACHING REPORT  ◆◆\n", tag="stroke")
        self._log(f"ASSESSMENT  →  {feedback['assessment']}\n\n")
        for rec in feedback["recommendations"]:
            self._log(f"  [{rec['priority']}]  #{rec['rank']}  {rec['text']}\n")
        self._log(f"\nNEXT FOCUS  →  {feedback['next_focus']}\n")
        self._log("─" * 50 + "\n", tag="muted")

    def _report_error(self, msg):
        self.btn_report.config(state=tk.NORMAL, text="✦  LLM REPORT")
        self.status_var.set("⚠ LLM error, is LM Studio running?")
        self._log(f"[LLM ERROR] {msg}\n", tag="danger")


if __name__ == "__main__":
    app = LiveApp()
    app.mainloop()
