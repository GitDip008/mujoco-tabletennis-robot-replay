"""
mujoco_live_teleop_tt.py
────────────────────────
PHASE 3: same as mujoco_live_teleop.py but loads the table-tennis scene
(humanoid_tt.xml) — adds a paddle attached to the right hand, a regulation
table, a net, and a 40 mm ball with realistic physics.

Pipeline:
    SiriusCeption sensors (id 7/8/9)
        → UDPIMUServer
        → SiriusCeptionTeleop → arm-segment positions
        → numerical IK on shoulder1_right / shoulder2_right / elbow_right
        → kinematic update of the avatar each frame (qpos written directly)
        → physics step (mj_step) so the BALL responds to gravity, the table,
          and contact with the paddle
        → MuJoCo viewer (mouse orbit / zoom / pan built in)

Key controls inside the viewer:
    R or SPACE   reset the ball above the avatar's right hand
    Esc          quit
"""
import sys
import time
import asyncio
import threading
import pathlib

import numpy as np
from scipy.optimize import minimize

import mujoco
import mujoco.viewer

# ── Paths ──────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "humanoid_tt.xml"
IMU_DIR    = pathlib.Path(r"E:\thesis_work\TT_thesis\imu_data")
TELEOP_DIR = pathlib.Path(r"E:\thesis_work\TT_thesis\SiriusCeption-UserUI")

sys.path.insert(0, str(IMU_DIR))
sys.path.insert(0, str(TELEOP_DIR))

from udp_imu_server       import UDPIMUServer
from SiriusCeption_Teleop import SiriusCeptionTeleop

# ── Config ─────────────────────────────────────────────────────────────
UDP_PORT      = 9999
SENSOR_HAND   = 7
SENSOR_FORE   = 8
SENSOR_UPPER  = 9
L_UPPER       = 0.28
L_FORE        = 0.28
CALIB_SECS    = 3.0
TARGET_FPS    = 60

COORD_SIGN    = np.array([1.0, 1.0, -1.0])   # from phase 2

# Two ball-delivery modes:
#   "toss"  : ball tossed UP near the player's hand (free practice, random
#             timing per swing). Use when you just want to feel contact.
#   "serve" : ball comes from the OPPOSITE end of the table toward the player
#             on a consistent arc — like a real serve, perfect for practising
#             the same stroke repeatedly.
BALL_MODE = "serve"

# ── "toss" mode (drop near the right hand) ───────────────────────────
TOSS_POS    = np.array([0.85, -0.10, 1.25])
TOSS_VEL    = np.array([0.0,  0.00, 3.80])

# ── "serve" mode (ball comes from the far end of the table) ──────────
# Origin is at the avatar; +X is forward; table center at x=1.7, top z=0.76.
# Far edge of table is x ≈ 3.07. Spawn just above and beyond.
SERVE_POS   = np.array([3.00,  0.10, 1.10])
SERVE_VEL   = np.array([-4.5,  0.00, 1.2])     # toward player, slight upward arc
# Adjust SERVE_POS[1] (left/right) and SERVE_VEL[0] (speed) for difficulty.

BALL_AUTO_RESET_Z   = 0.30
BALL_RESET_COOLDOWN = 0.8
SERVE_INTERVAL      = 2.5      # seconds between automatic serves (serve mode only)


# ── Background thread: UDPIMUServer + SiriusCeptionTeleop ─────────────

class TeleopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt   = threading.Event()
        self.calibrated  = threading.Event()
        self.sensors_seen = threading.Event()
        self.teleop      = None
        self.server      = None

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
            imu_server   = self.server,
            upper_id     = SENSOR_UPPER,
            fore_id      = SENSOR_FORE,
            hand_id      = SENSOR_HAND,
            l_upper      = L_UPPER,
            l_fore       = L_FORE,
            init_pose    = "forward",
            earth_frame  = "SEU",
        )
        print(f"[teleop] STAND IN T-POSE. Calibrating for {CALIB_SECS:.0f}s…")
        await self.teleop.calibrate(duration=CALIB_SECS)
        print("[teleop] Calibration complete.")
        self.calibrated.set()

        while not self._stop_evt.is_set():
            await asyncio.sleep(0.05)
        await self.server.stop()


# ── Main: MuJoCo viewer with TT scene driven by teleop ─────────────────

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    # Helpers
    def jaddr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    # Avatar right-arm joints (qpos addr, dof addr)
    s1_q, s1_v = jaddr("shoulder1_right")
    s2_q, s2_v = jaddr("shoulder2_right")
    el_q, el_v = jaddr("elbow_right")

    # Body ids needed for IK target reading
    bid_upper = bid("upper_arm_right")
    bid_lower = bid("lower_arm_right")
    bid_hand  = bid("hand_right")

    # Ball freejoint
    ball_q, ball_v = jaddr("ball_root")     # qpos addr, dof addr (qvel)

    # Stand pose: torso 1.282 m up, identity orientation
    data.qpos[2] = 1.282
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    # Snapshot the "stand pose" — every frame we restore everything from
    # this except the right-arm joints we drive AND the ball.
    stand_qpos = data.qpos.copy()

    last_reset = [0.0]   # mutable holder so nested funcs can update

    def _mode_pos_vel():
        if BALL_MODE == "serve":
            return SERVE_POS, SERVE_VEL
        return TOSS_POS, TOSS_VEL

    def reset_ball():
        pos, vel = _mode_pos_vel()
        data.qpos[ball_q:ball_q+3]     = pos
        data.qpos[ball_q+3:ball_q+7]   = [1.0, 0.0, 0.0, 0.0]
        data.qvel[ball_v:ball_v+3]     = vel
        data.qvel[ball_v+3:ball_v+6]   = 0.0
        stand_qpos[ball_q:ball_q+3]    = pos
        stand_qpos[ball_q+3:ball_q+7]  = [1.0, 0.0, 0.0, 0.0]
        last_reset[0] = time.time()
        print(f"[ball] {BALL_MODE} reset")

    reset_ball()
    mujoco.mj_forward(model, data)

    # ── Start IMU + teleop ─────────────────────────────────────────────
    th = TeleopThread(); th.start()
    print("[main] Waiting for sensors + calibration…")
    th.sensors_seen.wait()
    th.calibrated.wait()
    print("[main] Teleop ready. Opening viewer.")

    # ── IK cost ────────────────────────────────────────────────────────
    def ik_cost(qv, t_se, t_ew):
        data.qpos[s1_q] = qv[0]
        data.qpos[s2_q] = qv[1]
        data.qpos[el_q] = qv[2]
        mujoco.mj_forward(model, data)
        cur_se = data.xpos[bid_lower] - data.xpos[bid_upper]
        cur_ew = data.xpos[bid_hand]  - data.xpos[bid_lower]
        return float(np.sum((cur_se - t_se)**2) + np.sum((cur_ew - t_ew)**2))

    q_warm     = np.array([0.0, 0.0, -0.5])
    q_prev_arm = q_warm.copy()        # previous frame's arm qpos (for qvel estimate)

    # Keyboard callback for ball reset (R or SPACE)
    def key_callback(keycode):
        # GLFW: R=82, SPACE=32
        if keycode in (82, 32):
            reset_ball()

    dt_frame  = 1.0 / TARGET_FPS
    dt_phys   = model.opt.timestep          # MuJoCo integrator timestep (0.005 by default)
    # number of physics steps per render frame
    phys_steps = max(1, int(round(dt_frame / dt_phys)))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            t0 = time.time()

            # ── Ball auto-respawn ──────────────────────────────────────
            # 1) fell out of reach → respawn
            # 2) serve mode: respawn every SERVE_INTERVAL seconds for rhythm
            ball_z = float(data.qpos[ball_q + 2])
            since  = time.time() - last_reset[0]
            if since > BALL_RESET_COOLDOWN and (
                ball_z < BALL_AUTO_RESET_Z or
                (BALL_MODE == "serve" and since > SERVE_INTERVAL)
            ):
                reset_ball()

            # ── Read teleop ────────────────────────────────────────────
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

            # ── IK ─────────────────────────────────────────────────────
            res = minimize(ik_cost, q_warm, args=(t_se, t_ew),
                           method="Nelder-Mead",
                           options={"maxiter": 30, "xatol": 1e-3, "fatol": 1e-4})
            q_warm = res.x

            # ── Kinematic update of avatar + physics step for ball ─────
            for _ in range(phys_steps):
                # Preserve current ball state across the avatar reset
                ball_pos = data.qpos[ball_q:ball_q+7].copy()
                ball_vel = data.qvel[ball_v:ball_v+6].copy()

                # Restore avatar to stand pose, then write IK arm angles
                data.qpos[:] = stand_qpos
                data.qpos[s1_q] = q_warm[0]
                data.qpos[s2_q] = q_warm[1]
                data.qpos[el_q] = q_warm[2]
                data.qpos[ball_q:ball_q+7] = ball_pos

                # Zero qvel for the avatar; estimate arm qvel from finite diff
                # so the paddle's swing has the right contact velocity.
                data.qvel[:] = 0.0
                data.qvel[s1_v] = (q_warm[0] - q_prev_arm[0]) / dt_phys
                data.qvel[s2_v] = (q_warm[1] - q_prev_arm[1]) / dt_phys
                data.qvel[el_v] = (q_warm[2] - q_prev_arm[2]) / dt_phys
                data.qvel[ball_v:ball_v+6] = ball_vel

                mujoco.mj_step(model, data)

            q_prev_arm = q_warm.copy()

            viewer.sync()
            elapsed = time.time() - t0
            if elapsed < dt_frame:
                time.sleep(dt_frame - elapsed)

    th.stop()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
