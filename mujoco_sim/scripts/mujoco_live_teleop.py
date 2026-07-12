"""
mujoco_live_teleop.py
─────────────────────
PHASE 2 of the MuJoCo teleop migration.

Drives the MuJoCo humanoid's right arm in real time from three SiriusCeption
IMU sensors via Zuyan's SiriusCeptionTeleop module.

Pipeline:
    SiriusCeption sensors (id 7=hand, 8=forearm, 9=upper arm)
        → UDPIMUServer (background asyncio thread)
        → SiriusCeptionTeleop.compute() → arm-segment positions
        → per-frame numerical IK on (shoulder1, shoulder2, elbow)
        → set MuJoCo qpos → mj_forward → viewer.sync()

Replaces the Unity skeleton for the live demo. Same sensors, same calibration
ritual (3-second T-pose), but MuJoCo's built-in viewer gives you mouse
orbit/zoom/pan for free.

Usage:
    python mujoco_live_teleop.py

Pre-flight:
    - All 3 sensors powered, on WiFi, configured for this PC's IP
    - Stand in T-POSE before launching (calibration starts after sensors
      are detected — hold for ~3 s)

Viewer controls:
    Left  mouse + drag → orbit
    Right mouse + drag → pan
    Scroll wheel       → zoom
    Esc                → quit
"""
import sys
import time
import math
import asyncio
import threading
import pathlib

import numpy as np
from scipy.optimize import minimize

import mujoco
import mujoco.viewer

# ── Paths ──────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "humanoid.xml"
IMU_DIR    = pathlib.Path(r"E:\thesis_work\TT_thesis\imu_data")          # udp_imu_server.py
TELEOP_DIR = pathlib.Path(r"E:\thesis_work\TT_thesis\SiriusCeption-UserUI")  # SiriusCeption_Teleop.pyd

sys.path.insert(0, str(IMU_DIR))
sys.path.insert(0, str(TELEOP_DIR))

from udp_imu_server          import UDPIMUServer
from SiriusCeption_Teleop    import SiriusCeptionTeleop

# ── Config ─────────────────────────────────────────────────────────────
UDP_PORT      = 9999
SENSOR_HAND   = 7      # racket hand
SENSOR_FORE   = 8      # forearm
SENSOR_UPPER  = 9      # upper arm

L_UPPER       = 0.28   # MuJoCo upper-arm length (from humanoid.xml fromto)
L_FORE        = 0.28   # MuJoCo forearm length

CALIB_SECS    = 3.0
TARGET_FPS    = 30

# Coordinate-frame fix between teleop output and MuJoCo world.
# Teleop with init_pose='forward' uses x=forward, y=left, z=up.
# MuJoCo's humanoid uses the same convention by default.
# If the arm moves the wrong way, change one of these to -1.
COORD_SIGN = np.array([1.0, 1.0, -1.0])     # multiplies each axis


# ── Background thread: UDPIMUServer + SiriusCeptionTeleop ─────────────

class TeleopThread(threading.Thread):
    """Runs the async UDP server and the teleop in its own event loop."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self.calibrated = threading.Event()
        self.sensors_seen = threading.Event()
        self.teleop = None
        self.server = None

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            asyncio.run(self._async_main())
        except Exception:
            import traceback; traceback.print_exc()

    async def _async_main(self):
        self.server = UDPIMUServer(port=UDP_PORT)
        self.server.run()
        await asyncio.sleep(0.4)

        # Wait until all 3 sensors are sending packets
        print(f"[teleop] Waiting for sensors {SENSOR_HAND}, {SENSOR_FORE}, {SENSOR_UPPER}…")
        while not self._stop_evt.is_set():
            latest = self.server.get_latest_data()
            if all(s in latest for s in (SENSOR_HAND, SENSOR_FORE, SENSOR_UPPER)):
                break
            await asyncio.sleep(0.1)
        if self._stop_evt.is_set():
            return
        print("[teleop] All 3 sensors online.")
        self.sensors_seen.set()

        # Build the teleop object
        self.teleop = SiriusCeptionTeleop(
            imu_server   = self.server,
            upper_id     = SENSOR_UPPER,
            fore_id      = SENSOR_FORE,
            hand_id      = SENSOR_HAND,
            l_upper      = L_UPPER,
            l_fore       = L_FORE,
            init_pose    = "forward",       # +X forward in teleop world frame
            earth_frame  = "SEU",           # BNO08X native
        )

        # ── 3-second still-pose calibration ─────────────────────────────
        print(f"[teleop] STAND IN T-POSE. Calibrating for {CALIB_SECS:.0f} seconds…")
        await self.teleop.calibrate(duration=CALIB_SECS)
        print("[teleop] Calibration complete.")
        self.calibrated.set()

        # Keep the loop alive so the server keeps reading packets
        while not self._stop_evt.is_set():
            await asyncio.sleep(0.05)

        await self.server.stop()


# ── Main: MuJoCo viewer driven by teleop output ────────────────────────

def main():
    # Load humanoid
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    # Stand upright
    data.qpos[2] = 1.282
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    # Resolve joint qpos addresses and body ids we need for IK
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

    print(f"shoulder1_right  qpos[{addr_s1}]")
    print(f"shoulder2_right  qpos[{addr_s2}]")
    print(f"elbow_right      qpos[{addr_el}]")

    # ── Start the IMU + teleop thread ───────────────────────────────────
    th = TeleopThread()
    th.start()

    print("[main] Waiting for sensors and calibration…")
    th.sensors_seen.wait()
    th.calibrated.wait()
    print("[main] Teleop ready. Opening viewer.")

    # ── IK cost: match shoulder→elbow and elbow→wrist direction vectors ─
    def ik_cost(q, target_se, target_ew):
        data.qpos[addr_s1] = q[0]
        data.qpos[addr_s2] = q[1]
        data.qpos[addr_el] = q[2]
        mujoco.mj_forward(model, data)
        cur_se = data.xpos[bid_lower] - data.xpos[bid_upper]
        cur_ew = data.xpos[bid_hand]  - data.xpos[bid_lower]
        return float(np.sum((cur_se - target_se) ** 2) +
                     np.sum((cur_ew - target_ew) ** 2))

    q_warm = np.array([0.0, 0.0, -0.5])   # warm-start from a slight elbow bend

    # ── Viewer loop ─────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        dt = 1.0 / TARGET_FPS
        while viewer.is_running():
            t0 = time.time()

            # Read latest teleop output
            try:
                positions, hand_R = th.teleop.compute()
            except Exception:
                viewer.sync()
                time.sleep(dt)
                continue

            # positions = [shoulder, elbow, wrist, hand_tip] in teleop world frame
            shoulder = np.asarray(positions[0], dtype=float)
            elbow    = np.asarray(positions[1], dtype=float)
            wrist    = np.asarray(positions[2], dtype=float)

            # Direction vectors (independent of teleop's absolute origin)
            se = elbow - shoulder
            ew = wrist - elbow
            n_se = np.linalg.norm(se)
            n_ew = np.linalg.norm(ew)
            if n_se < 1e-6 or n_ew < 1e-6:
                viewer.sync()
                time.sleep(dt)
                continue

            # Normalize, apply coord sign fix, scale to MuJoCo bone lengths
            target_se = (se / n_se) * COORD_SIGN * L_UPPER
            target_ew = (ew / n_ew) * COORD_SIGN * L_FORE

            # IK — cheap because only 3 variables
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

            # Pace
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    th.stop()
    print("Viewer closed. Bye.")


if __name__ == "__main__":
    main()
