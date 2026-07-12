"""
mujoco_live_viewer.py
─────────────────────
PHASE 1 of the MuJoCo teleop migration.

Goal: open the same humanoid we use for synthetic data in an INTERACTIVE
3-D viewer (mouse orbit / zoom / pan built-in), and animate the right arm
with a sine wave so we confirm:
    - the viewer launches
    - the humanoid is visible
    - we can set qpos and see the result in real time
    - the right-arm joint indices we'll drive from IMUs are correct

PHASE 2 (next step, separate file) will swap the sine source for real
SiriusCeption IMU data via Zuyan's SiriusCeptionTeleop module.

Usage:
    python mujoco_live_viewer.py

Controls inside the viewer window:
    Left mouse + drag   → orbit
    Right mouse + drag  → pan
    Scroll              → zoom
    Esc                 → quit
"""
import math
import time
import pathlib

import mujoco
import mujoco.viewer

HERE       = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "humanoid.xml"


def main():
    # ── 1. Load the humanoid ──────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    # Joint indices (qpos addresses) for the right arm
    name_to_qpos_addr = {}
    for jname in ("shoulder1_right", "shoulder2_right", "elbow_right"):
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        addr = model.jnt_qposadr[jid]
        name_to_qpos_addr[jname] = addr
        print(f"{jname:>18s}  joint_id={jid}  qpos_addr={addr}")

    # Start at a neutral standing pose
    data.qpos[2] = 1.282           # root height so feet sit on floor
    data.qpos[3] = 1.0             # root quaternion w = 1 (identity)
    mujoco.mj_forward(model, data)

    print()
    print("Opening interactive viewer (mouse: orbit/zoom/pan, Esc to quit)…")

    # ── 2. Launch the passive viewer ─────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0

            # Sine-wave the right-arm joints so we see the avatar move
            data.qpos[name_to_qpos_addr["shoulder1_right"]] = 0.5 * math.sin(2.0 * t)
            data.qpos[name_to_qpos_addr["shoulder2_right"]] = 0.3 * math.sin(2.0 * t + 1.0)
            data.qpos[name_to_qpos_addr["elbow_right"]]     = -0.8 + 0.6 * math.sin(2.0 * t)

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)         # ~60 fps

    print("Viewer closed.")


if __name__ == "__main__":
    main()
