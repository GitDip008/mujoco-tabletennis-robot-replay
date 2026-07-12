"""
stream_to_unity.py
==================
Streams any IMU CSV (Zuyan format OR synthetic MuJoCo CSV) to Unity
using the SAME IMUUnityController .pyd that realtimeapp_new.py uses.

This is the correct approach — the .pyd handles all coordinate frame
math, calibration, and bone offset calculations internally.
No coordinate conversion needed on our side.

REQUIRES: Python 3.11  (the .pyd is cp311-win_amd64)
Run with:
    E:\\thesis_work\\imu_to_unity\\.py11_venv\\Scripts\\python.exe stream_to_unity.py --csv <path>

USAGE
-----
    # Stream synthetic MuJoCo data:
    python stream_to_unity.py --csv ..\\output\\tt_synthetic_full.csv

    # Stream Zuyan's real data:
    python stream_to_unity.py --csv E:\\thesis_work\\thesis_works_new\\imu_data_log_20250624_204958.csv

    # Slower playback to see strokes clearly:
    python stream_to_unity.py --csv ..\\output\\tt_synthetic_full.csv --speed 0.3

    # More calibration frames (stand still at start of CSV):
    python stream_to_unity.py --csv ..\\output\\tt_synthetic_full.csv --calib 120
"""

import argparse
import csv
import sys
import time
import pathlib

# ── Paths ──────────────────────────────────────────────────────────────────────
IMU_DIR = pathlib.Path(r"E:\thesis_work\thesis_works_new")
sys.path.insert(0, str(IMU_DIR))

from SiriusCeption_unity_controller import IMUUnityController


def stream(csv_path: str, calib_frames: int, speed: float, port: int):

    print(f"[INFO] CSV             : {csv_path}")
    print(f"[INFO] Calibration     : first {calib_frames} frames (stand still)")
    print(f"[INFO] Playback speed  : {speed}x")
    print(f"[INFO] Unity UDP port  : {port}")
    print()

    controller = IMUUnityController(
        bone_hierarchy_path=str(IMU_DIR / "BoneHierarchy.txt"),
        bone_offsets_path   =str(IMU_DIR / "BoneOffsets.json"),
        tpose_quats_path    =str(IMU_DIR / "InitialPoseExport.txt"),
        udp_ip              ="127.0.0.1",
        udp_port            =port,
        position_scale      =1.0,
        hips_y_scale        =1.0,
    )

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            # ── Calibration phase ──────────────────────────────────────────────
            print(f"[INFO] Calibrating — make sure Unity is in Play mode...")
            controller.calibrate(reader, frames=calib_frames)
            print(f"[INFO] Calibration done ✓")
            print(f"[INFO] Streaming to Unity — press Ctrl+C to stop\n")

            # ── Streaming phase ────────────────────────────────────────────────
            frame = 0
            for row in reader:
                sleep_s, _ = controller.process_row(row)

                # Print stroke label if present (synthetic data has it)
                label = row.get("stroke_name", "")
                if frame % 60 == 0:
                    label_str = f"  [{label}]" if label else ""
                    print(f"  Frame {frame:>6}{label_str}")

                # Respect timing, adjusted for playback speed
                if 0 < sleep_s < 1.0:
                    time.sleep(sleep_s / speed)

                frame += 1

    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped by user after {frame} frames.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        controller.close()
        print("[INFO] Done.")


def main():
    p = argparse.ArgumentParser(
        description="Stream IMU CSV → Unity skeleton via SiriusCeption .pyd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",   required=True,
                   help="Path to IMU CSV file (Zuyan format or MuJoCo synthetic)")
    p.add_argument("--calib", default=100, type=int,
                   help="Calibration frames (default 100)")
    p.add_argument("--speed", default=1.0, type=float,
                   help="Playback speed multiplier — 0.3 = slow motion (default 1.0)")
    p.add_argument("--port",  default=5005, type=int,
                   help="Unity UDP port (default 5005)")
    args = p.parse_args()

    stream(args.csv, args.calib, args.speed, args.port)


if __name__ == "__main__":
    main()
