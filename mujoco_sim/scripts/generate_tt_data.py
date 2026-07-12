"""
generate_tt_data.py
===================
Generates synthetic table tennis IMU data using MuJoCo humanoid simulation.

WHAT IT DOES
------------
1. Loads MuJoCo's humanoid model (17 body segments)
2. Keyframe-animates 4 stroke types:
      0 = No Stroke  (standing still, slight sway)
      1 = Forehand Topspin
      2 = Backhand Drive
      3 = Forehand Smash
3. Extracts virtual IMU readings per body:
      - Quaternion  (world-frame orientation)
      - Accelerometer (body-frame linear acceleration)
      - Gyroscope   (body-frame angular velocity)
4. Exports CSV in Zuyan's format:
      imu_{0-16}_quat_x/y/z/w, imu_{0-16}_accel_x/y/z, imu_{0-16}_gyro_x/y/z
   → Feed directly into your existing Python pipeline / classifier

BODY → IMU ID MAPPING (matches Zuyan's Xsens sensor assignment)
----------------------------------------------------------------
  0=Head        1=RightFoot    2=RightLowerLeg  3=RightUpperLeg
  4=LeftFoot    5=LeftLowerLeg 6=LeftUpperLeg   7=RightHand
  8=RightLowerArm 9=RightUpperArm 10=LeftHand  11=LeftLowerArm
  12=LeftUpperArm 13=Hips      14=Spine         15=RightShoulder
  16=LeftShoulder

USAGE
-----
  python generate_tt_data.py
  python generate_tt_data.py --reps 50 --fps 60 --out ../output/tt_synthetic.csv
  python generate_tt_data.py --reps 100 --noise 0.03 --out ../output/tt_synthetic.csv
"""

import argparse
import csv
import math
import pathlib
import time

import mujoco
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "humanoid.xml"
OUT_DIR    = HERE.parent / "output"

# ── MuJoCo body name → IMU sensor ID (matching Zuyan's Xsens layout) ─────────
# MuJoCo humanoid bodies:
#   world, torso, head, waist_lower, pelvis,
#   thigh_right, shin_right, foot_right,
#   thigh_left,  shin_left,  foot_left,
#   upper_arm_right, lower_arm_right, hand_right,
#   upper_arm_left,  lower_arm_left,  hand_left

BODY_TO_IMU = {
    "head":            0,   # Head
    "foot_right":      1,   # RightFoot
    "shin_right":      2,   # RightLowerLeg
    "thigh_right":     3,   # RightUpperLeg
    "foot_left":       4,   # LeftFoot
    "shin_left":       5,   # LeftLowerLeg
    "thigh_left":      6,   # LeftUpperLeg
    "hand_right":      7,   # RightHand  ← racket hand
    "lower_arm_right": 8,   # RightLowerArm
    "upper_arm_right": 9,   # RightUpperArm
    "hand_left":       10,  # LeftHand
    "lower_arm_left":  11,  # LeftLowerArm
    "upper_arm_left":  12,  # LeftUpperArm
    "pelvis":          13,  # Hips
    "waist_lower":     14,  # Spine
    # No dedicated shoulder bodies in humanoid — use upper_arm as proxy
    "upper_arm_right": 15,  # RightShoulder (same body, different sensor)
    "upper_arm_left":  16,  # LeftShoulder  (same body, different sensor)
}

# Final ordered mapping: IMU ID → MuJoCo body name
IMU_TO_BODY = {
    0:  "head",
    1:  "foot_right",
    2:  "shin_right",
    3:  "thigh_right",
    4:  "foot_left",
    5:  "shin_left",
    6:  "thigh_left",
    7:  "hand_right",
    8:  "lower_arm_right",
    9:  "upper_arm_right",
    10: "hand_left",
    11: "lower_arm_left",
    12: "upper_arm_left",
    13: "pelvis",
    14: "waist_lower",
    15: "upper_arm_right",   # RightShoulder proxy
    16: "upper_arm_left",    # LeftShoulder proxy
}


# ══════════════════════════════════════════════════════════════════════════════
#  STROKE KEYFRAMES
#  Each stroke = list of (phase_t, joint_angles_dict)
#  phase_t in [0, 1] — interpolated smoothly
#
#  Joint indices in qpos (hinge joints, each 1 DOF):
#    abdomen_z=7  abdomen_y=8  abdomen_x=9
#    hip_x_right=10  hip_z_right=11  hip_y_right=12
#    knee_right=13   ankle_y_right=14  ankle_x_right=15
#    hip_x_left=16   hip_z_left=17   hip_y_left=18
#    knee_left=19    ankle_y_left=20   ankle_x_left=21
#    shoulder1_right=22  shoulder2_right=23  elbow_right=24
#    shoulder1_left=25   shoulder2_left=26   elbow_left=27
# ══════════════════════════════════════════════════════════════════════════════

def make_neutral_qpos(nq: int) -> np.ndarray:
    """Standing T-pose: root at origin, all joints at zero."""
    q = np.zeros(nq)
    q[2] = 1.282  # root height (humanoid stands at ~1.28m)
    q[3] = 1.0    # root quaternion w=1 (identity)
    return q


def _interp(keyframes: list, t: float) -> dict:
    """
    Smoothly interpolate between keyframes.
    keyframes = [(t0, {joint: angle}), (t1, {...}), ...]
    t in [0, 1]
    """
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]

    for i in range(len(keyframes) - 1):
        t0, d0 = keyframes[i]
        t1, d1 = keyframes[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            # Smooth step
            alpha = alpha * alpha * (3 - 2 * alpha)
            result = {}
            for key in d0:
                result[key] = d0[key] * (1 - alpha) + d1.get(key, 0.0) * alpha
            return result
    return keyframes[-1][1]


# ── Forehand Topspin ──────────────────────────────────────────────────────────
# Classic TT forehand: backswing low-right → contact → follow-through high
# Full body: weight shifts right→left, right knee bends in backswing, hips rotate
#
# Joint reference (qpos indices):
#   abdomen_z=7  abdomen_y=8  abdomen_x=9
#   hip_x_right=10  hip_z_right=11  hip_y_right=12   knee_right=13
#   ankle_y_right=14  ankle_x_right=15
#   hip_x_left=16   hip_z_left=17   hip_y_left=18    knee_left=19
#   shoulder1_right=22  shoulder2_right=23  elbow_right=24
#   shoulder1_left=25   shoulder2_left=26   elbow_left=27
FOREHAND_TOPSPIN = [
    (0.00, {  # Athletic ready stance — knees slightly bent, weight centred
        # Right arm ready
        22: -0.3, 23:  0.2, 24:  0.9,
        # Left arm relaxed guard
        25: -0.1, 26: -0.1, 27:  0.6,
        # Torso neutral, slight forward lean
        7: -0.10, 8:  0.0,  9:  0.05,
        # Both knees slightly bent
        13:  0.25, 19:  0.25,
        # Hips level
        10:  0.0,  11:  0.0,  12:  0.0,
        16:  0.0,  17:  0.0,  18:  0.0,
    }),
    (0.20, {  # Backswing — weight transfers to right foot
        # Right arm swings back and low
        22: -0.9, 23:  0.5, 24:  1.4,
        # Left arm opens out for balance
        25: -0.2, 26:  0.2, 27:  0.5,
        # Torso rotates right (hip leads)
        7: -0.45, 8: -0.10, 9:  0.08,
        # Right knee bends deeper (loading), left leg extends
        13:  0.45, 19:  0.10,
        # Right hip loads: slight inward rotation + extension
        10:  0.10, 11: -0.15, 12: -0.20,
        # Left hip slight outward
        16: -0.05, 17:  0.10, 18:  0.15,
    }),
    (0.50, {  # Impact — explosive hip/torso rotation + arm extension
        # Right arm drives forward-up
        22:  0.4, 23: -0.15, 24:  0.35,
        # Left arm pulls back for counter-rotation
        25:  0.15, 26: -0.25, 27:  0.55,
        # Torso snaps left
        7:  0.30, 8:  0.12, 9: -0.08,
        # Right knee extends (push off), left knee bends (weight arrival)
        13:  0.10, 19:  0.35,
        # Right hip rotates forward
        10: -0.05, 11:  0.10, 12:  0.20,
        # Left hip accepts weight
        16:  0.08, 17: -0.10, 18: -0.15,
    }),
    (0.75, {  # Follow-through — arm high, weight fully on left
        22:  1.0, 23: -0.45, 24:  0.15,
        25:  0.25, 26: -0.15, 27:  0.50,
        7:  0.40, 8:  0.18, 9: -0.08,
        # Right leg trailing, left knee absorbs
        13:  0.05, 19:  0.40,
        10: -0.10, 11:  0.15, 12:  0.25,
        16:  0.12, 17: -0.15, 18: -0.20,
    }),
    (1.00, {  # Return to ready
        22: -0.3, 23:  0.2, 24:  0.9,
        25: -0.1, 26: -0.1, 27:  0.6,
        7: -0.10, 8:  0.0,  9:  0.05,
        13:  0.25, 19:  0.25,
        10:  0.0,  11:  0.0,  12:  0.0,
        16:  0.0,  17:  0.0,  18:  0.0,
    }),
]

# ── Backhand Drive ────────────────────────────────────────────────────────────
# Horizontal compact swing; weight shifts LEFT→RIGHT (opposite of forehand)
# Elbow leads, arm crosses body, body rotates left-to-right
BACKHAND_DRIVE = [
    (0.00, {  # Ready — weight centred, slightly left-weighted for backhand
        22: -0.15, 23: -0.10, 24:  0.80,
        25: -0.15, 26:  0.10, 27:  0.75,
        7:  0.10,  8:  0.0,   9:  0.0,
        13:  0.25, 19:  0.25,
        10:  0.0,  11:  0.0,  12:  0.0,
        16:  0.0,  17:  0.0,  18:  0.0,
    }),
    (0.22, {  # Backswing — weight shifts LEFT, torso turns left
        # Right arm tucks across body (elbow leads)
        22: -0.05, 23: -0.70, 24:  1.50,
        # Left arm draws back
        25:  0.20, 26:  0.35, 27:  1.00,
        # Torso rotates LEFT
        7:  0.40, 8:  0.08, 9:  0.10,
        # Left knee bends (loading), right leg extends
        13:  0.10, 19:  0.45,
        # Left hip loads
        16:  0.12, 17:  0.15, 18:  0.20,
        # Right hip opens slightly
        10: -0.08, 11: -0.10, 12: -0.15,
    }),
    (0.52, {  # Impact — torso drives RIGHT, arm extends horizontally
        22:  0.25, 23:  0.35, 24:  0.45,
        25: -0.10, 26: -0.20, 27:  0.65,
        # Torso snaps RIGHT
        7: -0.25, 8: -0.05, 9: -0.08,
        # Left knee extends (push), right knee bends (receives weight)
        13:  0.35, 19:  0.15,
        16: -0.05, 17: -0.12, 18: -0.18,
        10:  0.10, 11:  0.08, 12:  0.15,
    }),
    (0.78, {  # Follow-through — arm extends right-forward, weight on right
        22:  0.55, 23:  0.65, 24:  0.25,
        25: -0.25, 26: -0.10, 27:  0.70,
        7: -0.35, 8: -0.10, 9: -0.08,
        13:  0.40, 19:  0.08,
        16: -0.10, 17: -0.15, 18: -0.20,
        10:  0.15, 11:  0.12, 12:  0.20,
    }),
    (1.00, {  # Return to ready
        22: -0.15, 23: -0.10, 24:  0.80,
        25: -0.15, 26:  0.10, 27:  0.75,
        7:  0.10,  8:  0.0,   9:  0.0,
        13:  0.25, 19:  0.25,
        10:  0.0,  11:  0.0,  12:  0.0,
        16:  0.0,  17:  0.0,  18:  0.0,
    }),
]

# ── Forehand Smash ────────────────────────────────────────────────────────────
# Aggressive overhead attack: big wind-up, explosive full-body extension,
# right leg pushes hard, body rises slightly, powerful downward snap
FOREHAND_SMASH = [
    (0.00, {  # Ready — slightly wider stance, weight centred
        22: -0.30, 23:  0.25, 24:  0.85,
        25: -0.15, 26: -0.10, 27:  0.55,
        7: -0.10,  8:  0.0,   9:  0.05,
        13:  0.30, 19:  0.30,
        10:  0.0,  11:  0.0,  12:  0.0,
        16:  0.0,  17:  0.0,  18:  0.0,
    }),
    (0.18, {  # Wind-up — arm high and back, deep knee bend both legs
        # Right arm raised high behind
        22: -1.30, 23:  0.60, 24:  0.40,
        # Left arm out for balance
        25: -0.10, 26:  0.25, 27:  0.40,
        # Big torso rotation right + lean back
        7: -0.55, 8: -0.20, 9:  0.12,
        # Both knees deep bend (coiling)
        13:  0.60, 19:  0.55,
        # Right hip loads strongly
        10:  0.15, 11: -0.20, 12: -0.30,
        16: -0.08, 17:  0.12, 18:  0.20,
    }),
    (0.40, {  # Launch — legs begin to extend, body rises
        22: -1.05, 23:  0.30, 24:  0.25,
        25:  0.05, 26:  0.15, 27:  0.50,
        7: -0.45, 8: -0.15, 9:  0.08,
        # Knees starting to extend (explosive push)
        13:  0.35, 19:  0.30,
        10:  0.05, 11: -0.10, 12: -0.15,
        16: -0.05, 17:  0.08, 18:  0.12,
    }),
    (0.58, {  # Impact — full body extension, arm snaps down-forward hard
        22:  0.70, 23: -0.25, 24:  0.20,
        25:  0.25, 26: -0.15, 27:  0.60,
        # Explosive torso snap left + forward
        7:  0.55, 8:  0.25, 9: -0.12,
        # Legs nearly extended (push-off)
        13:  0.05, 19:  0.08,
        10: -0.10, 11:  0.15, 12:  0.25,
        16:  0.12, 17: -0.15, 18: -0.20,
    }),
    (0.78, {  # Follow-through — arm swings through low, weight lands forward
        22:  0.90, 23: -0.55, 24:  0.55,
        25:  0.30, 26: -0.20, 27:  0.55,
        7:  0.45, 8:  0.15, 9: -0.08,
        # Knees re-bend to absorb landing
        13:  0.35, 19:  0.40,
        10: -0.08, 11:  0.12, 12:  0.20,
        16:  0.10, 17: -0.12, 18: -0.18,
    }),
    (1.00, {  # Return
        22: -0.3, 23:  0.3, 24:  0.8,
        25: -0.2, 26: -0.1, 27:  0.5,
        7:  -0.1, 8:  0.0, 9:  0.0,
    }),
]

# ── No Stroke — athletic ready position with natural weight-shift sway ────────
# Player stands in TT ready stance: knees bent, slight forward lean, arms up
NO_STROKE = [
    (0.00, {
        22: -0.20, 23:  0.10, 24:  0.85, 25: -0.15, 26:  0.05, 27:  0.70,
        7:  -0.08,  8:  0.00,  9:  0.05,
        13:  0.28,  19:  0.28,
        10:  0.0,  11:  0.0,  12:  0.0, 16:  0.0, 17:  0.0, 18:  0.0,
    }),
    (0.25, {  # Slight sway right
        22: -0.22, 23:  0.12, 24:  0.85, 25: -0.12, 26:  0.05, 27:  0.70,
        7:  -0.12,  8:  0.02,  9:  0.03,
        13:  0.30,  19:  0.24,  # right knee slightly more bent
        10:  0.03, 11: -0.04, 12: -0.05, 16: -0.02, 17:  0.04, 18:  0.05,
    }),
    (0.50, {  # Back to centre
        22: -0.20, 23:  0.10, 24:  0.85, 25: -0.15, 26:  0.05, 27:  0.70,
        7:  -0.08,  8:  0.00,  9:  0.05,
        13:  0.28,  19:  0.28,
        10:  0.0,  11:  0.0,  12:  0.0, 16:  0.0, 17:  0.0, 18:  0.0,
    }),
    (0.75, {  # Slight sway left
        22: -0.18, 23:  0.08, 24:  0.85, 25: -0.18, 26:  0.06, 27:  0.70,
        7:  -0.05,  8: -0.02,  9:  0.07,
        13:  0.24,  19:  0.30,  # left knee slightly more bent
        10: -0.02, 11:  0.04, 12:  0.05, 16:  0.03, 17: -0.04, 18: -0.05,
    }),
    (1.00, {
        22: -0.20, 23:  0.10, 24:  0.85, 25: -0.15, 26:  0.05, 27:  0.70,
        7:  -0.08,  8:  0.00,  9:  0.05,
        13:  0.28,  19:  0.28,
        10:  0.0,  11:  0.0,  12:  0.0, 16:  0.0, 17:  0.0, 18:  0.0,
    }),
]

STROKE_KEYFRAMES = {
    0: NO_STROKE,
    1: FOREHAND_TOPSPIN,
    2: BACKHAND_DRIVE,
    3: FOREHAND_SMASH,
}

STROKE_NAMES = {
    0: "NoStroke",
    1: "ForehandTopspin",
    2: "BackhandDrive",
    3: "ForehandSmash",
}


# ══════════════════════════════════════════════════════════════════════════════
#  QUATERNION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """MuJoCo uses [w,x,y,z]; our CSV uses [x,y,z,w]."""
    return np.array([q[1], q[2], q[3], q[0]])

def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  IMU EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_imu(model: mujoco.MjModel, data: mujoco.MjData,
                body_id: int,
                prev_vel: np.ndarray | None,
                dt: float) -> tuple:
    """
    Extract virtual IMU readings for one body.

    Returns:
        quat_xyzw  : np.ndarray shape (4,)   [x,y,z,w]
        accel      : np.ndarray shape (3,)   body-frame linear accel (m/s²)
        gyro       : np.ndarray shape (3,)   body-frame angular vel (rad/s)
        vel_world  : np.ndarray shape (3,)   world-frame linear velocity (for next step)
    """
    # World-frame orientation quaternion [w,x,y,z]
    q_wxyz = data.xquat[body_id].copy()
    quat_xyzw = wxyz_to_xyzw(q_wxyz)

    # Rotation matrix: world → body frame
    R = quat_to_rotmat(q_wxyz)   # body columns in world frame
    R_T = R.T                    # world → body

    # World-frame velocity from subtree (6DOF: [lin, ang] for body root)
    # data.cvel shape: (nbody, 6) → [angular_world(3), linear_world(3)]
    cvel = data.cvel[body_id]
    ang_world = cvel[:3]
    lin_world = cvel[3:]

    # Angular velocity → body frame (gyro)
    gyro = R_T @ ang_world

    # Linear acceleration → finite difference of world velocity → body frame
    if prev_vel is not None:
        lin_accel_world = (lin_world - prev_vel) / dt
    else:
        lin_accel_world = np.zeros(3)

    # Add gravity (MuJoCo gravity is [0,0,-9.81] by default)
    gravity_world = np.array([0.0, 0.0, -9.81])
    lin_accel_world -= gravity_world  # IMU measures reaction force, subtract gravity

    accel = R_T @ lin_accel_world

    # Add realistic sensor noise — configurable via globals so MAIN can tune
    accel += np.random.normal(0, ACCEL_NOISE_STD, 3)
    gyro  += np.random.normal(0, GYRO_NOISE_STD,  3)

    return quat_xyzw, accel, gyro, lin_world.copy()


# ── Per-run noise globals (set by main from CLI) ─────────────────────────────
ACCEL_NOISE_STD = 0.05      # m/s²
GYRO_NOISE_STD  = 0.005     # rad/s


# ══════════════════════════════════════════════════════════════════════════════
#  STROKE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_stroke(model: mujoco.MjModel, data: mujoco.MjData,
                    stroke_class: int, frames: int,
                    noise_scale: float,
                    body_ids: dict,
                    amplitude_jitter: float = 0.0,
                    speed_jitter: float = 0.0) -> list:
    """
    Simulate one stroke repetition.

    amplitude_jitter : per-rep multiplicative jitter applied to ALL joint
                       angles (e.g. 0.10 → amplitude ~ N(1, 0.10)).
    speed_jitter     : per-rep multiplicative jitter on the time index
                       (e.g. 0.10 → effective duration ~ N(1, 0.10)).
    Returns list of row dicts (one per frame).
    """
    keyframes = STROKE_KEYFRAMES[stroke_class]
    q0        = make_neutral_qpos(model.nq)
    rows      = []

    # Per-body previous velocity buffer for accel computation
    prev_vels = {imu_id: None for imu_id in range(17)}

    dt = model.opt.timestep

    # Per-rep augmentation factors (one draw per stroke)
    amp_factor   = (1.0 + np.random.normal(0.0, amplitude_jitter)
                    if amplitude_jitter > 0 else 1.0)
    speed_factor = (1.0 + np.random.normal(0.0, speed_jitter)
                    if speed_jitter > 0 else 1.0)
    speed_factor = float(np.clip(speed_factor, 0.6, 1.4))   # avoid extreme stretch

    for frame in range(frames):
        # Apply per-rep speed warp by re-mapping the phase t
        raw_t = frame / max(frames - 1, 1)
        t = float(np.clip(raw_t * speed_factor, 0.0, 1.0))

        # Interpolate joint angles from keyframes
        joints = _interp(keyframes, t)

        # Build qpos with per-rep amplitude scale + per-frame angle noise
        qpos = q0.copy()
        for jnt_idx, angle in joints.items():
            noise = np.random.normal(0, noise_scale)
            qpos[jnt_idx] = angle * amp_factor + noise

        # Set state and step forward
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Build one CSV row
        row = {
            "timestamp": f"{frame * dt:.4f}",
            "stroke_label": stroke_class,
            "stroke_name":  STROKE_NAMES[stroke_class],
        }

        for imu_id in range(17):
            body_name = IMU_TO_BODY[imu_id]
            bid       = body_ids[body_name]
            qxyzw, accel, gyro, vel = extract_imu(model, data, bid,
                                                   prev_vels[imu_id], dt)
            prev_vels[imu_id] = vel

            p = f"imu_{imu_id}_"
            row[p + "quat_x"] = f"{qxyzw[0]:.6f}"
            row[p + "quat_y"] = f"{qxyzw[1]:.6f}"
            row[p + "quat_z"] = f"{qxyzw[2]:.6f}"
            row[p + "quat_w"] = f"{qxyzw[3]:.6f}"
            row[p + "accel_x"] = f"{accel[0]:.6f}"
            row[p + "accel_y"] = f"{accel[1]:.6f}"
            row[p + "accel_z"] = f"{accel[2]:.6f}"
            row[p + "gyro_x"]  = f"{gyro[0]:.6f}"
            row[p + "gyro_y"]  = f"{gyro[1]:.6f}"
            row[p + "gyro_z"]  = f"{gyro[2]:.6f}"

        rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic TT IMU data with MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--reps",    default=30,  type=int,
                        help="Stroke repetitions per class (default 30)")
    parser.add_argument("--fps",     default=60,  type=int,
                        help="Frames per stroke (default 60)")
    parser.add_argument("--sample_rate", default=200, type=int,
                        help="MuJoCo simulation rate Hz (200 default). "
                             "Set to 100 to match SiriusCeption real IMU rate.")
    parser.add_argument("--noise",   default=0.02, type=float,
                        help="Joint angle noise std dev in radians")
    parser.add_argument("--accel_noise", default=0.05, type=float,
                        help="Per-sample accelerometer noise std (m/s²)")
    parser.add_argument("--gyro_noise",  default=0.005, type=float,
                        help="Per-sample gyroscope noise std (rad/s)")
    parser.add_argument("--amplitude_jitter", default=0.0, type=float,
                        help="Per-rep multiplicative jitter on joint angles "
                             "(0.10 = ±10%% amplitude variation)")
    parser.add_argument("--speed_jitter", default=0.0, type=float,
                        help="Per-rep multiplicative jitter on stroke duration")
    parser.add_argument("--out",     default=str(OUT_DIR / "tt_synthetic.csv"),
                        help="Output CSV path")
    parser.add_argument("--model",   default=str(MODEL_PATH),
                        help="Path to humanoid.xml")
    parser.add_argument("--seed",    default=42, type=int,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Wire the per-sample noise globals so extract_imu() uses CLI values
    global ACCEL_NOISE_STD, GYRO_NOISE_STD
    ACCEL_NOISE_STD = args.accel_noise
    GYRO_NOISE_STD  = args.gyro_noise

    print(f"[INFO] Loading model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)

    # Override the physics timestep to match the desired sample rate
    model.opt.timestep = 1.0 / args.sample_rate
    print(f"[INFO] MuJoCo timestep set to {model.opt.timestep:.5f}s "
          f"({args.sample_rate} Hz)")

    # Build body name → body ID lookup
    body_ids = {}
    for imu_id, body_name in IMU_TO_BODY.items():
        if body_name not in body_ids:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                print(f"[WARN] Body '{body_name}' not found in model!")
            body_ids[body_name] = bid
    print(f"[INFO] Body IDs resolved: {len(body_ids)} unique bodies")

    # Build CSV columns
    fieldnames = ["timestamp", "stroke_label", "stroke_name"]
    for imu_id in range(17):
        p = f"imu_{imu_id}_"
        for s in ["quat_x", "quat_y", "quat_z", "quat_w",
                  "accel_x", "accel_y", "accel_z",
                  "gyro_x",  "gyro_y",  "gyro_z"]:
            fieldnames.append(p + s)

    # Generate all strokes
    all_rows  = []
    class_counts = {c: 0 for c in range(4)}

    stroke_order = []
    for cls in range(4):
        stroke_order.extend([cls] * args.reps)

    # Shuffle so strokes are interleaved (more realistic session)
    np.random.shuffle(stroke_order)

    total = len(stroke_order)
    print(f"[INFO] Generating {total} stroke repetitions × {args.fps} frames "
          f"= {total * args.fps:,} rows")
    print(f"[INFO] Classes: 0=NoStroke, 1=ForehandTopspin, "
          f"2=BackhandDrive, 3=ForehandSmash")

    t0 = time.perf_counter()
    for i, cls in enumerate(stroke_order):
        rows = simulate_stroke(
            model, data, cls, args.fps, args.noise, body_ids,
            amplitude_jitter=args.amplitude_jitter,
            speed_jitter=args.speed_jitter,
        )
        all_rows.extend(rows)
        class_counts[cls] += 1

        if (i + 1) % 20 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1:>4}/{total}] {STROKE_NAMES[cls]:<20}  "
                  f"total rows: {len(all_rows):>7,}  ({elapsed:.1f}s)")

    # Write CSV
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.perf_counter() - t0
    print(f"\n[INFO] Done in {elapsed:.1f}s")
    print(f"[INFO] Output: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"[INFO] Total rows: {len(all_rows):,}")
    print(f"[INFO] Class distribution:")
    for cls, count in class_counts.items():
        print(f"         {cls} {STROKE_NAMES[cls]:<22}: "
              f"{count} reps × {args.fps} frames = {count * args.fps:,} rows")

    print(f"\n[INFO] To stream to Unity:")
    print(f"         python stream_any_csv.py --dataset zuyan --csv {out_path}")
    print(f"\n[INFO] To run stroke classifier (after training):")
    print(f"         python run_classifier.py --csv {out_path}")


if __name__ == "__main__":
    main()
