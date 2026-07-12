"""
kinematics.py
─────────────
Joint-angle computation from adjacent IMU quaternions.

IMU sensor layout (Zuyan / MuJoCo):
    7  = RightHand       (wrist — NOT actuated in MuJoCo; identical to imu_8)
    8  = RightLowerArm   (forearm)
    9  = RightUpperArm   (upper arm)
    13 = Hips
    14 = Spine
    15 = RightUpperArm proxy (RightShoulder)
    16 = LeftUpperArm  proxy (LeftShoulder)

Three signals computed:
    elbow       : imu_9  × imu_8   (upper arm → forearm) — total 3-D angle
    forearm_roll: imu_8  alone     — axial roll of the forearm around its
                                     long axis (yaw component of world quat)
    torso_yaw   : yaw(imu_14) − yaw(imu_13)
                                   — axial rotation of spine vs hips,
                                     i.e. the "torso twist" coaches care about

NOTE: wrist (imu_8 vs imu_7) is intentionally omitted.
      MuJoCo's default humanoid has no wrist DoF — imu_7 and imu_8 are
      identical, so the wrist angle is always 0°. This is a known simulator
      limitation and is discussed in the thesis (Section 6.X).
"""

import numpy as np


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _quat_norm(q: np.ndarray) -> np.ndarray:
    """Normalise a quaternion [x, y, z, w]."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else np.array([0., 0., 0., 1.])


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of [x, y, z, w]  →  [-x, -y, -z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both in [x, y, z, w] convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _yaw_deg(q: np.ndarray) -> float:
    """
    Extract yaw (rotation about world Z-axis) from a unit quaternion [x,y,z,w].
    Returns degrees in [-180, 180].
    """
    q = _quat_norm(q)
    x, y, z, w = q
    yaw_rad = np.arctan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z))
    return float(np.degrees(yaw_rad))


def relative_angle_deg(q_parent: np.ndarray, q_child: np.ndarray) -> float:
    """
    Total 3-D angle (degrees) between two bones given world-frame quaternions.
    Uses the half-angle formula: angle = 2 × arccos(|q_rel.w|).
    """
    q_rel = _quat_multiply(_quat_conjugate(_quat_norm(q_parent)),
                           _quat_norm(q_child))
    q_rel = _quat_norm(q_rel)
    w = float(np.clip(q_rel[3], -1.0, 1.0))
    return round(2.0 * np.degrees(np.arccos(abs(w))), 3)


# ── Row → quaternion helpers ───────────────────────────────────────────────────

def _quat_from_row(row: dict, imu_id: int) -> np.ndarray:
    """Extract [x, y, z, w] quaternion for one IMU from a CSV row dict."""
    return np.array([
        float(row[f"imu_{imu_id}_quat_x"]),
        float(row[f"imu_{imu_id}_quat_y"]),
        float(row[f"imu_{imu_id}_quat_z"]),
        float(row[f"imu_{imu_id}_quat_w"]),
    ])


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_joint_angles(row: dict) -> dict:
    """
    Compute elbow angle, forearm roll, and torso yaw from a single CSV row.

    Returns:
        {
            "elbow"       : float  — total 3-D elbow angle (deg)
            "forearm_roll": float  — forearm axial yaw in world frame (deg)
            "torso_yaw"   : float  — spine yaw minus hips yaw (deg)
        }
    """
    try:
        q_upper = _quat_from_row(row, 9)   # RightUpperArm
        q_fore  = _quat_from_row(row, 8)   # RightLowerArm
        q_hips  = _quat_from_row(row, 13)  # Hips
        q_spine = _quat_from_row(row, 14)  # Spine

        elbow        = relative_angle_deg(q_upper, q_fore)
        forearm_roll = _yaw_deg(q_fore)
        torso        = relative_angle_deg(q_hips, q_spine)

    except (KeyError, ValueError):
        elbow = forearm_roll = torso = 0.0

    return {
        "elbow"       : round(elbow, 3),
        "forearm_roll": round(forearm_roll, 3),
        "torso"       : round(torso, 3),
    }


def compute_joint_angle_series(rows: list) -> dict:
    """
    Compute joint angle time series over a window of rows.

    Returns:
        {
            "elbow"       : np.ndarray (N,)  degrees
            "forearm_roll": np.ndarray (N,)  degrees
            "torso_yaw"   : np.ndarray (N,)  degrees
        }
    """
    keys   = ("elbow", "forearm_roll", "torso")
    series = {k: [] for k in keys}
    for row in rows:
        angles = compute_joint_angles(row)
        for k in keys:
            series[k].append(angles[k])
    return {k: np.array(v) for k, v in series.items()}


def kinematic_summary(rows: list, fps: float = 60.0) -> dict:
    """
    Compute kinematic statistics over a stroke window.

    Returns:
        {
            "elbow_peak_angle"     : float  (deg)
            "elbow_range"          : float  (deg, max-min)
            "elbow_peak_velocity"  : float  (deg/s)
            "time_to_elbow_peak"   : float  (seconds from window start)
            "forearm_roll_range"   : float  (deg, max-min)
            "torso_yaw_range"      : float  (deg, max-min)
        }
    """
    series = compute_joint_angle_series(rows)

    elbow  = series["elbow"]
    froll  = series["forearm_roll"]
    torso  = series["torso"]

    elbow_vel      = np.abs(np.diff(elbow) * fps)
    elbow_peak_idx = int(np.argmax(elbow))

    return {
        "elbow_peak_angle"   : round(float(np.max(elbow)), 2),
        "elbow_range"        : round(float(np.ptp(elbow)), 2),
        "elbow_peak_velocity": round(float(np.max(elbow_vel)) if len(elbow_vel) > 0 else 0.0, 2),
        "time_to_elbow_peak" : round(elbow_peak_idx / fps, 3),
        "forearm_roll_range" : round(float(np.ptp(froll)), 2),
        "torso_range"        : round(float(np.ptp(torso)), 2),
    }
