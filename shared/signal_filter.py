"""
Shared signal-processing helpers.

A Butterworth low-pass is applied to the raw accel / gyro arrays before
the 34-feature aggregation, both during training (train_baselines.py)
and during inference (feature_extractor.py). Keeping the same filter on
both sides ensures the train-time and run-time feature distributions
agree.
"""
import numpy as np
from scipy.signal import butter, filtfilt

# ── Defaults (used by feature_extractor.py and train_baselines.py) ──
DEFAULT_FS      = 100.0     # Hz — matches new MuJoCo + SiriusCeption recordings
DEFAULT_CUTOFF  = 20.0      # Hz — well above human stroke content (~10 Hz)
DEFAULT_ORDER   = 4


def lowpass(signal: np.ndarray,
            fs: float = DEFAULT_FS,
            cutoff: float = DEFAULT_CUTOFF,
            order: int = DEFAULT_ORDER) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass for IMU channels.

    filtfilt's pad length is ~3*order — the input must be longer than
    that, otherwise it raises. For very short windows (cold-start) we
    silently return the input unchanged.
    """
    n = len(signal)
    min_len = 3 * (order + 1)
    if n < min_len:
        return signal
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, signal)
