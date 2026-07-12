"""
Sliding-window feature extraction from raw IMU CSV rows.

Raw IMU CSV columns for each sensor:
    imu_{id}_accel_x/y/z  (m/s²)
    imu_{id}_gyro_x/y/z   (deg/s)
    imu_{id}_quat_x/y/z/w (quaternion)

The 34 extracted features match FEATURE_COLS in inference.py.

NOTE: The existing MLP checkpoints were trained on the TTSwing dataset which
used a different sensor unit scale (raw ADC counts, ~1000x larger magnitudes).
Scale-independent features (kurtosis, skewness, entropy) will transfer well.
Scale-dependent features (mean, RMS, variance, FFT power) will differ and may
need a new model trained on raw-IMU data for accurate stroke classification.
"""
import numpy as np
from collections import deque
from scipy.stats import kurtosis, skew
from scipy.signal import periodogram

from signal_filter import lowpass, DEFAULT_FS, DEFAULT_CUTOFF

RACKET_IMU_ID = 7       # RightHand — primary stroke sensor
WINDOW_SIZE   = 50      # frames per feature window
STEP_SIZE     = 10      # compute a new prediction every N frames
APPLY_FILTER  = True    # apply Butterworth low-pass before aggregation

# Energy gate: a window whose total accel-axis variance is below this is
# treated as "no motion" → forced to No Stroke without trusting the MLP.
# Units are (m/s²)². Rest windows sit near ~0.01; real swings are tens+.
ENERGY_GATE_THRESHOLD = 0.5


def window_energy(features: np.ndarray) -> float:
    """
    Motion-energy proxy = sum of the three accelerometer-axis variances.
    FEATURE_COLS indices 6,7,8 = ax_var, ay_var, az_var.
    """
    return float(features[6] + features[7] + features[8])


def is_idle_window(features: np.ndarray) -> bool:
    """True when the window carries negligible motion (player at rest)."""
    return window_energy(features) < ENERGY_GATE_THRESHOLD


def _spectral_entropy(signal: np.ndarray) -> float:
    psd = np.abs(np.fft.rfft(signal)) ** 2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))


def _safe_kurtosis(signal: np.ndarray) -> float:
    # kurtosis/skew are undefined for a (near-)constant signal — return 0
    if np.std(signal) < 1e-8:
        return 0.0
    return float(kurtosis(signal))


def _safe_skew(signal: np.ndarray) -> float:
    if np.std(signal) < 1e-8:
        return 0.0
    return float(skew(signal))


def extract_features_from_window(rows: list) -> np.ndarray:
    """
    rows : list of dict, length == WINDOW_SIZE, each a csv.DictReader row.
    Returns np.ndarray shape (34,) matching FEATURE_COLS order.
    """
    imu = RACKET_IMU_ID
    ax = np.array([float(r[f"imu_{imu}_accel_x"]) for r in rows], dtype=np.float64)
    ay = np.array([float(r[f"imu_{imu}_accel_y"]) for r in rows], dtype=np.float64)
    az = np.array([float(r[f"imu_{imu}_accel_z"]) for r in rows], dtype=np.float64)
    gx = np.array([float(r[f"imu_{imu}_gyro_x"])  for r in rows], dtype=np.float64)
    gy = np.array([float(r[f"imu_{imu}_gyro_y"])  for r in rows], dtype=np.float64)
    gz = np.array([float(r[f"imu_{imu}_gyro_z"])  for r in rows], dtype=np.float64)

    if APPLY_FILTER:
        ax = lowpass(ax); ay = lowpass(ay); az = lowpass(az)
        gx = lowpass(gx); gy = lowpass(gy); gz = lowpass(gz)

    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    _, a_psd = periodogram(a_mag)
    _, g_psd = periodogram(g_mag)

    feats = [
        # axis means
        float(np.mean(ax)),  float(np.mean(ay)),  float(np.mean(az)),
        float(np.mean(gx)),  float(np.mean(gy)),  float(np.mean(gz)),
        # axis variances
        float(np.var(ax)),   float(np.var(ay)),   float(np.var(az)),
        float(np.var(gx)),   float(np.var(gy)),   float(np.var(gz)),
        # axis RMS
        float(np.sqrt(np.mean(ax**2))), float(np.sqrt(np.mean(ay**2))), float(np.sqrt(np.mean(az**2))),
        float(np.sqrt(np.mean(gx**2))), float(np.sqrt(np.mean(gy**2))), float(np.sqrt(np.mean(gz**2))),
        # magnitude stats
        float(np.max(a_mag)), float(np.mean(a_mag)), float(np.min(a_mag)),
        float(np.max(g_mag)), float(np.mean(g_mag)), float(np.min(g_mag)),
        # FFT total power of magnitude signal
        float(np.sum(np.abs(np.fft.rfft(a_mag)) ** 2)),
        float(np.sum(np.abs(np.fft.rfft(g_mag)) ** 2)),
        # PSD peak
        float(np.max(a_psd)),
        float(np.max(g_psd)),
        # kurtosis
        _safe_kurtosis(a_mag),
        _safe_kurtosis(g_mag),
        # skewness
        _safe_skew(a_mag),
        _safe_skew(g_mag),
        # spectral entropy
        _spectral_entropy(a_mag),
        _spectral_entropy(g_mag),
    ]

    assert len(feats) == 34, f"Bug: expected 34 features, got {len(feats)}"
    # Final safety net: replace any NaN/inf (e.g. from flat windows) with finite values
    arr = np.array(feats, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


class SlidingWindowExtractor:
    """
    Accumulates raw IMU rows and yields a 34-dim feature vector every
    STEP_SIZE frames once the buffer is filled to WINDOW_SIZE.
    """

    def __init__(self, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE):
        self.window_size = window_size
        self.step_size   = step_size
        self._buffer     = deque(maxlen=window_size)
        self._frame_count = 0

    def add_frame(self, row: dict) -> "np.ndarray | None":
        """
        Returns extracted feature vector (34,) when a prediction is due,
        otherwise returns None.
        """
        self._buffer.append(row)
        self._frame_count += 1

        if (len(self._buffer) == self.window_size
                and self._frame_count % self.step_size == 0):
            return extract_features_from_window(list(self._buffer))
        return None

    def reset(self):
        self._buffer.clear()
        self._frame_count = 0
