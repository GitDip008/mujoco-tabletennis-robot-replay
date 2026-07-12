import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
import pickle
import os

# ── Label map ──────────────────────────────────────────────────────────────
# testmode 0 → teststage 0 → class 0 (No-Stroke A)
# testmode 1 → teststage 1 → class 1 (Stroke Type 1)
# testmode 1 → teststage 2 → class 2 (Stroke Type 2)
# testmode 1 → teststage 3 → class 3 (Stroke Type 3)
# testmode 2 → teststage 0 → class 0 (No-Stroke, merged with class 0)

CLASS_NAMES = {
    0: "No Stroke",
    1: "Stroke Type 1",
    2: "Stroke Type 2",
    3: "Stroke Type 3",
}

FEATURE_COLS = [
    "ax_mean", "ay_mean", "az_mean",
    "gx_mean", "gy_mean", "gz_mean",
    "ax_var",  "ay_var",  "az_var",
    "gx_var",  "gy_var",  "gz_var",
    "ax_rms",  "ay_rms",  "az_rms",
    "gx_rms",  "gy_rms",  "gz_rms",
    "a_max",   "a_mean",  "a_min",
    "g_max",   "g_mean",  "g_min",
    "a_fft",   "g_fft",
    "a_psdx",  "g_psdx",
    "a_kurt",  "g_kurt",
    "a_skewn", "g_skewn",
    "a_entropy","g_entropy",
]


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TTSwing CSV and return:
        df       – full dataframe
        X        – feature matrix (N, 34)  float32
        y        – labels (N,)             int64
        groups   – subject IDs (N,)        int64  - for LOSO
    """
    df = pd.read_csv(csv_path)

    # testmode=2, teststage=0 → same class 0 (no-stroke), already correct
    y = df["teststage"].values.astype(np.int64)

    X = df[FEATURE_COLS].values.astype(np.float32)
    groups = df["id"].values.astype(np.int64)

    return df, X, y, groups


def get_loso_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """
    Generator → yields (X_train, X_val, y_train, y_val, test_subject_id)
    for each Leave-One-Subject-Out fold.
    Each fold: scaler is fit ONLY on train split to prevent data leakage.
    """
    loso = LeaveOneGroupOut()
    for train_idx, val_idx in loso.split(X, y, groups):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val         = y[train_idx],  y[val_idx]
        subject_id             = groups[val_idx][0]

        # Fit scaler on train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val   = scaler.transform(X_val_raw)

        yield X_train, X_val, y_train, y_val, subject_id, scaler


def save_scaler(scaler: StandardScaler, path: str = "checkpoints/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = "checkpoints/scaler.pkl") -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    df, X, y, groups = load_and_prepare(r"E:\thesis_work\TT_thesis\tt_coaching_pipeline\data\raw\TTSWING.csv")
    print(f"X shape     : {X.shape}")
    print(f"y shape     : {y.shape}")
    print(f"Num subjects: {len(np.unique(groups))}")
    print(f"Class dist  : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Features    : {FEATURE_COLS}")