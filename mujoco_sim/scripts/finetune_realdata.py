"""
Fine-tune the MuJoCo synthetic-trained MLP on real SiriusCeption recordings
so the classifier closes the sim-to-real domain gap.

Inputs (real CSVs from E:\\thesis_work\\TT_thesis\\imu_data\\recordings):
    tpose_rest_*.csv   → label 0 (NoStroke)
    fh_topspin_*.csv   → label 1 (ForehandTopspin)
    bh_drive_*.csv     → label 2 (BackhandDrive)
    fh_smash_*.csv     → label 3 (ForehandSmash)

For stroke files we skip the first SKIP_SEC seconds (the T-pose calibration
region at the start of every recording).

Training:
    - Slide a 50-frame window with step=10 across slot 7 of each CSV
    - Extract the 34 features (same filter as train_baselines.py)
    - Combine real-data features with the full synthetic dataset
    - Re-fit StandardScaler on the combined data
    - Fine-tune the existing MLP for FT_EPOCHS epochs at LR=FT_LR
    - Sample weight: real windows get REAL_WEIGHT× the synthetic weight
      so the model emphasises real data without forgetting synthetic structure
    - Save with descriptive timestamped + accuracy name
"""
import argparse
import csv
import pathlib
import pickle
import sys
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

# Reuse helpers from train_baselines.py
HERE          = pathlib.Path(__file__).resolve().parent
ROOT          = HERE.parent.parent
SRC_DIR       = ROOT / "src"
OUT_DIR       = HERE.parent / "output"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC_DIR))

from train_baselines import (
    MLP, train_mlp, predict_mlp, extract_features, FEATURE_COLS, CLASS_NAMES,
    DEVICE,
)
from signal_filter import DEFAULT_FS
import train_baselines as tb_mod   # for monkey-patching APPLY_FILTER/FS

# ── Config ────────────────────────────────────────────────────────────────────
RECORDINGS_DIR    = pathlib.Path(r"E:\thesis_work\TT_thesis\imu_data\recordings")
DEFAULT_SYN_CSV   = OUT_DIR / "tt_synthetic_100hz_500reps_aug.csv"
DEFAULT_BASE_MODEL  = OUT_DIR / "model_synthetic.pt"
DEFAULT_BASE_SCALER = OUT_DIR / "scaler_synthetic.pkl"

WINDOW_SIZE   = 50
STEP_SIZE     = 10
SKIP_SEC      = 10              # skip first N seconds of stroke recordings (T-pose)
FS            = 100             # SiriusCeption sample rate
REAL_WEIGHT   = 5.0             # real-data weight multiplier vs synthetic
FT_EPOCHS     = 40
FT_LR         = 5e-4


LABEL_FROM_PREFIX = {
    "tpose_rest": 0,
    "fh_topspin": 1,
    "bh_drive"  : 2,
    "fh_smash"  : 3,
}


def label_from_filename(name: str) -> int:
    for prefix, lab in LABEL_FROM_PREFIX.items():
        if name.startswith(prefix):
            return lab
    raise ValueError(f"Cannot infer label from filename: {name}")


def latest_per_class(dir_path: pathlib.Path) -> dict:
    """Legacy: pick the newest CSV per prefix (kept for reference)."""
    chosen = {}
    for fp in sorted(dir_path.glob("*.csv"), key=lambda p: p.stat().st_mtime):
        for prefix in LABEL_FROM_PREFIX:
            if fp.name.startswith(prefix):
                chosen[prefix] = fp
                break
    return chosen


def all_per_class(dir_path: pathlib.Path) -> dict:
    """Pick ALL CSVs per prefix — gives the model multi-session diversity.
    Skips internal debug dumps (filenames starting with '_')."""
    chosen = {p: [] for p in LABEL_FROM_PREFIX}
    for fp in sorted(dir_path.glob("*.csv"), key=lambda p: p.stat().st_mtime):
        if fp.name.startswith("_"):
            continue
        for prefix in LABEL_FROM_PREFIX:
            if fp.name.startswith(prefix):
                chosen[prefix].append(fp)
                break
    return chosen


def window_real_csv(fp: pathlib.Path, label: int, skip_sec: float):
    """Slide windows across the file, return (features, labels)."""
    with open(fp, newline="") as f:
        rows = list(csv.DictReader(f))

    skip_frames = int(skip_sec * FS) if label != 0 else 0  # NoStroke uses whole file
    rows = rows[skip_frames:]

    X, y = [], []
    buf = deque(maxlen=WINDOW_SIZE)
    for i, row in enumerate(rows):
        buf.append(row)
        if len(buf) == WINDOW_SIZE and (i + 1) % STEP_SIZE == 0:
            X.append(extract_features(list(buf)))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def window_synthetic(fp: pathlib.Path):
    """Same windowing recipe used by train_baselines.load_synthetic()."""
    with open(fp, newline="") as f:
        rows = list(csv.DictReader(f))
    X, y = [], []
    buf = deque(maxlen=WINDOW_SIZE)
    for i, row in enumerate(rows):
        buf.append(row)
        if len(buf) == WINDOW_SIZE and (i + 1) % STEP_SIZE == 0:
            X.append(extract_features(list(buf)))
            label = int(np.bincount([int(r["stroke_label"]) for r in buf]).argmax())
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def train_weighted_mlp(model: MLP,
                       X_tr, y_tr, w_tr,
                       X_real_val, y_real_val,
                       epochs: int, lr: float, batch_size: int = 256):
    """
    Fine-tune `model` on weighted (synthetic + real) data.  Validation is
    performed on the held-out real-data slice ONLY so the reported accuracy
    reflects real-world performance.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss(reduction="none")

    Xv = torch.tensor(X_real_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_real_val, dtype=torch.long).to(DEVICE)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(w_tr, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )

    best_acc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds_val = model(Xv).argmax(1).cpu().numpy()
        acc = accuracy_score(y_real_val, preds_val)
        f1  = f1_score(y_real_val, preds_val, average="macro",
                       labels=[0,1,2,3], zero_division=0)
        msg = f"  Ep {ep+1:>3}/{epochs}  val_acc={acc:.4f}  val_f1={f1:.4f}"
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
            msg += "  *"
        else:
            wait += 1
        print(msg)
        if wait >= 10:
            print(f"  Early stop at epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    return model, best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic_csv", default=str(DEFAULT_SYN_CSV))
    p.add_argument("--base_model",    default=str(DEFAULT_BASE_MODEL))
    p.add_argument("--base_scaler",   default=str(DEFAULT_BASE_SCALER))
    p.add_argument("--epochs",        default=FT_EPOCHS, type=int)
    p.add_argument("--lr",            default=FT_LR, type=float)
    p.add_argument("--real_weight",   default=REAL_WEIGHT, type=float)
    p.add_argument("--update_canonical", action="store_true",
                   help="Overwrite model_synthetic.pt + scaler_synthetic.pkl")
    args = p.parse_args()

    # Make sure the imported feature extractor uses the same filter the
    # baseline pipeline does (matches inference path in feature_extractor.py)
    tb_mod.APPLY_FILTER = True
    tb_mod.FILTER_FS    = FS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp : {ts}")
    print(f"Synthetic CSV : {args.synthetic_csv}")
    print(f"Recordings dir: {RECORDINGS_DIR}")

    # ── 1. Find ALL real CSVs per class (multi-session diversity) ────────
    chosen = all_per_class(RECORDINGS_DIR)
    print("\nReal CSVs picked (ALL files per class):")
    for k, files in chosen.items():
        print(f"  {k}:")
        for fp in files:
            print(f"    - {fp.name}")
    missing = [k for k, files in chosen.items() if not files]
    if missing:
        raise SystemExit(f"\nMissing real recordings for: {sorted(missing)}")

    # ── 2. Window every real CSV ─────────────────────────────────────────
    real_X_parts, real_y_parts = [], []
    for prefix, files in chosen.items():
        lab = LABEL_FROM_PREFIX[prefix]
        for fp in files:
            Xp, yp = window_real_csv(fp, lab, skip_sec=SKIP_SEC)
            print(f"  {fp.name}: {len(Xp)} windows  (label {lab} = {CLASS_NAMES[lab]})")
            real_X_parts.append(Xp); real_y_parts.append(yp)
    X_real = np.concatenate(real_X_parts, axis=0)
    y_real = np.concatenate(real_y_parts, axis=0)
    print(f"\nReal windows total: {len(X_real)}")

    # ── 3. Window the synthetic CSV ──────────────────────────────────────
    print(f"\nLoading synthetic from {args.synthetic_csv} …")
    X_syn, y_syn = window_synthetic(pathlib.Path(args.synthetic_csv))
    print(f"Synthetic windows total: {len(X_syn)}")

    # ── 4. Hold out 20 % of REAL as validation (the only thing that matters) ─
    X_real_tr, X_real_val, y_real_tr, y_real_val = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real)
    print(f"Real train/val split: {len(X_real_tr)} / {len(X_real_val)}")

    # ── 5. Combine real-train + full synthetic, with sample weights ──────
    X_all = np.concatenate([X_real_tr, X_syn], axis=0)
    y_all = np.concatenate([y_real_tr, y_syn], axis=0)
    w_all = np.concatenate([
        np.full(len(X_real_tr), args.real_weight, dtype=np.float32),
        np.ones(len(X_syn), dtype=np.float32),
    ])

    # ── 6. Refit scaler on combined data ─────────────────────────────────
    scaler = StandardScaler()
    X_all_s     = scaler.fit_transform(X_all).astype(np.float32)
    X_real_val_s = scaler.transform(X_real_val).astype(np.float32)

    # ── 7. Load base model, fine-tune ────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    model = MLP().to(DEVICE)
    model.load_state_dict(torch.load(args.base_model, map_location=DEVICE,
                                     weights_only=True))

    print(f"\nFine-tuning {args.epochs} epochs, LR={args.lr}, "
          f"real_weight={args.real_weight}x")
    model, best_val_acc = train_weighted_mlp(
        model,
        X_all_s, y_all, w_all,
        X_real_val_s, y_real_val,
        epochs=args.epochs, lr=args.lr,
    )

    # ── 8. Final report on real validation set ───────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_real_val_s, dtype=torch.float32)
                      .to(DEVICE)).argmax(1).cpu().numpy()
    final_acc = accuracy_score(y_real_val, preds)
    final_f1  = f1_score(y_real_val, preds, average="macro",
                         labels=[0,1,2,3], zero_division=0)
    print("\n" + "=" * 60)
    print(f"REAL-DATA VALIDATION   acc={final_acc:.4f}   macro-F1={final_f1:.4f}")
    print("=" * 60)
    print(classification_report(
        y_real_val, preds, labels=[0,1,2,3],
        target_names=CLASS_NAMES, zero_division=0))

    # ── 9. Save model + scaler with descriptive filename ─────────────────
    acc_per = int(round(final_acc * 100))
    base = f"model_synthetic_finetuned_real4cls_{ts}_acc{acc_per}per"
    model_path  = OUT_DIR / f"{base}.pt"
    scaler_path = OUT_DIR / f"{base.replace('model_', 'scaler_')}.pkl"
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved model  → {model_path}")
    print(f"Saved scaler → {scaler_path}")

    if args.update_canonical:
        torch.save(model.state_dict(), DEFAULT_BASE_MODEL)
        with open(DEFAULT_BASE_SCALER, "wb") as f:
            pickle.dump(scaler, f)
        print("Updated canonical model_synthetic.pt + scaler_synthetic.pkl")
    else:
        print("(Canonical model not overwritten — pass --update_canonical to do that.)")


if __name__ == "__main__":
    main()
