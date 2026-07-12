"""
train_baselines.py
──────────────────
Benchmark the MLP stroke classifier against three simpler baselines:
    SVM (RBF kernel), Random Forest, k-NN

Runs on BOTH datasets using IDENTICAL splits to comparison_notebook.ipynb:
    random_state=42, test_size=0.2, stratify=y

Outputs (all saved to mujoco_sim/output/):
    baseline_comparison.csv     — full per-classifier metrics table
    baseline_comparison.pdf     — grouped bar chart (thesis figure)
    baseline_comparison.png     — preview
    synthetic_per_class.csv     — per-class metrics for synthetic model only
    ttswing_per_class.csv       — per-class metrics for TTSWING model only

Run:
    python train_baselines.py
"""

import argparse
import csv
import pathlib
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import deque
from scipy.stats import kurtosis, skew
from scipy.signal import periodogram

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE          = pathlib.Path(__file__).resolve().parent
ROOT          = HERE.parent.parent
SRC_DIR       = ROOT / "src"
OUT_DIR       = HERE.parent / "output"
TTSWING_CSV   = ROOT / "data" / "raw" / "TTSWING.csv"
SYNTHETIC_CSV = OUT_DIR / "tt_synthetic_100reps.csv"   # default; overridden via CLI
sys.path.insert(0, str(SRC_DIR))

# Use the shared Butterworth filter so train and inference treat features identically
from signal_filter import lowpass, DEFAULT_FS

# These are set from CLI in main() — keep at module scope so extract_features()
# can pick them up without threading args through every call.
APPLY_FILTER = True
FILTER_FS    = DEFAULT_FS

CLASS_NAMES = ["No Stroke", "FH Topspin", "BH Drive", "FH Smash"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "ax_mean","ay_mean","az_mean","gx_mean","gy_mean","gz_mean",
    "ax_var", "ay_var", "az_var", "gx_var", "gy_var", "gz_var",
    "ax_rms", "ay_rms", "az_rms", "gx_rms", "gy_rms", "gz_rms",
    "a_max",  "a_mean", "a_min",  "g_max",  "g_mean", "g_min",
    "a_fft",  "g_fft",  "a_psdx", "g_psdx",
    "a_kurt", "g_kurt", "a_skewn","g_skewn",
    "a_entropy","g_entropy",
]


# ── MLP (same architecture as notebook) ───────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim=34, hidden_dims=[256,128,64],
                 num_classes=4, dropout=0.3):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train_mlp(X_tr, y_tr, X_te, y_te,
              epochs=60, batch_size=256, lr=0.001, patience=10):
    model = MLP().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    Xv    = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    yv    = torch.tensor(y_te, dtype=torch.long).to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    best_acc, best_state, wait = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            acc = accuracy_score(y_te, model(Xv).argmax(1).cpu().numpy())
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)
                     ).argmax(1).cpu().numpy()


# ── Feature extraction (matches notebook exactly) ─────────────────────────────

def _spectral_entropy(sig):
    psd = np.abs(np.fft.rfft(sig)) ** 2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))


def extract_features(rows, imu_id=7):
    p  = f"imu_{imu_id}_"
    ax = np.array([float(r[p+"accel_x"]) for r in rows])
    ay = np.array([float(r[p+"accel_y"]) for r in rows])
    az = np.array([float(r[p+"accel_z"]) for r in rows])
    gx = np.array([float(r[p+"gyro_x"])  for r in rows])
    gy = np.array([float(r[p+"gyro_y"])  for r in rows])
    gz = np.array([float(r[p+"gyro_z"])  for r in rows])

    if APPLY_FILTER:
        ax = lowpass(ax, fs=FILTER_FS); ay = lowpass(ay, fs=FILTER_FS); az = lowpass(az, fs=FILTER_FS)
        gx = lowpass(gx, fs=FILTER_FS); gy = lowpass(gy, fs=FILTER_FS); gz = lowpass(gz, fs=FILTER_FS)

    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    _, a_psd = periodogram(a_mag)
    _, g_psd = periodogram(g_mag)
    feats = np.array([
        np.mean(ax), np.mean(ay), np.mean(az),
        np.mean(gx), np.mean(gy), np.mean(gz),
        np.var(ax),  np.var(ay),  np.var(az),
        np.var(gx),  np.var(gy),  np.var(gz),
        np.sqrt(np.mean(ax**2)), np.sqrt(np.mean(ay**2)), np.sqrt(np.mean(az**2)),
        np.sqrt(np.mean(gx**2)), np.sqrt(np.mean(gy**2)), np.sqrt(np.mean(gz**2)),
        np.max(a_mag), np.mean(a_mag), np.min(a_mag),
        np.max(g_mag), np.mean(g_mag), np.min(g_mag),
        np.sum(np.abs(np.fft.rfft(a_mag))**2),
        np.sum(np.abs(np.fft.rfft(g_mag))**2),
        np.max(a_psd), np.max(g_psd),
        float(kurtosis(a_mag)) if np.std(a_mag) > 1e-8 else 0.0,
        float(kurtosis(g_mag)) if np.std(g_mag) > 1e-8 else 0.0,
        float(skew(a_mag))     if np.std(a_mag) > 1e-8 else 0.0,
        float(skew(g_mag))     if np.std(g_mag) > 1e-8 else 0.0,
        _spectral_entropy(a_mag), _spectral_entropy(g_mag),
    ], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ── Load data ─────────────────────────────────────────────────────────────────

def load_ttswing():
    df = pd.read_csv(TTSWING_CSV)
    df = df[df["testmode"] == 1].reset_index(drop=True)
    X  = df[FEATURE_COLS].values.astype(np.float32)
    y  = df["teststage"].values.astype(int)
    print(f"TTSWING loaded: {X.shape[0]:,} samples")
    return X, y


def load_synthetic():
    with open(SYNTHETIC_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))
    X_list, y_list = [], []
    buf = deque(maxlen=50)
    for i, row in enumerate(all_rows):
        buf.append(row)
        if len(buf) == 50 and (i + 1) % 10 == 0:
            feats = extract_features(list(buf))
            label = int(np.bincount([int(r["stroke_label"])
                                     for r in buf]).argmax())
            X_list.append(feats)
            y_list.append(label)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    print(f"Synthetic loaded: {X.shape[0]:,} samples")
    return X, y


# ── Evaluate any classifier ───────────────────────────────────────────────────

def evaluate(name, preds, y_te):
    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average="macro",
                   labels=[0,1,2,3], zero_division=0)
    rep = classification_report(y_te, preds,
                                 labels=[0,1,2,3],
                                 target_names=CLASS_NAMES,
                                 output_dict=True,
                                 zero_division=0)
    print(f"  {name:<20} acc={acc:.4f}  macro-F1={f1:.4f}")
    return acc, f1, rep


# ── Run all baselines on one dataset ─────────────────────────────────────────

def run_baselines(X, y, dataset_name):
    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*55}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    results = {}

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("  Training MLP…")
    mlp    = train_mlp(X_tr_s, y_tr, X_te_s, y_te)
    preds  = predict_mlp(mlp, X_te_s)
    results["MLP"] = evaluate("MLP", preds, y_te)

    # Save MLP + scaler — returned so main() can handle timestamped saving
    if dataset_name == "MuJoCo Synthetic":
        results["_mlp_model"]  = mlp
        results["_scaler"]     = scaler

    # ── SVM (RBF) ─────────────────────────────────────────────────────────────
    print("  Training SVM (RBF)…")
    svm   = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm.fit(X_tr_s, y_tr)
    results["SVM (RBF)"] = evaluate("SVM (RBF)", svm.predict(X_te_s), y_te)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("  Training Random Forest…")
    rf    = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    results["Random Forest"] = evaluate("Random Forest", rf.predict(X_te_s), y_te)

    # ── k-NN ──────────────────────────────────────────────────────────────────
    print("  Training k-NN (k=5)…")
    knn   = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_tr_s, y_tr)
    results["k-NN (k=5)"] = evaluate("k-NN (k=5)", knn.predict(X_te_s), y_te)

    return results


# ── Build summary DataFrame ───────────────────────────────────────────────────

def build_csv(tt_results, syn_results, ts):
    rows = []
    for clf in ["MLP", "SVM (RBF)", "Random Forest", "k-NN (k=5)"]:
        tt_acc,  tt_f1,  _  = tt_results[clf]
        syn_acc, syn_f1, _  = syn_results[clf]
        rows.append({
            "Classifier" : clf,
            "TT_Accuracy": round(tt_acc,  4),
            "TT_MacroF1" : round(tt_f1,   4),
            "Syn_Accuracy": round(syn_acc, 4),
            "Syn_MacroF1" : round(syn_f1,  4),
        })
    df = pd.DataFrame(rows)
    # Timestamped copy (never overwritten)
    ts_path = OUT_DIR / f"baseline_comparison_{ts}.csv"
    df.to_csv(ts_path, index=False)
    # Canonical copy for quick access (latest run)
    df.to_csv(OUT_DIR / "baseline_comparison.csv", index=False)
    print(f"\nSaved: {ts_path}")
    return df


def build_per_class_csvs(tt_results, syn_results, ts):
    for label, results in [("ttswing", tt_results), ("synthetic", syn_results)]:
        _, _, rep = results["MLP"]
        rows = []
        for cls in CLASS_NAMES:
            rows.append({
                "Class"    : cls,
                "Precision": round(rep[cls]["precision"], 4),
                "Recall"   : round(rep[cls]["recall"],    4),
                "F1"       : round(rep[cls]["f1-score"],  4),
                "Support"  : int(rep[cls]["support"]),
            })
        df = pd.DataFrame(rows)
        ts_path = OUT_DIR / f"{label}_per_class_{ts}.csv"
        df.to_csv(ts_path, index=False)
        df.to_csv(OUT_DIR / f"{label}_per_class.csv", index=False)
        print(f"Saved: {ts_path}")


def save_models(syn_results, ts, tag: str = "", update_canonical: bool = True):
    """
    Saves the trained MLP + scaler with a descriptive, timestamped filename
    that includes the accuracy in percent, e.g.
        model_synthetic_100hz_aug_filt_20260528_191522_acc96per.pt
        scaler_synthetic_100hz_aug_filt_20260528_191522_acc96per.pkl
    """
    mlp     = syn_results["_mlp_model"]
    scaler  = syn_results["_scaler"]
    acc, _, _ = syn_results["MLP"]
    acc_per = int(round(acc * 100))

    parts = ["model_synthetic"]
    if tag:
        parts.append(tag)
    parts.append(ts)
    parts.append(f"acc{acc_per}per")
    base = "_".join(parts)

    ts_model  = OUT_DIR / f"{base}.pt"
    ts_scaler = OUT_DIR / f"{base.replace('model_synthetic', 'scaler_synthetic')}.pkl"
    torch.save(mlp.state_dict(), ts_model)
    with open(ts_scaler, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved model  → {ts_model}")
    print(f"Saved scaler → {ts_scaler}")

    if update_canonical:
        torch.save(mlp.state_dict(), OUT_DIR / "model_synthetic.pt")
        with open(OUT_DIR / "scaler_synthetic.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("Updated canonical model_synthetic.pt + scaler_synthetic.pkl")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(df, ts):
    classifiers = df["Classifier"].tolist()
    x     = np.arange(len(classifiers))
    width = 0.20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Classifier Baseline Comparison: TTSWING vs MuJoCo Synthetic",
                 fontsize=13, fontweight="bold")

    for ax, (acc_col, f1_col, title) in zip(axes, [
        ("TT_Accuracy",  "TT_MacroF1",  "TTSWING (Real Sensor)"),
        ("Syn_Accuracy", "Syn_MacroF1", "MuJoCo Synthetic"),
    ]):
        accs = df[acc_col].values * 100
        f1s  = df[f1_col].values  * 100

        b1 = ax.bar(x - width/2, accs, width, label="Accuracy (%)",
                    color="#3B82F6", alpha=0.88, edgecolor="white")
        b2 = ax.bar(x + width/2, f1s,  width, label="Macro F1 (%)",
                    color="#10B981", alpha=0.88, edgecolor="white")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.axhline(80, color="red", linestyle="--",
                   linewidth=0.9, alpha=0.5, label="80% line")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=12, fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=8)

    plt.tight_layout()
    # Timestamped
    fig.savefig(OUT_DIR / f"baseline_comparison_{ts}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"baseline_comparison_{ts}.png", dpi=150, bbox_inches="tight")
    # Canonical
    fig.savefig(OUT_DIR / "baseline_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved figures → baseline_comparison_{ts}.pdf/png")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_csv", type=str, default=str(SYNTHETIC_CSV),
                        help="Path to the synthetic MuJoCo CSV to train on")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS,
                        help="Sample rate Hz of the synthetic CSV (for filter)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Disable Butterworth filtering during training")
    parser.add_argument("--tag", type=str, default="",
                        help="Descriptive tag added to the model filename")
    parser.add_argument("--no_canonical", action="store_true",
                        help="Don't overwrite the canonical model_synthetic.pt")
    args = parser.parse_args()

    SYNTHETIC_CSV = pathlib.Path(args.synthetic_csv)
    APPLY_FILTER  = not args.no_filter
    FILTER_FS     = args.fs

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp : {ts}")
    print(f"Synthetic CSV : {SYNTHETIC_CSV}")
    print(f"Filter        : {'Butterworth low-pass @ ' + str(FILTER_FS) + ' Hz' if APPLY_FILTER else 'OFF'}")
    print(f"Tag           : {args.tag or '(none)'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_tt,  y_tt  = load_ttswing()
    X_syn, y_syn = load_synthetic()

    tt_results  = run_baselines(X_tt,  y_tt,  "TTSWING")
    syn_results = run_baselines(X_syn, y_syn, "MuJoCo Synthetic")

    df = build_csv(tt_results, syn_results, ts)
    build_per_class_csvs(tt_results, syn_results, ts)
    save_models(syn_results, ts, tag=args.tag, update_canonical=not args.no_canonical)
    plot_comparison(df, ts)

    print("\n" + "="*55)
    print("FINAL SUMMARY")
    print("="*55)
    print(df.to_string(index=False))
