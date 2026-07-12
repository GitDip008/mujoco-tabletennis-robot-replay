# src/inference.py
import torch
import numpy as np
import pickle
import pathlib
import yaml
from model import MLP

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


class StrokePredictor:
    """
    Loads a trained MLP checkpoint + its fitted StandardScaler
    and exposes a clean predict() interface.

    Usage:
        predictor = StrokePredictor.from_subject(subject_id=5)
        result = predictor.predict(feature_vector)
    """

    def __init__(self, model: MLP, scaler, device: torch.device):
        self.model  = model
        self.scaler = scaler
        self.device = device
        self.model.eval()

    @classmethod
    def from_subject(
        cls,
        subject_id: int,
        checkpoint_dir: str = None,
        cfg: dict = None,
    ) -> "StrokePredictor":
        """
        Load the best checkpoint saved for a specific subject (LOSO fold).
        Falls back to a global best checkpoint if subject-specific not found.
        """
        if cfg is None:
            root = pathlib.Path(__file__).resolve().parent.parent
            with open(root / "config.yaml") as f:
                cfg = yaml.safe_load(f)

        if checkpoint_dir is None:
            root = pathlib.Path(__file__).resolve().parent.parent
            checkpoint_dir = str(root / cfg["evaluation"]["checkpoint_dir"])

        ckpt_path   = pathlib.Path(checkpoint_dir) / f"best_subj{subject_id:03d}.pt"
        scaler_path = pathlib.Path(checkpoint_dir) / f"best_subj{subject_id:03d}_scaler.pkl"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for subject {subject_id} at {ckpt_path}\n"
                f"Available checkpoints: {list(pathlib.Path(checkpoint_dir).glob('*.pt'))}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model with same config used in training
        model = MLP(
            input_dim   = cfg["model"]["input_dim"],
            hidden_dims = cfg["model"]["hidden_dims"],
            num_classes = cfg["model"]["num_classes"],
            dropout     = cfg["model"]["dropout"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        return cls(model, scaler, device)

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        scaler_path: str,
        cfg: dict = None,
    ) -> "StrokePredictor":
        """
        Load a model from explicit checkpoint + scaler paths.
        Used for the synthetic MuJoCo model and any non-LOSO checkpoints.
        """
        if cfg is None:
            root = pathlib.Path(__file__).resolve().parent.parent
            with open(root / "config.yaml") as f:
                cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MLP(
            input_dim   = cfg["model"]["input_dim"],
            hidden_dims = cfg["model"]["hidden_dims"],
            num_classes = cfg["model"]["num_classes"],
            dropout     = cfg["model"]["dropout"],
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        return cls(model, scaler, device)

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict stroke class for a single feature vector.

        Args:
            features: np.ndarray of shape (34,) or (1, 34)
                      Raw (unscaled) feature values in FEATURE_COLS order.

        Returns:
            {
                "label_id"    : int,
                "label_name"  : str,
                "confidence"  : float,       # probability of predicted class
                "probabilities": dict,       # full softmax distribution
            }
        """
        features = np.array(features, dtype=np.float32).reshape(1, -1)

        if features.shape[1] != len(FEATURE_COLS):
            raise ValueError(
                f"Expected {len(FEATURE_COLS)} features, got {features.shape[1]}"
            )

        # Scale using the fold-specific scaler
        features_scaled = self.scaler.transform(features)

        with torch.no_grad():
            x      = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(x)                          # (1, 4)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

        label_id = int(np.argmax(probs))

        return {
            "label_id"     : label_id,
            "label_name"   : CLASS_NAMES[label_id],
            "confidence"   : round(float(probs[label_id]), 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
        }

    def predict_batch(self, features_matrix: np.ndarray) -> list[dict]:
        """
        Predict for N feature vectors at once.

        Args:
            features_matrix: np.ndarray of shape (N, 34)

        Returns:
            List of N prediction dicts (same format as predict())
        """
        features_matrix = np.array(features_matrix, dtype=np.float32)
        if features_matrix.ndim == 1:
            features_matrix = features_matrix.reshape(1, -1)

        features_scaled = self.scaler.transform(features_matrix)

        with torch.no_grad():
            x      = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        for p in probs:
            label_id = int(np.argmax(p))
            results.append({
                "label_id"     : label_id,
                "label_name"   : CLASS_NAMES[label_id],
                "confidence"   : round(float(p[label_id]), 4),
                "probabilities": {
                    CLASS_NAMES[i]: round(float(v), 4)
                    for i, v in enumerate(p)
                },
            })
        return results


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    root = pathlib.Path(__file__).resolve().parent.parent

    # Load a real sample from the dataset for subject 5
    df = pd.read_csv(root / "data/raw/TTSWING.csv")

    SUBJECT_ID = 5
    sample_row = df[df["id"] == SUBJECT_ID].iloc[0]
    features   = sample_row[FEATURE_COLS].values.astype(np.float32)
    true_label = int(sample_row["teststage"])

    print(f"Subject       : {SUBJECT_ID}")
    print(f"True label    : {true_label} ({CLASS_NAMES[true_label]})")

    predictor = StrokePredictor.from_subject(subject_id=SUBJECT_ID)
    result    = predictor.predict(features)

    print(f"Predicted     : {result['label_id']} ({result['label_name']})")
    print(f"Confidence    : {result['confidence']:.4f}")
    print(f"All probs     : {result['probabilities']}")
    print(f"Correct       : {result['label_id'] == true_label}")