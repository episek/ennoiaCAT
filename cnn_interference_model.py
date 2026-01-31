"""
1D CNN Interference Detection Model for O-RAN EVM-per-PRB data.

Provides a lightweight convolutional neural network that produces per-PRB
interference probabilities from EVM dB profiles.  Designed to run on top of
the existing threshold-based state-machine detector.

Usage:
    # Training from accumulated labeled data
    python cnn_interference_model.py --train

    # Programmatic inference
    from cnn_interference_model import CNNInterferenceDetector
    det = CNNInterferenceDetector("cnn_interference_model.pth")
    if det.is_available():
        proba = det.predict_proba(evm_db_layer)  # (num_prbs,)
"""

import os
import glob
import argparse

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class InterferenceCNN1D(nn.Module):
        """Tiny fully-convolutional 1D CNN for per-PRB interference detection.

        Input : (batch, 1, num_prbs)   — normalised EVM dB profile
        Output: (batch, num_prbs)      — per-PRB interference probability [0, 1]
        ~3 200 parameters, <1 ms inference on CPU.
        """

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (batch, 1, num_prbs)
            return self.net(x).squeeze(1)  # (batch, num_prbs)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_evm(evm_db_layer: np.ndarray) -> np.ndarray:
    """Clip, normalise, and reshape a single-layer EVM dB vector for the CNN.

    Args:
        evm_db_layer: 1-D array of EVM values in dB (num_prbs,).

    Returns:
        numpy array of shape (1, 1, num_prbs) ready for ``torch.from_numpy``.
    """
    x = np.array(evm_db_layer, dtype=np.float32)
    x = np.clip(x, -40.0, 10.0)
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + 1e-6)
    return x.reshape(1, 1, -1)


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class CNNInterferenceDetector:
    """High-level wrapper around :class:`InterferenceCNN1D`.

    Handles model loading, graceful fallback when the weights file is missing
    or PyTorch is unavailable, and conversion between numpy and torch tensors.
    """

    def __init__(self, model_path: str = "cnn_interference_model.pth"):
        self._model = None
        self._model_path = model_path
        self._available = False

        if not TORCH_AVAILABLE:
            print("CNN detector: PyTorch not installed — CNN disabled.")
            return

        if not os.path.isfile(model_path):
            print(f"CNN detector: weights file '{model_path}' not found — CNN disabled.")
            return

        try:
            self._model = InterferenceCNN1D()
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()
            self._available = True
            print(f"CNN detector: loaded model from '{model_path}'.")
        except Exception as exc:
            print(f"CNN detector: failed to load model — {exc}")
            self._model = None

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    def predict_proba(self, evm_db_layer: np.ndarray) -> np.ndarray:
        """Return per-PRB interference probability in [0, 1].

        Args:
            evm_db_layer: 1-D numpy array (num_prbs,) of EVM in dB.

        Returns:
            1-D numpy array (num_prbs,) of probabilities.
        """
        if not self._available:
            return np.zeros(len(evm_db_layer), dtype=np.float32)

        inp = preprocess_evm(evm_db_layer)
        with torch.no_grad():
            tensor_in = torch.from_numpy(inp)
            tensor_out = self._model(tensor_in)
        return tensor_out.squeeze(0).numpy()

    # ------------------------------------------------------------------
    def detect(self, evm_db_layer: np.ndarray, threshold: float = 0.5,
               min_region_size: int = 3):
        """Convert per-PRB probabilities into contiguous interference regions.

        Returns:
            list of (start, end) tuples — same format as
            ``detect_snr_drop_regions``.
        """
        proba = self.predict_proba(evm_db_layer)
        return regions_from_mask(proba >= threshold, min_region_size)


# ---------------------------------------------------------------------------
# Utility: binary mask -> region list
# ---------------------------------------------------------------------------

def regions_from_mask(mask: np.ndarray, min_region_size: int = 3):
    """Convert a boolean mask into a list of (start, end) contiguous regions.

    Short regions (length < *min_region_size*) are discarded.
    """
    regions = []
    in_region = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_region:
            in_region = True
            start = i
        elif not val and in_region:
            in_region = False
            if (i - start) >= min_region_size:
                regions.append((start, i))
    if in_region:
        length = len(mask) - start
        if length >= min_region_size:
            regions.append((start, len(mask)))
    return regions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cnn_model(data_dir: str = "cnn_training_data",
                    model_path: str = "cnn_interference_model.pth",
                    epochs: int = 50, lr: float = 1e-3,
                    val_split: float = 0.2, batch_size: int = 16):
    """Train the 1D CNN from labeled CSV files in *data_dir*.

    Each CSV is expected to have columns:
        PRB, Layer0_EVM_dB, Layer0_Label, Layer1_EVM_dB, Layer1_Label, ...

    Labels are binary (0 = clean, 1 = interference).
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for training. pip install torch")
        return False

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV training files found in '{data_dir}/'.")
        return False

    print(f"Found {len(csv_files)} training CSV file(s) in '{data_dir}/'.")

    evm_samples = []
    label_samples = []

    for csv_path in csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  Skipping {csv_path}: {exc}")
            continue

        # Detect layer columns
        for layer_idx in range(4):
            evm_col = f"Layer{layer_idx}_EVM_dB"
            lbl_col = f"Layer{layer_idx}_Label"
            if evm_col in df.columns and lbl_col in df.columns:
                evm_vals = df[evm_col].values.astype(np.float32)
                lbl_vals = df[lbl_col].values.astype(np.float32)
                if len(evm_vals) > 0:
                    evm_samples.append(evm_vals)
                    label_samples.append(lbl_vals)

    if not evm_samples:
        print("No usable training samples found.")
        return False

    # Pad / truncate to common length (273 for standard NR)
    max_len = max(len(s) for s in evm_samples)
    X = np.zeros((len(evm_samples), max_len), dtype=np.float32)
    Y = np.zeros((len(evm_samples), max_len), dtype=np.float32)
    for i, (evm, lbl) in enumerate(zip(evm_samples, label_samples)):
        n = len(evm)
        X[i, :n] = evm
        Y[i, :n] = lbl

    # Preprocess X (clip + per-sample normalise)
    X = np.clip(X, -40.0, 10.0)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True) + 1e-6
    X = (X - means) / stds

    # Reshape for Conv1d: (N, 1, L)
    X = X[:, np.newaxis, :]

    # Train / val split
    n_total = len(X)
    n_val = max(1, int(n_total * val_split))
    indices = np.random.permutation(n_total)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = InterferenceCNN1D()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Training: {len(train_idx)} samples train, {len(val_idx)} val, "
          f"{epochs} max epochs, lr={lr}")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_idx)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_idx)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    print(f"Best val_loss={best_val_loss:.4f}  — model saved to '{model_path}'.")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Interference Detection Model")
    parser.add_argument("--train", action="store_true",
                        help="Train the model from CSV files in cnn_training_data/")
    parser.add_argument("--data-dir", default="cnn_training_data",
                        help="Directory containing labeled CSV files")
    parser.add_argument("--model-path", default="cnn_interference_model.pth",
                        help="Path to save/load model weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        train_cnn_model(data_dir=args.data_dir, model_path=args.model_path,
                        epochs=args.epochs, lr=args.lr)
    else:
        parser.print_help()
