"""
Dataset Management and Kinematic Sequence Augmentations.

Loads pre-split dynamic sign-language landmark archives (`.npz`) produced by
`extract_landmarks.py` and applies training-only kinematic augmentations.
"""

import os
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


def load_landmark_splits(
    npz_path: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    List[str],
]:
    """
    Loads pre-split train, val, and test landmark arrays plus class names from `.npz`.

    Args:
        npz_path: Path to the `.npz` archive created by `extract_landmarks.py`.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test), class_names).
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Landmark dataset not found at '{npz_path}'. "
            "Run `python extract_landmarks.py --help` to generate a real landmark archive."
        )

    with np.load(npz_path, allow_pickle=False) as data:
        required_keys = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "class_names"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f"Archive '{npz_path}' is missing required keys: {sorted(missing)}")

        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.int64)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)
        class_names = [str(c) for c in data["class_names"].tolist()]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


class GestureDataset(Dataset):
    """
    PyTorch Dataset for dynamic gesture landmark sequences.
    Applies realistic training-only kinematic augmentations while preserving
    exact zeros on missing-hand frames.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        frame_drop_prob: float = 0.05,
        temporal_jitter: bool = True,
    ):
        """
        Args:
            sequences: Array of shape (N, sequence_length, feature_dim).
            labels: 1D array of class indices (N,).
            augment: Whether to apply stochastic kinematic data augmentations.
            frame_drop_prob: Probability of zeroing out an individual frame during training.
            temporal_jitter: Whether to apply mild temporal speed/crop resampling when augment=True.
        """
        assert len(sequences) == len(labels), "Sequence and label counts must match"
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.frame_drop_prob = frame_drop_prob
        self.temporal_jitter = temporal_jitter

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()  # (T, D)

        if self.augment:
            seq = self._apply_augmentations(seq)

        x = torch.from_numpy(seq)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def _apply_augmentations(self, seq: np.ndarray) -> np.ndarray:
        """Applies training-only kinematic augmentations."""
        seq_len = seq.shape[0]

        # 1. Mild temporal speed/crop jitter (sub-window resample back to T frames)
        if self.temporal_jitter and seq_len >= 4 and np.random.random() < 0.5:
            crop_ratio = float(np.random.uniform(0.85, 1.0))
            crop_len = max(2, int(round(seq_len * crop_ratio)))
            if crop_len < seq_len:
                start = int(np.random.randint(0, seq_len - crop_len + 1))
                sub_seq = seq[start : start + crop_len]
                indices = np.round(np.linspace(0, crop_len - 1, seq_len)).astype(int)
                seq = sub_seq[indices].copy()

        # 2. Zero-masked spatial Gaussian jitter (sigma = 0.015)
        # Only perturb non-zero coordinates so missing-hand frames stay strictly zero.
        nonzero_mask = seq != 0.0
        if np.any(nonzero_mask):
            jitter = np.random.normal(0.0, 0.015, size=seq.shape).astype(np.float32)
            seq[nonzero_mask] += jitter[nonzero_mask]

        # 3. Anthropometric scale perturbation (random +/- 10% hand size variation)
        scale_factor = float(np.random.uniform(0.90, 1.10))
        seq *= scale_factor

        # 4. Random temporal frame dropout (simulates transient MediaPipe occlusion)
        if self.frame_drop_prob > 0.0:
            drop_mask = np.random.random(seq_len) < self.frame_drop_prob
            seq[drop_mask] = 0.0

        return seq
