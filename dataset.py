"""
Dataset Management, Augmentation, and Synthetic Kinematic Gesture Generation.

Provides PyTorch Dataset implementations, temporal data augmentations,
and synthetic kinematic trajectory generators to enable instant model training
and validation out-of-the-box.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from config import PipelineConfig


class GestureDataset(Dataset):
    """
    PyTorch Dataset for gesture sequence arrays.
    Applies spatial jitter, scale perturbations, and temporal warping augmentations.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        """
        Args:
            sequences: Array of shape (N, sequence_length, feature_dim).
            labels: 1D array of class indices (N,).
            augment: Whether to apply stochastic kinematic data augmentations.
        """
        assert len(sequences) == len(labels), "Sequence and label counts must match"
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

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
        """Applies spatial Gaussian noise and global amplitude scaling."""
        # 1. Subtle spatial coordinate jitter (sigma = 0.015)
        jitter = np.random.normal(0.0, 0.015, size=seq.shape).astype(np.float32)
        seq += jitter

        # 2. Scale perturbation (random +/- 10% hand size variation)
        scale_factor = float(np.random.uniform(0.90, 1.10))
        seq *= scale_factor

        return seq


class SyntheticGestureGenerator:
    """
    Generates synthetic kinematic trajectories for sign language classes.
    Enables immediate pipeline verification and out-of-the-box model training.
    """

    @staticmethod
    def generate_dataset(
        config: Optional[PipelineConfig] = None,
        samples_per_class: int = 150,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates distinct dynamic harmonic trajectories for each configured gesture class.

        Returns:
            Tuple of (sequences, labels) where:
                sequences shape: (num_classes * samples_per_class, 30, 126)
                labels shape: (num_classes * samples_per_class,)
        """
        cfg = config or PipelineConfig()
        t = np.linspace(0, 2 * np.pi, cfg.sequence_length)  # (30,)

        all_sequences = []
        all_labels = []

        for class_idx, label in enumerate(cfg.labels):
            for _ in range(samples_per_class):
                seq = np.zeros((cfg.sequence_length, cfg.feature_dim), dtype=np.float32)

                if label == "IDLE":
                    # Static near-zero drift
                    noise = np.random.normal(0, 0.01, size=seq.shape)
                    seq += noise.astype(np.float32)

                elif label == "HELLO":
                    # Dynamic waving motion along X & Y axes for right hand (first 63 dims)
                    freq = np.random.uniform(1.8, 2.4)
                    phase = np.random.uniform(0, 0.5)
                    wave = np.sin(freq * t + phase)[:, None]  # (30, 1)
                    # Modulate finger tips (indices 4, 8, 12, 16, 20 -> 3 coords each)
                    seq[:, :63] += 0.35 * wave * np.random.uniform(0.8, 1.2, size=(1, 63))

                elif label == "THANK_YOU":
                    # Hand moving downward/forward away from chin (Z & Y axes)
                    glide = np.linspace(0.8, -0.4, cfg.sequence_length)[:, None]
                    seq[:, 1:63:3] += 0.4 * glide  # Y drift
                    seq[:, 2:63:3] += 0.3 * np.abs(glide)  # Z forward drift

                elif label == "YES":
                    # Vertical fist nodding rhythm (Y-axis oscillation)
                    freq = np.random.uniform(3.0, 3.5)
                    nod = np.cos(freq * t)[:, None]
                    seq[:, 1:63:3] += 0.45 * nod

                elif label == "NO":
                    # Horizontal head/finger shake oscillation (X-axis rapid pulse)
                    freq = np.random.uniform(4.0, 4.5)
                    shake = np.sin(freq * t)[:, None]
                    seq[:, 0:63:3] += 0.4 * shake

                elif label == "HELP":
                    # Both hands interacting: ascending translation (Y-axis upward ramp)
                    ramp = np.linspace(-0.5, 0.6, cfg.sequence_length)[:, None]
                    seq[:, 1::3] += 0.5 * ramp  # Both hands ascending

                elif label == "I_LOVE_YOU":
                    # Static hand posture with slight deliberate forward gesture
                    thrust = np.linspace(0.0, 0.4, cfg.sequence_length)[:, None]
                    seq[:, 2:63:3] += 0.3 * thrust
                    # Extended thumb, index, pinky distinctive signature
                    seq[:, 12:24] *= 0.1  # Closed middle & ring fingers

                elif label == "PLEASE":
                    # Circular palm motion on chest (orthogonal sin/cos coupling)
                    circ_x = np.sin(2 * t)[:, None]
                    circ_y = np.cos(2 * t)[:, None]
                    seq[:, 0:63:3] += 0.3 * circ_x
                    seq[:, 1:63:3] += 0.3 * circ_y

                # Add sensor noise
                seq += np.random.normal(0, 0.02, size=seq.shape).astype(np.float32)

                all_sequences.append(seq)
                all_labels.append(class_idx)

        X = np.stack(all_sequences, axis=0)
        y = np.array(all_labels, dtype=np.int64)

        # Shuffle
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]

