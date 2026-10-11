"""
Temporal Feature Buffer Module.

Maintains a sliding temporal window of hand landmark feature vectors
for dynamic sequential gesture classification.
"""

from collections import deque
from typing import Optional
import numpy as np
import torch

from config import PipelineConfig


class TemporalFeatureBuffer:
    """
    Sliding window ring-buffer storing the most recent T frames of landmark vectors.
    Outputs zero-copy / contiguous PyTorch tensors for neural model inference.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.sequence_length = self.config.sequence_length
        self.feature_dim = self.config.feature_dim
        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._frame_counter: int = 0

    def append(self, feature_vector: np.ndarray) -> None:
        """
        Appends a 1D feature vector to the temporal window.

        Args:
            feature_vector: 1D array of length `feature_dim`.
        """
        if feature_vector.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected feature vector shape ({self.feature_dim},), got {feature_vector.shape}"
            )
        self._buffer.append(feature_vector.astype(np.float32))
        self._frame_counter += 1

    def is_full(self) -> bool:
        """Returns True if the buffer contains a full sequence of frames."""
        return len(self._buffer) == self.sequence_length

    def ready_for_inference(self, stride: Optional[int] = None) -> bool:
        """
        Determines whether the model should execute on the current frame.
        Guarantees that inference executes only after buffer is full,
        and thereafter every `stride` frames to preserve CPU budget.
        """
        step = stride if stride is not None else self.config.inference_stride
        return self.is_full() and (self._frame_counter % step == 0)

    def get_sequence_numpy(self) -> np.ndarray:
        """
        Returns the temporal sequence as a NumPy array of shape (sequence_length, feature_dim).
        If the buffer is not yet full, zero-pads the leading frames.
        """
        if len(self._buffer) == 0:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        data = np.array(self._buffer, dtype=np.float32)
        if len(self._buffer) < self.sequence_length:
            pad_len = self.sequence_length - len(self._buffer)
            pad = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            data = np.vstack([pad, data])

        return data

    def get_sequence_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Returns a batched PyTorch tensor of shape (1, sequence_length, feature_dim).

        Args:
            device: torch.device target (CPU or CUDA).

        Returns:
            torch.Tensor ready for input into sequential neural models.
        """
        np_seq = self.get_sequence_numpy()
        tensor = torch.from_numpy(np_seq).unsqueeze(0).to(device)  # (1, T, D)
        return tensor

    def clear(self) -> None:
        """Flushes the buffer and resets frame counter."""
        self._buffer.clear()
        self._frame_counter = 0

    @property
    def current_length(self) -> int:
        """Returns current count of buffered frames."""
        return len(self._buffer)

