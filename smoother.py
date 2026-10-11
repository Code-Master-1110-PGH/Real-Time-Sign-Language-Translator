"""
Prediction Post-Processing, Temporal Smoothing, and Debounce Module.

Eliminates flickering and spurious frame-level classifications using
Exponential Moving Average (EMA) probability filtering, confidence gating,
and state machine debounce.
"""

from typing import List, Optional, Tuple
import numpy as np

from config import PipelineConfig


class PredictionSmoother:
    """
    State machine that stabilizes raw neural network logits into
    smooth, flicker-free gesture labels and maintains sentence history.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.num_classes = self.config.num_classes
        self.labels = self.config.labels
        self.alpha = self.config.smoothing_alpha
        self.confidence_threshold = self.config.confidence_threshold
        self.required_consecutive = self.config.consecutive_trigger_frames

        # EMA probability state vector
        self._smoothed_probs: np.ndarray = np.ones(self.num_classes, dtype=np.float32) / self.num_classes

        # Debounce state tracking
        self._candidate_class_idx: Optional[int] = None
        self._candidate_count: int = 0
        self._current_stable_class_idx: int = 0  # 0 corresponds to "IDLE"
        self._current_confidence: float = 0.0

        # Translation Sentence Accumulation
        self._sentence: List[str] = []
        self._last_committed_label: Optional[str] = None

    def update(self, raw_logits_or_probs: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Processes new model output, applies EMA smoothing, checks confidence threshold,
        and executes debouncing logic.

        Args:
            raw_logits_or_probs: 1D array of logits or probabilities of shape (num_classes,).

        Returns:
            Tuple of:
                - stable_label: Currently confirmed label (e.g., "HELLO").
                - current_confidence: Smoothed confidence of the active label.
                - smoothed_probs: Entire smoothed probability distribution.
        """
        # Ensure input is a probability distribution (via softmax if logits)
        arr = np.asarray(raw_logits_or_probs, dtype=np.float32)
        if np.abs(arr.sum() - 1.0) > 1e-3 or (arr < 0).any():
            # Apply numerically stable softmax
            exp_scores = np.exp(arr - np.max(arr))
            probs = exp_scores / np.sum(exp_scores)
        else:
            probs = arr

        # 1. Exponential Moving Average (EMA) smoothing
        self._smoothed_probs = self.alpha * self._smoothed_probs + (1.0 - self.alpha) * probs
        self._smoothed_probs /= np.sum(self._smoothed_probs)  # Re-normalize

        top_class_idx = int(np.argmax(self._smoothed_probs))
        top_prob = float(self._smoothed_probs[top_class_idx])

        # 2. Confidence Thresholding
        if top_prob >= self.confidence_threshold:
            # 3. Consecutive-frame Debouncing
            if top_class_idx == self._candidate_class_idx:
                self._candidate_count += 1
            else:
                self._candidate_class_idx = top_class_idx
                self._candidate_count = 1

            if self._candidate_count >= self.required_consecutive:
                # Confirmed stable state transition
                if self._current_stable_class_idx != top_class_idx:
                    self._current_stable_class_idx = top_class_idx
                    new_label = self.labels[top_class_idx]
                    self._on_stable_transition(new_label)

                self._current_confidence = top_prob
        else:
            # Below confidence threshold: reset candidate counter to prevent unstable flips
            self._candidate_count = 0
            # If current probability for the stable class degrades significantly, decay confidence
            self._current_confidence = float(self._smoothed_probs[self._current_stable_class_idx])

        stable_label = self.labels[self._current_stable_class_idx]
        return stable_label, self._current_confidence, self._smoothed_probs

    def _on_stable_transition(self, new_label: str) -> None:
        """
        Appends committed gestures to a cumulative sentence without repeat spamming.
        """
        if new_label == "IDLE":
            self._last_committed_label = None
            return

        if new_label != self._last_committed_label:
            self._sentence.append(new_label)
            self._last_committed_label = new_label

            # Limit sentence history to last 7 words for clean UI rendering
            if len(self._sentence) > 7:
                self._sentence.pop(0)

    @property
    def sentence(self) -> str:
        """Returns the full recognized phrase as a formatted string."""
        return " ".join(self._sentence) if self._sentence else ""

    def clear_sentence(self) -> None:
        """Resets the sentence history."""
        self._sentence.clear()
        self._last_committed_label = None

    def reset(self) -> None:
        """Fully resets all smoothing and debounce internal states."""
        self._smoothed_probs = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self._candidate_class_idx = None
        self._candidate_count = 0
        self._current_stable_class_idx = 0
        self._current_confidence = 0.0
        self.clear_sentence()

