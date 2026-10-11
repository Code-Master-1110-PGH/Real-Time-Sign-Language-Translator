"""
System configuration for Real-Time Sign Language Translator.
Centralizes hyperparameters, geometry dimensions, model architectures,
and runtime execution thresholds.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration parameters for the end-to-end translation pipeline."""

    # Camera & Video Stream
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    # Landmark Extraction
    num_hands: int = 2
    landmarks_per_hand: int = 21
    coords_per_landmark: int = 3  # (x, y, z)
    # 2 hands * 21 landmarks * 3 coordinates = 126 features per frame
    feature_dim: int = 2 * 21 * 3
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    model_asset_path: str = "hand_landmarker.task"

    # Sequence & Temporal Windowing
    sequence_length: int = 30       # 30 frames ~ 1.0 second window at 30 FPS
    inference_stride: int = 2       # Run neural inference every 2 frames to conserve CPU

    # Neural Model Architecture
    model_type: str = "gru"         # "gru", "lstm", or "tcn"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Classification Vocabulary
    labels: List[str] = field(
        default_factory=lambda: [
            "IDLE",          # No sign / resting position
            "HELLO",         # Dynamic waving gesture
            "THANK_YOU",     # Hand from chin moving forward
            "YES",           # Fist nodding vertically
            "NO",            # Index and middle finger snapping shut on thumb
            "HELP",          # Thumbs up on flat palm moving up
            "I_LOVE_YOU",    # Thumb, index, pinky extended
            "PLEASE",        # Flat hand rubbing chest in circle
        ]
    )

    # Post-Processing & Smoothing
    confidence_threshold: float = 0.70  # Minimum softmax probability to trigger state update
    consecutive_trigger_frames: int = 3 # Debounce count before accepting a new classification
    smoothing_alpha: float = 0.65       # EMA weight for softmax probability smoothing
    model_weights_path: str = "gesture_model.pt"

    @property
    def num_classes(self) -> int:
        """Returns the number of gesture target classes."""
        return len(self.labels)

