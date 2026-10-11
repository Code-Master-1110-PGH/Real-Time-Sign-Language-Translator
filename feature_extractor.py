"""
Feature Extraction and Normalization Module.

Extracts 3D hand landmarks via MediaPipe and applies geometric normalization
to guarantee scale- and translation-invariance across variable camera depths
and spatial positions in the video frame.
"""

import os
import urllib.request
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import PipelineConfig


class LandmarkNormalizer:
    """
    Applies mathematical transformations to raw 3D landmarks to produce
    translation- and scale-invariant geometric representation.
    """

    @staticmethod
    def normalize_hand_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Normalizes a single hand's 21 landmarks (shape: [21, 3]).

        1. Translation Invariance:
           The wrist (landmark index 0) is set as the origin (0, 0, 0).
           All remaining landmarks are shifted relative to the wrist:
           P_rel = P_i - P_wrist

        2. Scale Invariance:
           The Euclidean distance between the wrist (index 0) and the
           Middle Finger MCP joint (index 9) represents hand scale.
           All shifted coordinates are scaled by this reference distance:
           P_norm = P_rel / (scale + eps)

        Args:
            landmarks_3d: Raw landmark array of shape (21, 3).

        Returns:
            Normalized landmark array of shape (21, 3).
        """
        if landmarks_3d.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {landmarks_3d.shape}")

        # Translation invariance: subtract wrist position
        wrist = landmarks_3d[0:1, :]  # shape: (1, 3)
        centered = landmarks_3d - wrist

        # Scale invariance: compute wrist-to-middle-MCP (index 9) Euclidean distance
        middle_mcp = centered[9]
        scale = np.linalg.norm(middle_mcp)

        # Numerical stability safeguard against near-zero division
        if scale < 1e-5:
            # Fallback to maximum pairwise distance from wrist
            distances = np.linalg.norm(centered, axis=-1)
            scale = np.max(distances)
            if scale < 1e-5:
                scale = 1.0

        normalized = centered / (scale + 1e-6)
        return normalized.astype(np.float32)


class HandFeatureExtractor:
    """
    Encapsulates MediaPipe Hand Landmark detection and feature vector construction.
    Supports both modern MediaPipe Tasks API (v0.10+/v1.x) and legacy mp.solutions.
    """

    TASK_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.normalizer = LandmarkNormalizer()
        self.use_tasks_api = False
        self._detector = None
        self._legacy_hands = None
        self._init_mediapipe()

    def _ensure_task_model(self, model_path: str) -> str:
        """Downloads the official MediaPipe hand landmarker task model if missing."""
        if not os.path.exists(model_path):
            print(f"[*] MediaPipe model asset not found at '{model_path}'. Downloading...")
            urllib.request.urlretrieve(self.TASK_URL, model_path)
            print(f"[+] Download complete: '{model_path}' ({os.path.getsize(model_path)} bytes)")
        return model_path

    def _init_mediapipe(self) -> None:
        """Initializes the underlying MediaPipe Hand Detector."""
        import mediapipe as mp

        try:
            from mediapipe.tasks.python import vision, BaseOptions

            model_path = self._ensure_task_model(self.config.model_asset_path)
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_hands=self.config.num_hands,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            print("[+] HandFeatureExtractor initialized with MediaPipe Tasks API.")
            return
        except Exception as e:
            print(f"[!] Tasks API initialization failed or unavailable ({e}). Checking legacy fallback...")

        # Fallback for legacy mediapipe installations
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_hands = mp.solutions.hands
            self.use_tasks_api = False
            print("[+] HandFeatureExtractor initialized with legacy mp.solutions API.")
        else:
            raise RuntimeError(
                "Failed to initialize MediaPipe Hand Landmarker. "
                "Ensure either mediapipe.tasks or mediapipe.solutions is available."
            )

    def extract_features(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Processes a BGR video frame, detects up to 2 hands, applies normalization,
        and packs into a flattened feature vector of size 126 (2 hands * 21 * 3).

        Args:
            frame_bgr: Raw input frame from OpenCV (H, W, 3) in BGR color format.

        Returns:
            Tuple containing:
                - feature_vector: 1D np.ndarray of shape (126,) float32.
                - raw_landmarks_list: List of raw 2D/3D pixel coordinates for visualization.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_hands_data: List[np.ndarray] = []

        if self.use_tasks_api:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self._detector.detect(mp_image)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    pts = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                        dtype=np.float32,
                    )
                    raw_hands_data.append(pts)
        else:
            results = self._legacy_hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    pts = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    )
                    raw_hands_data.append(pts)

        # Allocate canonical buffer for up to 2 hands (shape: 2, 21, 3)
        normalized_hands = np.zeros(
            (self.config.num_hands, self.config.landmarks_per_hand, self.config.coords_per_landmark),
            dtype=np.float32,
        )

        for i, raw_hand in enumerate(raw_hands_data[: self.config.num_hands]):
            normalized_hands[i] = self.normalizer.normalize_hand_landmarks(raw_hand)

        feature_vector = normalized_hands.flatten()  # 126 elements
        return feature_vector, raw_hands_data

    def draw_landmarks(
        self, frame_bgr: np.ndarray, raw_hands_data: List[np.ndarray]
    ) -> np.ndarray:
        """
        Renders lightweight hand landmarks and skeletal connections on the OpenCV frame.
        """
        h, w, _ = frame_bgr.shape
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
        ]

        for hand in raw_hands_data:
            # Convert normalized (x, y) to image coordinates
            pts_px = [(int(pt[0] * w), int(pt[1] * h)) for pt in hand]

            # Draw bones
            for p1, p2 in connections:
                cv2.line(frame_bgr, pts_px[p1], pts_px[p2], (0, 220, 100), 2)

            # Draw joints
            for idx, pt in enumerate(pts_px):
                color = (0, 0, 255) if idx == 0 else (255, 200, 0)
                radius = 4 if idx == 0 else 3
                cv2.circle(frame_bgr, pt, radius, color, -1)

        return frame_bgr

    def close(self) -> None:
        """Releases underlying MediaPipe graph resources safely."""
        if self._detector is not None:
            det = self._detector
            self._detector = None
            try:
                if hasattr(det, "close"):
                    det.close()
            except Exception:
                pass
        if self._legacy_hands is not None:
            hands = self._legacy_hands
            self._legacy_hands = None
            try:
                if hasattr(hands, "close"):
                    hands.close()
            except Exception:
                pass
