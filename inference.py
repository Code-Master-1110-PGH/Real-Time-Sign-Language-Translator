"""
Real-Time Sign Language Translator: Real-Time OpenCV Inference Pipeline.

Integrates MediaPipe landmark extraction, sliding temporal buffer,
PyTorch sequence neural inference, and temporal debouncing into a high-FPS,
low-latency video processing loop with an informative HUD overlay.
"""

import argparse
import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from buffer import TemporalFeatureBuffer
from config import PipelineConfig
from feature_extractor import HandFeatureExtractor
from models import build_model
from smoother import PredictionSmoother


class HUDVisualizer:
    """
    Renders a high-quality, professional Heads-Up Display (HUD) overlay
    including live FPS, translation subtitles, confidence bars, and status.
    """

    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        current_label: str,
        confidence: float,
        sentence: str,
        fps: float,
        smoothed_probs: np.ndarray,
        labels: list,
        buffer_ratio: float,
    ) -> np.ndarray:
        """
        Draws diagnostic telemetry and translation results on the frame.
        """
        h, w, _ = frame.shape

        # 1. Top Diagnostic Header Bar (Translucent dark overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)

        # 2. Bottom Subtitle Ribbon (for translated sentence)
        cv2.rectangle(overlay, (0, h - 70), (w, h), (15, 15, 15), -1)

        # 3. Mini Probability Bar Chart on Right side
        panel_w = 210
        cv2.rectangle(overlay, (w - panel_w - 10, 75), (w - 10, 75 + len(labels) * 22 + 10), (25, 25, 25), -1)

        # Alpha blend HUD panels
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Header: FPS and Active Gesture Badge
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

        # Buffer status indicator
        buf_w = 80
        buf_fill = int(buf_w * buffer_ratio)
        cv2.rectangle(frame, (18, 44), (18 + buf_w, 54), (70, 70, 70), 1)
        cv2.rectangle(frame, (18, 44), (18 + buf_fill, 54), (0, 200, 255), -1)
        cv2.putText(frame, "BUFFER", (105, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Main active sign label
        badge_color = (0, 220, 80) if current_label != "IDLE" else (160, 160, 160)
        label_text = f"SIGN: {current_label} ({confidence * 100:.1f}%)"
        cv2.putText(frame, label_text, (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_color, 2)

        # Mini Probability Panel
        for idx, (lbl, prob) in enumerate(zip(labels, smoothed_probs)):
            y_offset = 95 + idx * 22
            bar_len = int(prob * 90)
            color = (0, 220, 80) if idx == np.argmax(smoothed_probs) else (140, 140, 140)

            # Label text
            cv2.putText(frame, lbl[:9], (w - panel_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            # Background track
            cv2.rectangle(frame, (w - 110, y_offset - 8), (w - 20, y_offset - 1), (50, 50, 50), -1)
            # Active fill
            cv2.rectangle(frame, (w - 110, y_offset - 8), (w - 110 + bar_len, y_offset - 1), color, -1)

        # Subtitle Ribbon: Translated Cumulative Sentence
        display_sentence = sentence if sentence else "[Perform dynamic signs to build sentence...]"
        sub_color = (255, 255, 255) if sentence else (130, 130, 130)
        cv2.putText(
            frame,
            f"TRANSLATION: {display_sentence}",
            (20, h - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            sub_color,
            2,
        )

        return frame


class RealTimeTranslator:
    """
    Main Real-Time Sign Language Translator Pipeline.
    Orchestrates video capture, landmark extraction, temporal buffering,
    PyTorch model inference, and smoothed HUD presentation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Feature Extractor
        print("[*] Initializing MediaPipe Hand Feature Extractor...")
        self.extractor = HandFeatureExtractor(self.config)

        # 2. Temporal Buffer
        self.buffer = TemporalFeatureBuffer(self.config)

        # 3. Post-processing Smoother
        self.smoother = PredictionSmoother(self.config)

        # 4. Neural Sequence Model
        print(f"[*] Initializing PyTorch Sequence Model ({self.config.model_type.upper()})...")
        self.model = build_model(self.config).to(self.device)
        self._load_model_weights()
        self.model.eval()

        self.visualizer = HUDVisualizer()

    def _load_model_weights(self) -> None:
        """Loads weights from checkpoint file if present."""
        weights_path = self.config.model_weights_path
        if os.path.exists(weights_path):
            print(f"[+] Loading model weights from: '{weights_path}'")
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
            except Exception:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)

            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print("[+] Weights loaded successfully.")
        else:
            print(f"[!] Warning: Weights checkpoint '{weights_path}' not found.")
            print("[!] Running with initialized weights. Run `python train.py` to train model.")

    def run(self, source=0, max_frames: Optional[int] = None) -> None:
        """
        Executes real-time inference loop.

        Args:
            source: Webcam device index (int) or path to video file (str).
            max_frames: Optional maximum frame limit for benchmark runs.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        print("[+] Video stream initialized.")
        print("[*] Controls: 'q' to quit | 'c' to clear sentence | 'r' to reset smoother")

        prev_time = time.perf_counter()
        fps = 0.0
        frame_idx = 0

        # State cache for stride intervals
        current_label = "IDLE"
        current_confidence = 0.0
        smoothed_probs = np.ones(self.config.num_classes, dtype=np.float32) / self.config.num_classes

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[!] Video stream ended.")
                    break

                frame_idx += 1
                curr_time = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
                prev_time = curr_time

                # Flip horizontally for natural selfie-view webcam interaction
                frame = cv2.flip(frame, 1)

                # 1. Extract 3D normalized landmarks
                features, raw_landmarks = self.extractor.extract_features(frame)

                # 2. Append to temporal sliding window buffer
                self.buffer.append(features)

                # 3. Model Inference (Decoupled execution via inference_stride)
                if self.buffer.ready_for_inference(self.config.inference_stride):
                    # Zero-overhead tensor conversion
                    tensor_seq = self.buffer.get_sequence_tensor(self.device)  # (1, T, D)

                    with torch.no_grad():
                        logits = self.model(tensor_seq)  # (1, C)
                        raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    # 4. Debounce and Exponential Moving Average smoothing
                    current_label, current_confidence, smoothed_probs = self.smoother.update(raw_probs)

                # 5. Render skeletal bones and joints
                self.extractor.draw_landmarks(frame, raw_landmarks)

                # 6. Render HUD telemetry and subtitles
                buffer_ratio = self.buffer.current_length / self.config.sequence_length
                frame = self.visualizer.draw_hud(
                    frame=frame,
                    current_label=current_label,
                    confidence=current_confidence,
                    sentence=self.smoother.sentence,
                    fps=fps,
                    smoothed_probs=smoothed_probs,
                    labels=self.config.labels,
                    buffer_ratio=buffer_ratio,
                )

                cv2.imshow("Real-Time Sign Language Translator", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.smoother.clear_sentence()
                elif key == ord("r"):
                    self.smoother.reset()
                    self.buffer.clear()

                if max_frames is not None and frame_idx >= max_frames:
                    print(f"[*] Benchmark limit of {max_frames} frames reached.")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            print("[+] Resources released successfully.")


def run_synthetic_benchmark(translator: RealTimeTranslator, num_frames: int = 90) -> None:
    """
    Simulates real-time video frames to benchmark end-to-end throughput and latency
    in headless or camera-free environments.
    """
    print(f"\n[*] Starting Synthetic Benchmark across {num_frames} frames...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    latencies = []
    fps_values = []

    for i in range(num_frames):
        t0 = time.perf_counter()

        # Landmark extraction
        features, _ = translator.extractor.extract_features(dummy_frame)
        translator.buffer.append(features)

        # Inference with stride
        if translator.buffer.ready_for_inference(translator.config.inference_stride):
            tensor_seq = translator.buffer.get_sequence_tensor(translator.device)
            with torch.no_grad():
                logits = translator.model(tensor_seq)
                raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            translator.smoother.update(raw_probs)

        dt = time.perf_counter() - t0
        latencies.append(dt * 1000.0)
        fps_values.append(1.0 / max(dt, 1e-5))

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    mean_fps = np.mean(fps_values)

    print("\n" + "=" * 50)
    print("      REAL-TIME SYSTEM LATENCY BENCHMARK        ")
    print("=" * 50)
    print(f"Device:               {translator.device}")
    print(f"Model Architecture:   {translator.config.model_type.upper()}")
    print(f"Average Latency:      {mean_lat:.2f} ms / frame")
    print(f"95th Percentile:      {p95_lat:.2f} ms / frame")
    print(f"Effective Throughput: {mean_fps:.1f} FPS")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Translator")
    parser.add_argument("--source", default=0, help="Webcam device ID (0) or path to video file")
    parser.add_argument("--model-type", default="gru", choices=["gru", "lstm", "tcn"])
    parser.add_argument("--benchmark", action="store_true", help="Run synthetic latency benchmark without camera")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    cfg = PipelineConfig(model_type=args.model_type)

    # Cast source to integer if numeric
    source = int(args.source) if str(args.source).isdigit() else args.source

    engine = RealTimeTranslator(cfg)

    if args.benchmark:
        run_synthetic_benchmark(engine, num_frames=60)
    else:
        try:
            engine.run(source=source, max_frames=args.max_frames)
        except Exception as err:
            print(f"[!] Unable to launch video window ({err}). Falling back to benchmark mode...")
            run_synthetic_benchmark(engine, num_frames=60)
