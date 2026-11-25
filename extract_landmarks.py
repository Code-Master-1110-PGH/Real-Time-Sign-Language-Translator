"""
Real Landmark Extraction CLI for Dynamic Sign Language Datasets.

Extracts 3D hand landmarks using MediaPipe and applies the exact wrist-anchored,
scale-invariant normalization from `feature_extractor.py`. Supports:
  1. WLASL JSON + video directory (--wlasl-json, --video-dir)
  2. Generic folder-structured video dataset (--folder-dir)

Outputs a compressed `.npz` archive containing pre-split train/val/test arrays,
class names, and extraction metadata.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import PipelineConfig
from feature_extractor import HandFeatureExtractor


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _find_video_file(video_dir: Path, video_id: str, gloss: Optional[str] = None) -> Optional[Path]:
    """Resolves a WLASL video_id to an existing video file path."""
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.is_file():
            return candidate
        if gloss is not None:
            nested = video_dir / gloss / f"{video_id}{ext}"
            if nested.is_file():
                return nested
    return None


def _normalize_split_name(split: Optional[str]) -> Optional[str]:
    """Normalizes split names to 'train', 'val', or 'test'."""
    if split is None:
        return None
    s = split.strip().lower()
    if s in ("train", "training"):
        return "train"
    if s in ("val", "valid", "validation", "dev"):
        return "val"
    if s in ("test", "testing"):
        return "test"
    return None


def _extract_clip_sequence(
    video_path: Path,
    extractor: HandFeatureExtractor,
    seq_len: int,
    frame_start: int = 1,
    frame_end: int = -1,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Extracts a fixed-length (seq_len, 126) landmark sequence from a video clip.

    Args:
        video_path: Path to video file.
        extractor: Initialized HandFeatureExtractor instance.
        seq_len: Target number of uniformly sampled frames (T).
        frame_start: 1-indexed start frame (WLASL convention).
        frame_end: 1-indexed end frame, or <= 0 to read until EOF.

    Returns:
        Tuple of (sequence of shape (seq_len, 126) or None if unreadable, valid_ratio in [0, 1]).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = max(0, int(frame_start) - 1) if frame_start and frame_start > 0 else 0

    if total_frames > 0:
        end_idx = int(frame_end) if frame_end and frame_end > 0 else total_frames
        end_idx = min(end_idx, total_frames)
    else:
        end_idx = int(frame_end) if frame_end and frame_end > 0 else -1

    # Read frames in [start_idx, end_idx)
    frames: List[np.ndarray] = []
    if start_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    curr_idx = start_idx
    while cap.isOpened():
        if end_idx > 0 and curr_idx >= end_idx:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        curr_idx += 1

    cap.release()

    if len(frames) == 0:
        return None, 0.0

    # Uniformly sample `seq_len` indices across available clip frames
    sample_indices = np.round(np.linspace(0, len(frames) - 1, seq_len)).astype(int)

    # Cache extracted features per unique frame index to avoid redundant MediaPipe calls
    cache: Dict[int, Tuple[np.ndarray, bool]] = {}
    seq = np.zeros((seq_len, extractor.config.feature_dim), dtype=np.float32)
    valid_count = 0

    for t, f_idx in enumerate(sample_indices):
        idx_int = int(f_idx)
        if idx_int not in cache:
            feat, raw_hands = extractor.extract_features(frames[idx_int])
            is_valid = len(raw_hands) > 0
            cache[idx_int] = (feat, is_valid)
        feat, is_valid = cache[idx_int]
        seq[t] = feat
        if is_valid:
            valid_count += 1

    valid_ratio = valid_count / float(seq_len)
    return seq, valid_ratio


def _assign_group_disjoint_splits(
    records: List[dict],
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> List[dict]:
    """
    Fallback signer/video-disjoint split assignment when official split labels are absent.
    Never performs frame-level splitting; all clips from the same signer (or video_id)
    stay in the same split.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_class[rec["gloss"]].append(rec)

    assigned: List[dict] = []
    for _, cls_records in sorted(by_class.items()):
        # Group by signer_id if available, otherwise video_id
        groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in cls_records:
            group_key = (
                f"signer_{rec['signer_id']}"
                if rec.get("signer_id") is not None
                else f"video_{rec['video_id']}"
            )
            groups[group_key].append(rec)

        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_test = max(1, int(round(n_groups * test_ratio))) if n_groups >= 3 else 0
        n_val = max(1, int(round(n_groups * val_ratio))) if n_groups >= 3 else 0
        if n_test + n_val >= n_groups:
            n_val = max(0, n_groups - n_test - 1)

        test_keys = set(group_keys[:n_test])
        val_keys = set(group_keys[n_test : n_test + n_val])

        for gk in group_keys:
            split_name = "test" if gk in test_keys else ("val" if gk in val_keys else "train")
            for rec in groups[gk]:
                rec_copy = dict(rec)
                rec_copy["split"] = split_name
                assigned.append(rec_copy)

    return assigned


def _collect_wlasl_records(
    wlasl_json: Path,
    video_dir: Path,
    top_k_glosses: int,
    seed: int,
) -> Tuple[List[dict], List[str]]:
    """Parses WLASL_v0.3.json and collects available video instances for the top-K glosses."""
    with open(wlasl_json, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    # Gather existing video instances per gloss
    gloss_to_records: Dict[str, List[dict]] = {}
    for entry in wlasl_data:
        gloss = entry["gloss"].strip()
        valid_instances = []
        for inst in entry.get("instances", []):
            vid_id = str(inst.get("video_id", "")).strip()
            if not vid_id:
                continue
            vid_path = _find_video_file(video_dir, vid_id, gloss=gloss)
            if vid_path is None:
                continue
            valid_instances.append(
                {
                    "gloss": gloss,
                    "video_id": vid_id,
                    "video_path": vid_path,
                    "frame_start": int(inst.get("frame_start", 1) or 1),
                    "frame_end": int(inst.get("frame_end", -1) or -1),
                    "signer_id": inst.get("signer_id"),
                    "split": _normalize_split_name(inst.get("split")),
                }
            )
        if valid_instances:
            gloss_to_records[gloss] = valid_instances

    if not gloss_to_records:
        raise FileNotFoundError(
            f"No matching video files found in '{video_dir}' for any gloss in '{wlasl_json}'."
        )

    # Select top-K glosses by available instance count (tie-break alphabetically)
    ranked_glosses = sorted(
        gloss_to_records.keys(),
        key=lambda g: (-len(gloss_to_records[g]), g),
    )[:top_k_glosses]
    class_names = sorted(ranked_glosses)

    selected_records: List[dict] = []
    missing_split_records: List[dict] = []
    for g in class_names:
        for rec in gloss_to_records[g]:
            if rec["split"] in ("train", "val", "test"):
                selected_records.append(rec)
            else:
                missing_split_records.append(rec)

    if missing_split_records:
        print(
            f"[!] {len(missing_split_records)} WLASL instances lacked official split tags; "
            "assigning signer/video-disjoint fallback splits."
        )
        selected_records.extend(_assign_group_disjoint_splits(missing_split_records, seed=seed))

    return selected_records, class_names


def _collect_folder_records(folder_dir: Path, seed: int) -> Tuple[List[dict], List[str]]:
    """
    Collects video records from `<root>/<split>/<class_name>/*.mp4`
    (or `<root>/<class_name>/*.mp4` with video-disjoint fallback splitting).
    """
    if not folder_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset directory not found: '{folder_dir}'")

    split_dirs = {
        split: folder_dir / split
        for split in ("train", "val", "test")
        if (folder_dir / split).is_dir()
    }

    records: List[dict] = []
    classes_set = set()

    if split_dirs:
        for split_name, split_path in split_dirs.items():
            for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
                gloss = class_dir.name
                classes_set.add(gloss)
                for vid_path in sorted(class_dir.iterdir()):
                    if vid_path.is_file() and vid_path.suffix.lower() in VIDEO_EXTENSIONS:
                        records.append(
                            {
                                "gloss": gloss,
                                "video_id": vid_path.stem,
                                "video_path": vid_path,
                                "frame_start": 1,
                                "frame_end": -1,
                                "signer_id": None,
                                "split": split_name,
                            }
                        )
    else:
        for class_dir in sorted(p for p in folder_dir.iterdir() if p.is_dir()):
            gloss = class_dir.name
            classes_set.add(gloss)
            for vid_path in sorted(class_dir.iterdir()):
                if vid_path.is_file() and vid_path.suffix.lower() in VIDEO_EXTENSIONS:
                    records.append(
                        {
                            "gloss": gloss,
                            "video_id": vid_path.stem,
                            "video_path": vid_path,
                            "frame_start": 1,
                            "frame_end": -1,
                            "signer_id": None,
                            "split": None,
                        }
                    )
        records = _assign_group_disjoint_splits(records, seed=seed)

    if not records:
        raise FileNotFoundError(f"No video files found under '{folder_dir}'.")

    return records, sorted(classes_set)


def build_landmark_dataset(
    records: List[dict],
    class_names: List[str],
    seq_len: int,
    min_valid_ratio: float,
    output_path: Path,
    source_mode: str,
) -> None:
    """Runs MediaPipe extraction on all records and saves the `.npz` dataset."""
    cfg = PipelineConfig(sequence_length=seq_len)
    extractor = HandFeatureExtractor(cfg)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    splits_x: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    splits_y: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    split_signers: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    valid_ratios: List[float] = []
    dropped_low_detection = 0
    dropped_unreadable = 0

    total = len(records)
    print(f"[*] Extracting MediaPipe landmarks for {total} clips across {len(class_names)} classes...")

    try:
        for i, rec in enumerate(records, start=1):
            seq, valid_ratio = _extract_clip_sequence(
                video_path=rec["video_path"],
                extractor=extractor,
                seq_len=seq_len,
                frame_start=rec["frame_start"],
                frame_end=rec["frame_end"],
            )
            if seq is None:
                dropped_unreadable += 1
                continue

            if valid_ratio < min_valid_ratio:
                dropped_low_detection += 1
                continue

            split = rec["split"]
            splits_x[split].append(seq)
            splits_y[split].append(class_to_idx[rec["gloss"]])
            valid_ratios.append(valid_ratio)
            if rec.get("signer_id") is not None:
                split_signers[split].add(rec["signer_id"])

            if i % 50 == 0 or i == total:
                print(f"    Processed {i}/{total} clips...")
    finally:
        extractor.close()

    def _pack(split: str) -> Tuple[np.ndarray, np.ndarray]:
        if not splits_x[split]:
            return (
                np.zeros((0, seq_len, cfg.feature_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.stack(splits_x[split], axis=0), np.array(splits_y[split], dtype=np.int64)

    X_train, y_train = _pack("train")
    X_val, y_val = _pack("val")
    X_test, y_test = _pack("test")

    mean_valid_rate = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    metadata = {
        "source_mode": source_mode,
        "seq_len": seq_len,
        "feature_dim": cfg.feature_dim,
        "min_valid_ratio": min_valid_ratio,
        "total_clips_scanned": total,
        "kept_clips": int(len(X_train) + len(X_val) + len(X_test)),
        "dropped_low_detection": dropped_low_detection,
        "dropped_unreadable": dropped_unreadable,
        "mean_valid_detection_rate": round(mean_valid_rate, 4),
        "split_counts": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "signer_counts": {
            split: len(signers) for split, signers in split_signers.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=np.array(class_names),
        metadata_json=np.array(json.dumps(metadata)),
    )

    print("\n" + "=" * 60)
    print("           LANDMARK EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Output Archive:          {output_path}")
    print(f"Classes ({len(class_names)}):            {', '.join(class_names)}")
    print(f"Train / Val / Test:      {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"Signer Counts (Tr/Va/Te):{len(split_signers['train'])} / {len(split_signers['val'])} / {len(split_signers['test'])}")
    print(f"Mean Hand Detection Rate:{mean_valid_rate * 100:.1f}%")
    print(f"Dropped (Low Detection): {dropped_low_detection}")
    print(f"Dropped (Unreadable):    {dropped_unreadable}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from WLASL or folder-based video datasets."
    )
    parser.add_argument("--wlasl-json", type=str, default=None, help="Path to WLASL_v0.3.json")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing WLASL video files")
    parser.add_argument("--folder-dir", type=str, default=None, help="Root directory of <split>/<class_name>/*.mp4")
    parser.add_argument("--top-k-glosses", type=int, default=15, help="Number of top WLASL glosses to extract")
    parser.add_argument("--seq-len", type=int, default=30, help="Target sequence length T per clip")
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.30,
        help="Minimum fraction of frames with detected hands to keep a clip",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wlasl_landmarks.npz",
        help="Output .npz path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fallback group-disjoint splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.wlasl_json and args.video_dir:
        records, class_names = _collect_wlasl_records(
            wlasl_json=Path(args.wlasl_json),
            video_dir=Path(args.video_dir),
            top_k_glosses=args.top_k_glosses,
            seed=args.seed,
        )
        source_mode = "wlasl"
    elif args.folder_dir:
        records, class_names = _collect_folder_records(
            folder_dir=Path(args.folder_dir),
            seed=args.seed,
        )
        source_mode = "folder"
    else:
        raise ValueError(
            "Provide either (--wlasl-json and --video-dir) for WLASL mode "
            "or (--folder-dir) for folder-based video dataset mode."
        )

    build_landmark_dataset(
        records=records,
        class_names=class_names,
        seq_len=args.seq_len,
        min_valid_ratio=args.min_valid_ratio,
        output_path=Path(args.output),
        source_mode=source_mode,
    )


if __name__ == "__main__":
    main()
