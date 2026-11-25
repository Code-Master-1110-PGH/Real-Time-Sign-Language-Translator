"""
Unit Tests for Real-Time Sign Language Translator Pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch

from config import PipelineConfig
from dataset import GestureDataset, load_landmark_splits
from feature_extractor import LandmarkNormalizer
from models import GestureGRU, GestureLSTM, GestureTCN, build_model


def test_model_forward_shapes_and_gru_param_count():
    batch_size, seq_len, input_dim, num_classes = 4, 30, 126, 8
    x = torch.randn(batch_size, seq_len, input_dim)

    gru = GestureGRU(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        num_classes=num_classes,
        bidirectional=False,
    )
    gru_params = sum(p.numel() for p in gru.parameters() if p.requires_grad)
    assert gru_params == 64969, f"Expected 64,969 params for GestureGRU, got {gru_params}"
    assert gru(x).shape == (batch_size, num_classes)

    lstm = GestureLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        num_classes=num_classes,
        bidirectional=False,
    )
    assert lstm(x).shape == (batch_size, num_classes)

    tcn = GestureTCN(
        input_dim=input_dim,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=2,
    )
    assert tcn(x).shape == (batch_size, num_classes)

    # Verify build_model factory aligns with PipelineConfig
    for m_type in ("gru", "lstm", "tcn"):
        cfg = PipelineConfig(model_type=m_type)
        model = build_model(cfg)
        assert model(x).shape == (batch_size, cfg.num_classes)


def test_tcn_strict_causality():
    torch.manual_seed(0)
    model = GestureTCN(input_dim=126, hidden_dim=64, num_classes=8, num_layers=3, dropout=0.0)
    model.eval()

    x_orig = torch.randn(2, 30, 126)
    x_perturbed = x_orig.clone()
    t0 = 14
    # Perturb strictly future timesteps t > t0
    x_perturbed[:, t0 + 1 :, :] += 10.0

    with torch.no_grad():
        feats_orig = model.tcn(x_orig.transpose(1, 2))
        feats_pert = model.tcn(x_perturbed.transpose(1, 2))

    # Outputs at t <= t0 must be identical
    assert torch.allclose(feats_orig[:, :, : t0 + 1], feats_pert[:, :, : t0 + 1], atol=1e-6)
    # Outputs at t > t0 should differ
    assert not torch.allclose(feats_orig[:, :, t0 + 1 :], feats_pert[:, :, t0 + 1 :], atol=1e-6)


def test_gesture_dataset_augment_behavior(tmp_path):
    seq = np.ones((2, 30, 126), dtype=np.float32)
    # Mark frames 0..9 as missing-hand frames (strictly zero)
    seq[:, :10, :] = 0.0
    labels = np.array([0, 1], dtype=np.int64)

    # augment=False must be deterministic and exact
    ds_eval = GestureDataset(seq, labels, augment=False, frame_drop_prob=0.5)
    x0, y0 = ds_eval[0]
    x0_repeat, _ = ds_eval[0]
    assert torch.equal(x0, torch.from_numpy(seq[0]))
    assert torch.equal(x0, x0_repeat)
    assert y0.item() == 0

    # augment=True must preserve exact zeros on missing-hand frames
    np.random.seed(123)
    ds_train = GestureDataset(seq, labels, augment=True, frame_drop_prob=0.0, temporal_jitter=False)
    x_aug, _ = ds_train[0]
    assert torch.all(x_aug[:10, :] == 0.0), "Missing-hand frames must remain strictly zero"
    assert not torch.allclose(x_aug[10:, :], torch.from_numpy(seq[0, 10:, :])), (
        "Non-zero frames should be perturbed when augment=True"
    )

    # Verify load_landmark_splits roundtrip
    npz_file = tmp_path / "sample_landmarks.npz"
    np.savez(
        npz_file,
        X_train=seq,
        y_train=labels,
        X_val=seq[:1],
        y_val=labels[:1],
        X_test=seq[1:],
        y_test=labels[1:],
        class_names=np.array(["SIGN_A", "SIGN_B"]),
    )
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), classes = load_landmark_splits(str(npz_file))
    assert X_tr.shape == (2, 30, 126)
    assert X_va.shape == (1, 30, 126)
    assert X_te.shape == (1, 30, 126)
    assert classes == ["SIGN_A", "SIGN_B"]


def test_kinematic_normalization_invariance():
    rng = np.random.default_rng(42)
    raw_hand = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    # Ensure wrist-to-middle-MCP (index 9) has non-trivial distance
    raw_hand[9] = raw_hand[0] + np.array([0.2, 0.3, 0.05], dtype=np.float32)

    norm_base = LandmarkNormalizer.normalize_hand_landmarks(raw_hand)

    # 1. Wrist origin centering: landmark[0] == [0, 0, 0]
    assert np.allclose(norm_base[0], np.zeros(3, dtype=np.float32), atol=1e-6)

    # 2. Translation invariance
    translated = raw_hand + np.array([12.5, -7.3, 4.1], dtype=np.float32)
    norm_translated = LandmarkNormalizer.normalize_hand_landmarks(translated)
    assert np.allclose(norm_base, norm_translated, atol=1e-5)

    # 3. Scale invariance
    scaled = raw_hand * 2.75
    norm_scaled = LandmarkNormalizer.normalize_hand_landmarks(scaled)
    assert np.allclose(norm_base, norm_scaled, atol=1e-5)
