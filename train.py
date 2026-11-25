"""
Model Training, Evaluation, and Research Ablation Suite.

Trains lightweight temporal models (GRU, LSTM, TCN) on pre-split real landmark
archives (`.npz`), selects checkpoints strictly by validation Macro-F1, and
evaluates once on the held-out test split with full metrics, confusion matrix,
and CPU latency benchmarking.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader

from config import PipelineConfig
from dataset import GestureDataset, load_landmark_splits
from models import build_model


def set_seed(seed: int) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: nn.Module) -> Tuple[int, float]:
    """Returns (trainable_param_count, fp32_size_kb)."""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_kb = (params * 4.0) / 1024.0
    return params, size_kb


def benchmark_cpu_latency(
    model: nn.Module,
    seq_len: int,
    input_dim: int,
    warmup_runs: int = 30,
    timed_runs: int = 300,
) -> Tuple[float, float]:
    """
    Measures single-sample CPU inference latency (mean ms and P95 ms).
    """
    cpu_model = model.to(torch.device("cpu"))
    cpu_model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = cpu_model(dummy_input)

        latencies_ms: List[float] = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = cpu_model(dummy_input)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    return mean_ms, p95_ms


def save_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_type: str,
    out_path: Path,
) -> None:
    """Renders and saves a normalized confusion matrix heatmap."""
    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(np.float32), row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums != 0)

    fig_size = max(6.5, min(14.0, 0.65 * len(class_names) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Test Confusion Matrix ({model_type.upper()})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j] * 100:.0f}%)",
                ha="center",
                va="center",
                fontsize=7 if len(class_names) > 10 else 8,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Runs evaluation on a DataLoader and returns (loss, y_true, y_pred, y_probs)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch_y.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=1)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, y_true, y_pred, y_probs


def train_and_evaluate(
    data_path: str,
    model_type: str = "gru",
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    save_path: str = "gesture_model.pt",
    results_dir: str = "results",
) -> Dict:
    """
    Trains a single model architecture on pre-split landmark data, selects the
    best checkpoint by validation Macro-F1, and evaluates once on held-out test.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Training {model_type.upper()} on device: {device} (seed={seed})")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = load_landmark_splits(data_path)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"All splits must be non-empty. Got train={len(X_train)}, val={len(X_val)}, test={len(X_test)}."
        )

    seq_len = int(X_train.shape[1])
    input_dim = int(X_train.shape[2])

    config = PipelineConfig(
        model_type=model_type,
        sequence_length=seq_len,
        feature_dim=input_dim,
        labels=class_names,
    )

    # Separate datasets: augment=True ONLY on train, never on val or test
    train_dataset = GestureDataset(X_train, y_train, augment=True, frame_drop_prob=0.05)
    val_dataset = GestureDataset(X_val, y_val, augment=False, frame_drop_prob=0.0)
    test_dataset = GestureDataset(X_test, y_test, augment=False, frame_drop_prob=0.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(config).to(device)
    param_count, size_kb = count_trainable_parameters(model)
    print(f"[*] {model_type.upper()} trainable params: {param_count:,} ({size_kb:.1f} KB FP32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_macro_f1 = -1.0
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = (correct / max(total, 1)) * 100.0

        val_loss, val_y_true, val_y_pred, _ = evaluate_split(model, val_loader, criterion, device)
        val_acc = float(np.mean(val_y_true == val_y_pred) * 100.0)
        val_macro_f1 = float(f1_score(val_y_true, val_y_pred, average="macro", zero_division=0) * 100.0)

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% Macro-F1: {val_macro_f1:.2f}%"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": config.model_type,
                    "class_names": class_names,
                    "num_classes": config.num_classes,
                    "input_dim": config.feature_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "bidirectional": config.bidirectional,
                    "val_macro_f1": best_val_macro_f1,
                    "seed": seed,
                },
                save_file,
            )

    # Load best checkpoint and evaluate ONCE on held-out test split
    ckpt = torch.load(save_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_y_true, test_y_pred, test_y_probs = evaluate_split(model, test_loader, criterion, device)
    test_top1 = float(np.mean(test_y_true == test_y_pred) * 100.0)

    k_top = min(3, config.num_classes)
    test_top3 = float(
        top_k_accuracy_score(
            test_y_true,
            test_y_probs,
            k=k_top,
            labels=list(range(config.num_classes)),
        )
        * 100.0
    )
    test_macro_f1 = float(f1_score(test_y_true, test_y_pred, average="macro", zero_division=0) * 100.0)

    cpu_mean_ms, cpu_p95_ms = benchmark_cpu_latency(model, seq_len=seq_len, input_dim=input_dim)

    cm_path = Path(results_dir) / f"confusion_matrix_{model_type.lower()}.png"
    save_confusion_matrix_png(test_y_true, test_y_pred, class_names, model_type, cm_path)

    report = classification_report(
        test_y_true,
        test_y_pred,
        labels=list(range(config.num_classes)),
        target_names=class_names,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"      HELD-OUT TEST RESULTS ({model_type.upper()})")
    print("=" * 60)
    print(f"Best Val Macro-F1:   {best_val_macro_f1:.2f}%")
    print(f"Test Top-1 Accuracy: {test_top1:.2f}%")
    print(f"Test Top-{k_top} Accuracy: {test_top3:.2f}%")
    print(f"Test Macro-F1:       {test_macro_f1:.2f}%")
    print(f"Trainable Params:    {param_count:,} ({size_kb:.1f} KB FP32)")
    print(f"CPU Latency (1x):    {cpu_mean_ms:.3f} ms mean | {cpu_p95_ms:.3f} ms P95")
    print(f"Checkpoint Saved:    {os.path.abspath(save_file)}")
    print(f"Confusion Matrix:    {os.path.abspath(cm_path)}")
    print("-" * 60)
    print(report)
    print("=" * 60)

    return {
        "model_type": model_type.lower(),
        "params": param_count,
        "size_kb": round(size_kb, 2),
        "val_macro_f1": round(best_val_macro_f1, 2),
        "test_top1_acc": round(test_top1, 2),
        "test_top3_acc": round(test_top3, 2),
        "test_macro_f1": round(test_macro_f1, 2),
        "cpu_mean_ms": round(cpu_mean_ms, 3),
        "cpu_p95_ms": round(cpu_p95_ms, 3),
        "seed": seed,
        "checkpoint_path": str(save_file),
        "confusion_matrix_path": str(cm_path),
    }


def run_ablation_suite(
    data_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    results_dir: str = "results",
) -> None:
    """Runs GRU, LSTM, and TCN back-to-back with identical seeds and splits."""
    architectures = ["gru", "lstm", "tcn"]
    results: List[Dict] = []

    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    for arch in architectures:
        ckpt_path = str(res_path / f"gesture_{arch}.pt")
        metrics = train_and_evaluate(
            data_path=data_path,
            model_type=arch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            save_path=ckpt_path,
            results_dir=results_dir,
        )
        results.append(metrics)

    # Build GitHub-flavored Markdown comparison table
    lines = [
        "| Model | Params | FP32 Size (KB) | Val Macro-F1 (%) | Test Top-1 (%) | Test Top-3 (%) | Test Macro-F1 (%) | CPU Mean (ms) | CPU P95 (ms) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r['model_type'].upper()}` | {r['params']:,} | {r['size_kb']:.1f} | "
            f"{r['val_macro_f1']:.2f} | {r['test_top1_acc']:.2f} | {r['test_top3_acc']:.2f} | "
            f"{r['test_macro_f1']:.2f} | {r['cpu_mean_ms']:.3f} | {r['cpu_p95_ms']:.3f} |"
        )
    table_md = "\n".join(lines) + "\n"

    summary_json_path = res_path / "ablation_summary.json"
    table_md_path = res_path / "ablation_table.md"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    print("\n### Architecture Ablation Summary\n")
    print(table_md)
    print(f"[+] Saved ablation JSON to: {summary_json_path}")
    print(f"[+] Saved ablation Markdown table to: {table_md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and Evaluate Dynamic Gesture Sequence Models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/wlasl_landmarks.npz",
        help="Path to pre-split landmark .npz archive generated by extract_landmarks.py",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--model-type", type=str, default="gru", choices=["gru", "lstm", "tcn"])
    parser.add_argument("--save-path", type=str, default="gesture_model.pt", help="Checkpoint output path")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for evaluation artifacts")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run GRU, LSTM, and TCN back-to-back with identical seeds and splits",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ablation:
        run_ablation_suite(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            results_dir=args.results_dir,
        )
    else:
        train_and_evaluate(
            data_path=args.data,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            save_path=args.save_path,
            results_dir=args.results_dir,
        )
