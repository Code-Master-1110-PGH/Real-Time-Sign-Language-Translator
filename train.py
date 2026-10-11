"""
Model Training and Checkpoint Pipeline.

Trains the lightweight temporal model (GRU/LSTM/TCN) with cosine learning rate
scheduling, gradient clipping, validation monitoring, and checkpoint persistence.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import PipelineConfig
from dataset import GestureDataset, SyntheticGestureGenerator
from models import build_model


def train_model(
    config: PipelineConfig,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str = "gesture_model.pt",
) -> None:
    """
    Executes training and validation loop for dynamic gesture classification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training target device: {device}")

    # 1. Dataset Generation / Loading
    print("[*] Synthesizing kinematic trajectory dataset...")
    X, y = SyntheticGestureGenerator.generate_dataset(config, samples_per_class=120)
    print(f"[+] Dataset created with {len(X)} total dynamic samples across {config.num_classes} classes.")

    dataset = GestureDataset(X, y, augment=True)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. Model Initialization
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print(f"[*] Beginning training ({config.model_type.upper()} for {epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Prevent gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_acc = (correct / total) * 100.0
        train_loss = total_loss / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                v_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == batch_y).sum().item()
                v_total += batch_x.size(0)

        val_acc = (v_correct / v_total) * 100.0
        val_loss = v_loss / v_total
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": config.model_type,
                    "val_acc": val_acc,
                },
                save_path,
            )

    print(f"\n[+] Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[+] Model checkpoint persisted to: '{os.path.abspath(save_path)}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dynamic Gesture Sequence Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-type", type=str, default="gru", choices=["gru", "lstm", "tcn"])
    args = parser.parse_args()

    cfg = PipelineConfig(model_type=args.model_type)
    train_model(
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=cfg.model_weights_path,
    )

