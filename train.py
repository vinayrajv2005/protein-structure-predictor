# train.py
# Trains the ProteinStructurePredictor on synthetic or real data

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import ProteinStructurePredictor
from data_utils import create_sample_dataset, pad_sequence as pad_seq

# ─── Configuration ────────────────────────────────────────────────────────────
VOCAB_SIZE   = 22          # 20 AAs + unknown + padding token
MAX_LEN      = 50
EMBED_DIM    = 64
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
NUM_CLASSES  = 3
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-3
MODEL_PATH   = "protein_model.pt"
PAD_IDX      = VOCAB_SIZE - 1   # padding token index
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein sequences and structure labels."""

    def __init__(self, sequences, labels, max_len=MAX_LEN):
        self.samples = []
        for seq, lbl in zip(sequences, labels):
            seq_t = torch.tensor(pad_seq(seq, max_len, PAD_IDX), dtype=torch.long)
            lbl_t = torch.tensor(pad_seq(lbl, max_len, pad_value=-1), dtype=torch.long)
            self.samples.append((seq_t, lbl_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_accuracy(logits, labels, ignore_index=-1):
    """Per-token accuracy, ignoring padding positions."""
    preds = logits.argmax(dim=-1)           # (batch, seq_len)
    mask = labels != ignore_index
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def train():
    print(f"Using device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    sequences, labels, _ = create_sample_dataset(n_samples=1000, max_len=MAX_LEN)

    split = int(0.8 * len(sequences))
    train_ds = ProteinDataset(sequences[:split], labels[:split])
    val_ds   = ProteinDataset(sequences[split:], labels[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ProteinStructurePredictor(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_acc = 0.0, 0.0

        for seqs, lbls in train_loader:
            seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()

            logits = model(seqs)                         # (B, L, C)
            # Reshape for CrossEntropyLoss: (B*L, C) vs (B*L,)
            loss = criterion(logits.view(-1, NUM_CLASSES), lbls.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc  += compute_accuracy(logits, lbls)

        avg_loss = total_loss / len(train_loader)
        avg_acc  = total_acc  / len(train_loader)

        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for seqs, lbls in val_loader:
                seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
                logits = model(seqs)
                val_acc += compute_accuracy(logits, lbls)
        val_acc /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {avg_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Model saved (val acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
