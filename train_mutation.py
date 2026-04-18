# train_mutation.py
# Trains the MutationPredictor on synthetic (or real ClinVar) data.
# Run:  python train_mutation.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report

from mutation_model import MutationPredictor, AA_TO_IDX, NUM_AA
from mutation_data_utils import create_mutation_dataset

# ─── Config ───────────────────────────────────────────────────────────────────
EMBED_DIM   = 32
HIDDEN_DIM  = 128
DROPOUT     = 0.3
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
MODEL_PATH  = "mutation_model.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


class MutationDataset(Dataset):
    def __init__(self, ref_aas, alt_aas, features, labels):
        self.refs    = [AA_TO_IDX.get(r, NUM_AA) for r in ref_aas]
        self.alts    = [AA_TO_IDX.get(a, NUM_AA) for a in alt_aas]
        self.feats   = torch.tensor(features, dtype=torch.float32)
        self.labels  = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.refs[idx],  dtype=torch.long),
            torch.tensor(self.alts[idx],  dtype=torch.long),
            self.feats[idx],
            self.labels[idx],
        )


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for ref, alt, feat, lbl in loader:
            ref, alt, feat, lbl = ref.to(DEVICE), alt.to(DEVICE), feat.to(DEVICE), lbl.to(DEVICE)
            logit = model(ref, alt, feat)
            loss  = criterion(logit, lbl)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logit).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    avg_loss = total_loss / len(loader)
    preds    = (np.array(all_probs) >= 0.5).astype(int)
    acc      = (preds == np.array(all_labels)).mean()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return avg_loss, acc, auc, all_probs, all_labels


def train():
    print(f"Device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating synthetic dataset …")
    refs, alts, poss, feats, labels = create_mutation_dataset(n_samples=5000)

    n = len(labels)
    split = int(0.8 * n)

    train_ds = MutationDataset(refs[:split], alts[:split], feats[:split], labels[:split])
    val_ds   = MutationDataset(refs[split:], alts[split:], feats[split:], labels[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Class imbalance weight
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)],
                               dtype=torch.float32).to(DEVICE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = MutationPredictor(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                                   dropout=DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0.0

    print(f"\nTraining for {EPOCHS} epochs …\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for ref, alt, feat, lbl in train_loader:
            ref, alt, feat, lbl = (ref.to(DEVICE), alt.to(DEVICE),
                                    feat.to(DEVICE), lbl.to(DEVICE))
            optimizer.zero_grad()
            logit = model(ref, alt, feat)
            loss  = criterion(logit, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            marker = "  ✓ saved"

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}{marker}")

    # Final report
    _, _, _, probs, true_labels = evaluate(model, val_loader, criterion)
    preds = (np.array(probs) >= 0.5).astype(int)
    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(true_labels, preds,
                                 target_names=["Benign", "Pathogenic"]))
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
