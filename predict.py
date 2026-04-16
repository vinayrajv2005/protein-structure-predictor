# predict.py
# Loads the trained model and predicts secondary structure for a given sequence

import torch
import numpy as np

from model import ProteinStructurePredictor
from data_utils import (
    encode_sequence, decode_labels,
    LABEL_NAMES, pad_sequence as pad_seq
)

# ─── Configuration ─────────────────────────────────────────────────────────────
VOCAB_SIZE  = 22
MAX_LEN     = 50
EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
NUM_CLASSES = 3
DROPOUT     = 0.0          # disable dropout at inference
MODEL_PATH  = "protein_model.pt"
PAD_IDX     = VOCAB_SIZE - 1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def load_model(model_path=MODEL_PATH):
    """Load the trained model from disk."""
    model = ProteinStructurePredictor(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def predict_structure(sequence: str, model=None):
    """
    Predict the secondary structure of a protein sequence.

    Args:
        sequence: String of single-letter amino acid codes (e.g., "ACDEFGHIKLM")
        model: Pre-loaded model (optional; loads from disk if None)

    Returns:
        dict with keys:
            - sequence: cleaned input sequence
            - structure: predicted structure string (H/E/C per position)
            - per_residue: list of dicts with per-residue detail
            - counts: dict with count of each structure type
            - percentages: dict with percentage of each structure type
    """
    if model is None:
        model = load_model()

    sequence = sequence.upper().strip()
    original_len = len(sequence)

    # Encode and pad
    encoded = encode_sequence(sequence)
    padded  = pad_seq(encoded, MAX_LEN, PAD_IDX)
    tensor  = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, MAX_LEN)

    with torch.no_grad():
        logits = model(tensor)                          # (1, MAX_LEN, 3)
        probs  = torch.softmax(logits, dim=-1)          # (1, MAX_LEN, 3)
        preds  = logits.argmax(dim=-1).squeeze(0)       # (MAX_LEN,)

    # Trim to original sequence length
    preds = preds[:original_len].cpu().numpy()
    probs = probs[0, :original_len].cpu().numpy()       # (original_len, 3)

    structure = decode_labels(preds)

    # Per-residue details
    per_residue = []
    label_keys = ['H', 'E', 'C']
    for i, (aa, label_idx) in enumerate(zip(sequence, preds)):
        per_residue.append({
            "position": i + 1,
            "amino_acid": aa,
            "structure": label_keys[label_idx],
            "structure_name": LABEL_NAMES[label_keys[label_idx]],
            "confidence": {
                "H": round(float(probs[i][0]), 4),
                "E": round(float(probs[i][1]), 4),
                "C": round(float(probs[i][2]), 4)
            }
        })

    # Summary counts
    counts = {"H": structure.count("H"), "E": structure.count("E"), "C": structure.count("C")}
    total  = len(structure)
    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}

    return {
        "sequence":    sequence,
        "structure":   structure,
        "per_residue": per_residue,
        "counts":      counts,
        "percentages": percentages
    }


if __name__ == "__main__":
    # Demo (run after training)
    test_seq = "ACDEFGHIKLMNPQRSTVWYACDE"
    result = predict_structure(test_seq)
    print(f"Sequence : {result['sequence']}")
    print(f"Structure: {result['structure']}")
    print(f"Counts   : {result['counts']}")
    print(f"Percents : {result['percentages']}")
