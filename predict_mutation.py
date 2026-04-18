# predict_mutation.py
# Loads the trained MutationPredictor and scores a point mutation.

import torch
import numpy as np

from mutation_model import MutationPredictor, AA_TO_IDX, NUM_AA
from mutation_data_utils import build_feature_vector, blosum62_score, _charge, AMINO_ACIDS

# ─── Config (must match train_mutation.py) ────────────────────────────────────
EMBED_DIM   = 32
HIDDEN_DIM  = 128
DROPOUT     = 0.0          # no dropout at inference
MODEL_PATH  = "mutation_model.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def load_mutation_model(path=MODEL_PATH):
    model = MutationPredictor(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                               dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


_MUTATION_MODEL = None   # module-level cache


def get_mutation_model():
    global _MUTATION_MODEL
    if _MUTATION_MODEL is None:
        import os
        if not os.path.exists(MODEL_PATH):
            return None
        _MUTATION_MODEL = load_mutation_model()
    return _MUTATION_MODEL


def predict_mutation(ref_aa: str, alt_aa: str, position: int,
                     protein_length: int = 500, model=None) -> dict:
    """
    Predict pathogenicity of a single amino acid substitution.

    Args:
        ref_aa         : Wild-type amino acid (single letter, e.g. 'A')
        alt_aa         : Mutant amino acid (single letter, e.g. 'V')
        position       : 1-based position in the protein
        protein_length : Total protein length (for normalisation)
        model          : Pre-loaded model (loads from disk if None)

    Returns dict with:
        ref_aa, alt_aa, position,
        probability     : float 0–1  (probability of being pathogenic)
        prediction      : 'Pathogenic' | 'Benign'
        confidence      : float 0–1  (distance from 0.5 decision boundary)
        risk_level      : 'High' | 'Moderate' | 'Low'
        blosum_score    : int
        interpretation  : human-readable explanation
        demo_mode       : bool
    """
    ref_aa = ref_aa.upper()
    alt_aa = alt_aa.upper()

    if model is None:
        model = get_mutation_model()

    # Compute shared features regardless of model availability
    bl_score    = blosum62_score(ref_aa, alt_aa)
    charge_diff = abs(_charge(alt_aa) - _charge(ref_aa))

    if model is None:
        return _demo_mutation_predict(ref_aa, alt_aa, position, bl_score, charge_diff)

    feat_np = build_feature_vector(ref_aa, alt_aa, position, protein_length)
    feat_t  = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    ref_t   = torch.tensor([AA_TO_IDX.get(ref_aa, NUM_AA)], dtype=torch.long).to(DEVICE)
    alt_t   = torch.tensor([AA_TO_IDX.get(alt_aa, NUM_AA)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logit = model(ref_t, alt_t, feat_t)
        prob  = torch.sigmoid(logit).item()

    return _format_result(ref_aa, alt_aa, position, prob, bl_score, demo_mode=False)


def _demo_mutation_predict(ref_aa, alt_aa, position, bl_score, charge_diff):
    """Heuristic-based demo prediction when no trained model is available."""
    import random
    random.seed(hash(f"{ref_aa}{alt_aa}{position}") & 0xFFFFFFFF)

    # Simple heuristic probability
    base = 0.30
    if bl_score <= -3:
        base += 0.35
    elif bl_score <= 0:
        base += 0.15
    if ref_aa == 'C' or alt_aa == 'C':
        base += 0.15
    if charge_diff > 0.5:
        base += 0.10
    prob = min(max(base + random.uniform(-0.05, 0.05), 0.01), 0.99)

    result = _format_result(ref_aa, alt_aa, position, prob, bl_score, demo_mode=True)
    return result


def _format_result(ref_aa, alt_aa, position, prob, bl_score, demo_mode):
    prediction = "Pathogenic" if prob >= 0.5 else "Benign"
    confidence = abs(prob - 0.5) * 2   # 0 = uncertain, 1 = certain

    if prob >= 0.75:
        risk_level = "High"
    elif prob >= 0.50:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Build interpretation
    parts = []
    if bl_score <= -3:
        parts.append("radical amino acid substitution (low BLOSUM62 score)")
    elif bl_score >= 1:
        parts.append("conservative substitution (positive BLOSUM62 score)")
    else:
        parts.append("semi-conservative substitution")

    if ref_aa == 'C' or alt_aa == 'C':
        parts.append("cysteine involvement (often disrupts disulfide bonds)")

    interpretation = (
        f"The {ref_aa}→{alt_aa} substitution at position {position} involves a "
        + " and ".join(parts) + f". "
        f"BLOSUM62 score: {bl_score}. "
        f"Predicted as {prediction} with {confidence*100:.0f}% confidence."
    )

    return {
        "ref_aa":        ref_aa,
        "alt_aa":        alt_aa,
        "position":      position,
        "probability":   round(prob, 4),
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "risk_level":    risk_level,
        "blosum_score":  bl_score,
        "interpretation": interpretation,
        "demo_mode":     demo_mode,
    }


if __name__ == "__main__":
    result = predict_mutation("A", "V", position=42, protein_length=300)
    for k, v in result.items():
        print(f"{k:18s}: {v}")
