# mutation_data_utils.py
# Encoding helpers, BLOSUM62 scores, and synthetic dataset generation
# for the mutation disease-prediction module.

import numpy as np

# ─── Amino acid alphabet ──────────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA      = len(AMINO_ACIDS)   # 20
UNKNOWN_IDX = NUM_AA             # index 20

# ─── Simplified BLOSUM62 substitution scores ─────────────────────────────────
# Full 20×20 matrix; positive = conservative, negative = radical substitution
_BLOSUM62_RAW = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,
    ('A','Q'):-1,('A','E'):-1,('A','G'):0,('A','H'):-2,('A','I'):-1,
    ('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,('A','P'):-1,
    ('A','S'):1,('A','T'):0,('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5,('R','N'):-1,('R','D'):-2,('R','C'):-3,('R','Q'):1,
    ('R','E'):0,('R','G'):-2,('R','H'):0,('R','I'):-3,('R','L'):-2,
    ('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,
    ('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6,('N','D'):1,('N','C'):-3,('N','Q'):0,('N','E'):0,
    ('N','G'):0,('N','H'):1,('N','I'):-3,('N','L'):-3,('N','K'):0,
    ('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6,('D','C'):-3,('D','Q'):0,('D','E'):2,('D','G'):-1,
    ('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,('D','M'):-3,
    ('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,
    ('C','C'):9,('C','Q'):-3,('C','E'):-4,('C','G'):-3,('C','H'):-3,
    ('C','I'):-1,('C','L'):-1,('C','K'):-3,('C','M'):-1,('C','F'):-2,
    ('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,('C','Y'):-2,
    ('C','V'):-1,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-2,('Q','H'):0,('Q','I'):-3,
    ('Q','L'):-2,('Q','K'):1,('Q','M'):0,('Q','F'):-3,('Q','P'):-1,
    ('Q','S'):0,('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,
    ('E','E'):5,('E','G'):-2,('E','H'):0,('E','I'):-3,('E','L'):-3,
    ('E','K'):1,('E','M'):-2,('E','F'):-3,('E','P'):-1,('E','S'):0,
    ('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-3,('G','P'):-2,('G','S'):0,('G','T'):-2,
    ('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8,('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,
    ('H','F'):-1,('H','P'):-2,('H','S'):-1,('H','T'):-2,('H','W'):-2,
    ('H','Y'):2,('H','V'):-3,
    ('I','I'):4,('I','L'):2,('I','K'):-1,('I','M'):1,('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,('K','M'):-1,('K','F'):-3,('K','P'):-1,('K','S'):0,
    ('K','T'):-1,('K','W'):-3,('K','Y'):-2,('K','V'):-2,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3,('F','V'):-1,
    ('P','P'):7,('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,
    ('P','V'):-2,
    ('S','S'):4,('S','T'):1,('S','W'):-3,('S','Y'):-2,('S','V'):-2,
    ('T','T'):5,('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2,('W','V'):-3,
    ('Y','Y'):7,('Y','V'):-1,
    ('V','V'):4,
}

def blosum62_score(aa1: str, aa2: str) -> int:
    """Return BLOSUM62 score for a pair of amino acids."""
    aa1, aa2 = aa1.upper(), aa2.upper()
    return _BLOSUM62_RAW.get((aa1, aa2), _BLOSUM62_RAW.get((aa2, aa1), 0))

# ─── Encoding ─────────────────────────────────────────────────────────────────

def encode_aa(aa: str) -> int:
    return AA_TO_IDX.get(aa.upper(), UNKNOWN_IDX)


def build_feature_vector(ref_aa: str, alt_aa: str, position: int,
                          protein_length: int = 500) -> np.ndarray:
    """
    Build the extra numeric feature vector for one mutation:
        [normalized_position, blosum_score_normalized, charge_change]
    Returns float32 array of shape (3,)
    """
    norm_pos   = position / max(protein_length, 1)         # 0–1
    bl_score   = blosum62_score(ref_aa, alt_aa) / 11.0     # approx –1 to 1
    charge_chg = _charge(alt_aa) - _charge(ref_aa)         # –2 to +2
    charge_chg /= 2.0                                       # normalise

    return np.array([norm_pos, bl_score, charge_chg], dtype=np.float32)


_CHARGED = {
    'R': +1, 'K': +1, 'H': +0.5,
    'D': -1, 'E': -1,
}
def _charge(aa: str) -> float:
    return _CHARGED.get(aa.upper(), 0.0)


# ─── Synthetic dataset ────────────────────────────────────────────────────────

def create_mutation_dataset(n_samples: int = 2000, seed: int = 42):
    """
    Generate a synthetic labelled dataset of protein point mutations.

    Label heuristics (mimics real biology, not ground truth):
        - Highly negative BLOSUM score → more likely pathogenic
        - Cysteine involved → more likely pathogenic
        - Conservative substitution → more likely benign

    Returns:
        ref_aas   : list[str]
        alt_aas   : list[str]
        positions : list[int]
        features  : np.ndarray (N, 3)
        labels    : np.ndarray (N,)  — 1 = pathogenic, 0 = benign
    """
    rng = np.random.default_rng(seed)
    aa_list = list(AMINO_ACIDS)

    ref_aas, alt_aas, positions, labels = [], [], [], []
    protein_lengths = rng.integers(50, 500, size=n_samples)

    for i in range(n_samples):
        ref = rng.choice(aa_list)
        alt = rng.choice([a for a in aa_list if a != ref])
        pos = int(rng.integers(1, protein_lengths[i] + 1))

        bl = blosum62_score(ref, alt)
        # Probabilistic label based on biology-inspired rules
        p_patho = 0.3   # base rate
        if bl <= -3:
            p_patho += 0.35
        elif bl <= 0:
            p_patho += 0.15
        if ref == 'C' or alt == 'C':
            p_patho += 0.20
        if abs(_charge(ref) - _charge(alt)) > 0.5:
            p_patho += 0.10
        p_patho = min(p_patho, 0.95)

        label = int(rng.random() < p_patho)
        ref_aas.append(ref)
        alt_aas.append(alt)
        positions.append(pos)
        labels.append(label)

    features = np.array([
        build_feature_vector(r, a, p, int(pl))
        for r, a, p, pl in zip(ref_aas, alt_aas, positions, protein_lengths)
    ], dtype=np.float32)

    return ref_aas, alt_aas, positions, features, np.array(labels, dtype=np.float32)


if __name__ == "__main__":
    refs, alts, poss, feats, lbls = create_mutation_dataset(10)
    for r, a, p, f, l in zip(refs, alts, poss, feats, lbls):
        print(f"{r}→{a} pos={p:3d} feat={f} label={'PATHO' if l else 'BENIGN'}")
