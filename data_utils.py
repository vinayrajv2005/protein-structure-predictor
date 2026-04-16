# data_utils.py
# Handles encoding of amino acid sequences and structure labels

import numpy as np

# Standard 20 amino acids + unknown
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_IDX = len(AMINO_ACIDS)  # index 20 for unknown

# Secondary structure labels
STRUCTURE_LABELS = ['H', 'E', 'C']  # Helix, Sheet (strand), Coil
LABEL_TO_IDX = {label: idx for idx, label in enumerate(STRUCTURE_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
LABEL_NAMES = {
    'H': 'Alpha Helix',
    'E': 'Beta Sheet',
    'C': 'Coil/Loop'
}

def encode_sequence(sequence):
    """
    Encode an amino acid sequence as integer indices.
    Unknown amino acids are mapped to UNKNOWN_IDX.
    Returns a numpy array of shape (len(sequence),)
    """
    sequence = sequence.upper().strip()
    encoded = [AA_TO_IDX.get(aa, UNKNOWN_IDX) for aa in sequence]
    return np.array(encoded, dtype=np.int64)

def encode_labels(structure_string):
    """
    Encode a structure label string (e.g., 'HHHEEECC') to integer indices.
    Returns a numpy array of shape (len(structure_string),)
    """
    structure_string = structure_string.upper().strip()
    encoded = [LABEL_TO_IDX.get(s, 2) for s in structure_string]  # default to Coil
    return np.array(encoded, dtype=np.int64)

def decode_labels(label_indices):
    """
    Decode integer label indices back to structure string.
    """
    return ''.join([IDX_TO_LABEL.get(idx, 'C') for idx in label_indices])

def pad_sequence(sequence, max_len, pad_value=UNKNOWN_IDX + 1):
    """
    Pad or truncate a sequence to max_len.
    """
    seq = list(sequence)
    if len(seq) >= max_len:
        return np.array(seq[:max_len], dtype=np.int64)
    else:
        padded = seq + [pad_value] * (max_len - len(seq))
        return np.array(padded, dtype=np.int64)

def create_sample_dataset(n_samples=500, max_len=50):
    """
    Generate a synthetic dataset for demonstration/testing.
    Each sample is a random amino acid sequence with randomly assigned structure labels.
    In a real project, replace this with actual protein databases like CB513 or CASP.
    """
    np.random.seed(42)
    sequences = []
    labels = []
    lengths = []

    for _ in range(n_samples):
        length = np.random.randint(20, max_len + 1)
        seq = np.random.randint(0, len(AMINO_ACIDS), size=length)
        label = np.random.randint(0, 3, size=length)
        sequences.append(seq)
        labels.append(label)
        lengths.append(length)

    return sequences, labels, lengths


if __name__ == "__main__":
    # Quick test
    seq = "ACDEFGHIKLMNPQRSTVWY"
    encoded = encode_sequence(seq)
    print("Encoded:", encoded)

    structure = "HHHEEECCCHHHEEE"
    enc_labels = encode_labels(structure)
    print("Label indices:", enc_labels)
    print("Decoded back:", decode_labels(enc_labels))
