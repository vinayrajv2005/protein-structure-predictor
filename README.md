
# 🧬 Protein Structure & Mutation Disease Predictor

A Flask web application that uses deep learning to predict protein secondary structure and assess the pathogenicity of amino acid mutations.

---

## Features

- **Secondary Structure Prediction** — Predicts Alpha Helix (H), Beta Sheet (E), or Coil/Loop (C) for every residue in a protein sequence using a Bidirectional LSTM.
- **Mutation Pathogenicity Prediction** — Scores a single amino acid substitution as Pathogenic or Benign using an MLP with biologically-informed features.
- **Sequence Scanner** — Scans every position in a sequence and ranks the most disruptive possible mutations.
- **Demo Mode** — Falls back to heuristic-based predictions if trained model files are not present, so the app never crashes.
- **REST API** — Clean JSON endpoints for easy integration with other tools.

---

## Project Structure

.
├── app.py                  # Flask app and API routes
├── model.py                # Bidirectional LSTM architecture
├── mutation_model.py       # MLP mutation predictor architecture
├── data_utils.py           # Sequence encoding and synthetic dataset generation
├── mutation_data_utils.py  # BLOSUM62 scores, mutation feature engineering
├── predict.py              # Inference for secondary structure
├── predict_mutation.py     # Inference for mutation pathogenicity
├── train.py                # Training script for structure model
├── train_mutation.py       # Training script for mutation model
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI
├── protein_model.pt        # Trained structure model weights (after training)
└── mutation_model.pt       # Trained mutation model weights (after training)


---

## Installation

**1. Clone the repository**
bash
git clone https://github.com/your-username/protein-predictor.git
cd protein-predictor


**2. Create a virtual environment (recommended)**
bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate


**3. Install dependencies**
bash
pip install -r requirements.txt


---

## Training the Models

Train the secondary structure predictor:
bash
python train.py


Train the mutation disease predictor:
bash
python train_mutation.py


Both scripts train on synthetic data by default and save model weights to protein_model.pt and mutation_model.pt. To use real biological data, replace the dataset generation calls in data_utils.py and mutation_data_utils.py with a loader for databases like CB513, CASP, or ClinVar.

---

## Running the App

bash
python app.py


Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Reference

### POST /predict
Predict secondary structure for a protein sequence.

**Request:**
json
{ "sequence": "ACDEFGHIKLMNPQRSTVWY" }


**Response:**
json
{
  "sequence": "ACDEFGHIKLMNPQRSTVWY",
  "structure": "HHHEEECCCHHHEEEHHH CC",
  "per_residue": [
    { "position": 1, "amino_acid": "A", "structure": "H", "structure_name": "Alpha Helix",
      "confidence": { "H": 0.82, "E": 0.10, "C": 0.08 } }
  ],
  "counts": { "H": 10, "E": 6, "C": 4 },
  "percentages": { "H": 50.0, "E": 30.0, "C": 20.0 }
}


**Constraints:** Sequence must be 3–500 standard amino acid characters (ACDEFGHIKLMNPQRSTVWY).

---

### POST /predict_mutation
Predict pathogenicity of a single point mutation.

**Request:**
json
{
  "ref_aa": "A",
  "alt_aa": "V",
  "position": 42,
  "protein_length": 300
}


**Response:**
json
{
  "ref_aa": "A",
  "alt_aa": "V",
  "position": 42,
  "probability": 0.73,
  "prediction": "Pathogenic",
  "confidence": 0.46,
  "risk_level": "High",
  "blosum_score": 0,
  "interpretation": "The A→V substitution at position 42 involves a semi-conservative substitution...",
  "demo_mode": false
}


---

### POST /scan_sequence
Scan all positions in a sequence for the most disruptive mutations.

**Request:**
json
{ "sequence": "ACDEFGHIKLM", "top_n": 5 }


**Response:**
json
{
  "sequence": "ACDEFGHIKLM",
  "top_hits": [ ... ],
  "total_scanned": 209
}


**Constraint:** Maximum sequence length is 200 amino acids.

---

### GET /health
Check model load status.

json
{
  "status": "ok",
  "structure_model_loaded": true,
  "mutation_model_loaded": true
}


---

## Models

### Secondary Structure Predictor
- **Architecture:** Embedding → 2-layer Bidirectional LSTM → Dropout → Linear
- **Input:** Amino acid sequence (padded to 50 residues)
- **Output:** Per-residue class probabilities (H / E / C)
- **Parameters:** ~200K trainable

### Mutation Pathogenicity Predictor
- **Architecture:** Dual embedding (ref + alt AA) → MLP with residual skip connection → Binary logit
- **Input:** Reference AA, alternate AA, normalized position, BLOSUM62 score, charge change
- **Output:** Probability of pathogenicity (0–1)
- **Parameters:** ~50K trainable

---

## Feature Engineering (Mutation Model)

| Feature | Description |
|---|---|
| ref_aa embedding | Learned 32-dim vector for wild-type amino acid |
| alt_aa embedding | Learned 32-dim vector for mutant amino acid |
| Normalized position | position / protein_length |
| BLOSUM62 score | Substitution score normalized by 11 (max value) |
| Charge change | Difference in residue charge, normalized to [-1, 1] |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Flask 3.0+
- NumPy 1.24+
- scikit-learn 1.3+

See requirements.txt for exact versions.

---

## Limitations

- Both models are trained on **synthetic data** by default. Predictions should be treated as illustrative rather than clinically actionable.
- The structure model uses a fixed max length of **50 residues** during training. Longer sequences are padded/truncated accordingly.
- The mutation model does not incorporate evolutionary conservation scores or 3D structural context, which are important features in production-grade pathogenicity predictors (e.g., SIFT, PolyPhen-2).

---

## Future Work

- Integrate real training data (CB513 for structure, ClinVar for mutations)
- Add conservation scoring (e.g., via PSI-BLAST or ESM embeddings)
- Extend structure prediction to Q8 (8-class) secondary structure
- Add protein language model features (ESM-2, ProtTrans)

---

## License

MIT License. See LICENSE for details
