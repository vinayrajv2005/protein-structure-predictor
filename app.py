# app.py
# Flask web application serving the protein structure prediction API + frontend

import os
import json
from flask import Flask, request, jsonify, render_template

from predict import load_model, predict_structure

app = Flask(__name__)

# Load model once at startup
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        model_path = "protein_model.pt"
        if not os.path.exists(model_path):
            # Return None — frontend will show "model not trained" message
            return None
        MODEL = load_model(model_path)
    return MODEL


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "sequence": "ACDEFGHIKLM..." }
    Returns: prediction result JSON
    """
    data = request.get_json(silent=True)
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request body"}), 400

    sequence = data["sequence"].strip().upper()

    # Validate
    if len(sequence) < 3:
        return jsonify({"error": "Sequence too short. Minimum 3 amino acids."}), 400
    if len(sequence) > 500:
        return jsonify({"error": "Sequence too long. Maximum 500 amino acids."}), 400

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = [c for c in sequence if c not in valid_aa]
    if invalid:
        return jsonify({
            "error": f"Invalid characters in sequence: {set(invalid)}. "
                     f"Use standard single-letter amino acid codes."
        }), 400

    model = get_model()
    if model is None:
        # Demo mode: return mock data so UI is usable without a trained model
        return demo_predict(sequence)

    try:
        result = predict_structure(sequence, model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def demo_predict(sequence):
    """
    Returns a plausible-looking prediction without a trained model.
    Useful for UI demos and portfolio demonstrations.
    """
    import random
    random.seed(sum(ord(c) for c in sequence))

    weights = [0.35, 0.30, 0.35]  # H, E, C
    labels = ['H', 'E', 'C']
    label_names = {'H': 'Alpha Helix', 'E': 'Beta Sheet', 'C': 'Coil/Loop'}

    structure = ''.join(random.choices(labels, weights=weights, k=len(sequence)))
    per_residue = []
    for i, (aa, s) in enumerate(zip(sequence, structure)):
        idx = labels.index(s)
        conf = [round(random.uniform(0.05, 0.25), 4) for _ in range(3)]
        conf[idx] = round(1.0 - sum(conf) + conf[idx], 4)
        per_residue.append({
            "position": i + 1,
            "amino_acid": aa,
            "structure": s,
            "structure_name": label_names[s],
            "confidence": dict(zip(['H', 'E', 'C'], conf))
        })

    counts = {l: structure.count(l) for l in labels}
    total = len(structure)
    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}

    return jsonify({
        "sequence":    sequence,
        "structure":   structure,
        "per_residue": per_residue,
        "counts":      counts,
        "percentages": percentages,
        "demo_mode":   True
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    print("Starting Protein Structure Predictor...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
