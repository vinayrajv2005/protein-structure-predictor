# app.py
# Flask web application serving the protein structure prediction API + frontend
# Now includes disease prediction from protein mutation endpoint.

import os
import json
from flask import Flask, request, jsonify, render_template

from predict import load_model, predict_structure
from predict_mutation import predict_mutation, get_mutation_model

app = Flask(__name__)

# ── Model cache ───────────────────────────────────────────────────────────────
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        model_path = "protein_model.pt"
        if not os.path.exists(model_path):
            return None
        MODEL = load_model(model_path)
    return MODEL


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "sequence": "ACDEFGHIKLM..." }
    Returns: secondary structure prediction JSON
    """
    data = request.get_json(silent=True)
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request body"}), 400

    sequence = data["sequence"].strip().upper()

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
        return demo_predict(sequence)

    try:
        result = predict_structure(sequence, model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_mutation", methods=["POST"])
def predict_mutation_route():
    """
    POST /predict_mutation
    Body: {
        "ref_aa"        : "A",       // wild-type amino acid (single letter)
        "alt_aa"        : "V",       // mutant amino acid
        "position"      : 42,        // 1-based position in protein
        "protein_length": 300        // optional, total protein length
    }
    Returns: mutation pathogenicity prediction JSON
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    # Validate required fields
    for field in ("ref_aa", "alt_aa", "position"):
        if field not in data:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    ref_aa = str(data["ref_aa"]).strip().upper()
    alt_aa = str(data["alt_aa"]).strip().upper()

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if ref_aa not in valid_aa:
        return jsonify({"error": f"Invalid reference amino acid: '{ref_aa}'"}), 400
    if alt_aa not in valid_aa:
        return jsonify({"error": f"Invalid alternate amino acid: '{alt_aa}'"}), 400
    if ref_aa == alt_aa:
        return jsonify({"error": "ref_aa and alt_aa must be different"}), 400

    try:
        position = int(data["position"])
        if position < 1:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "position must be a positive integer"}), 400

    protein_length = int(data.get("protein_length", 500))
    protein_length = max(protein_length, position)   # sanity-clamp

    try:
        model = get_mutation_model()
        result = predict_mutation(ref_aa, alt_aa, position, protein_length, model=model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scan_sequence", methods=["POST"])
def scan_sequence():
    """
    POST /scan_sequence
    Scans every position of a sequence for the most disruptive single mutation.
    Body: { "sequence": "ACDEF...", "top_n": 10 }
    Returns: list of top N most pathogenic predicted mutations
    """
    data = request.get_json(silent=True)
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request body"}), 400

    sequence  = data["sequence"].strip().upper()
    top_n     = int(data.get("top_n", 10))
    valid_aa  = set("ACDEFGHIKLMNPQRSTVWY")
    invalid   = [c for c in sequence if c not in valid_aa]
    if invalid:
        return jsonify({"error": f"Invalid characters: {set(invalid)}"}), 400
    if len(sequence) > 200:
        return jsonify({"error": "Sequence too long for scan. Maximum 200 amino acids."}), 400

    model = get_mutation_model()
    results = []

    for i, ref in enumerate(sequence):
        # Try all alternate amino acids for this position
        for alt in valid_aa:
            if alt == ref:
                continue
            r = predict_mutation(ref, alt, i + 1, len(sequence), model=model)
            results.append(r)

    # Sort by probability descending, return top N
    results.sort(key=lambda x: x["probability"], reverse=True)
    return jsonify({
        "sequence":  sequence,
        "top_hits":  results[:top_n],
        "total_scanned": len(results)
    })


def demo_predict(sequence):
    """Returns plausible-looking secondary structure without a trained model."""
    import random
    random.seed(sum(ord(c) for c in sequence))

    weights = [0.35, 0.30, 0.35]
    labels  = ['H', 'E', 'C']
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
    total  = len(structure)
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
    return jsonify({
        "status":                 "ok",
        "structure_model_loaded": MODEL is not None,
        "mutation_model_loaded":  os.path.exists("mutation_model.pt"),
    })


if __name__ == "__main__":
    print("Starting Protein Structure & Mutation Disease Predictor …")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)