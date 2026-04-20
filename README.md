## 🚀 Project Summary

An AI-powered protein analysis system that predicts:

- 🧬 **Secondary Structure** (Helix, Sheet, Coil) using BiLSTM
- ⚠️ **Mutation Impact** (Pathogenic vs Benign) using an MLP model

Built with **Flask + PyTorch**, this project simulates real-world bioinformatics tools used in disease research and drug discovery.

🔍 Includes:
- Per-residue structure prediction
- Mutation risk scoring with biological features (BLOSUM62, charge change)
- Sequence-wide mutation scanning
- REST API for integration

📊 Designed as an end-to-end ML system combining sequence modeling and biological feature engineering.

> ⚠️ Note: Models are trained on synthetic data for demonstration purposes.
