# model.py
# Defines the Bidirectional LSTM model for secondary structure prediction

import torch
import torch.nn as nn

class ProteinStructurePredictor(nn.Module):
    """
    Bidirectional LSTM model for protein secondary structure prediction.

    Architecture:
    - Embedding layer: maps amino acid indices to dense vectors
    - Bidirectional LSTM: captures both forward and backward sequence context
    - Dropout: regularization
    - Fully connected output layer: predicts one of 3 structure classes per position

    Input:
        x: LongTensor of shape (batch_size, seq_len) with amino acid indices
    Output:
        Tensor of shape (batch_size, seq_len, num_classes) with raw class scores
    """

    def __init__(
        self,
        vocab_size=22,        # 20 amino acids + unknown + padding
        embed_dim=64,         # size of embedding vectors
        hidden_dim=128,       # LSTM hidden units per direction
        num_layers=2,         # number of stacked LSTM layers
        num_classes=3,        # H, E, C
        dropout=0.3
    ):
        super(ProteinStructurePredictor, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=vocab_size - 1  # last index used for padding
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        # BiLSTM outputs hidden_dim * 2 features per position
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        lstm_out, _ = self.lstm(embedded)      # (batch, seq_len, hidden*2)
        lstm_out = self.dropout(lstm_out)

        logits = self.fc(lstm_out)             # (batch, seq_len, num_classes)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ProteinStructurePredictor()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    batch = torch.randint(0, 20, (4, 30))  # batch of 4 sequences, length 30
    output = model(batch)
    print(f"Output shape: {output.shape}")  # Expected: (4, 30, 3)
