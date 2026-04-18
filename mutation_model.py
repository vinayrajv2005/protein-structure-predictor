# mutation_model.py
# Neural network model for predicting disease likelihood from protein mutations

import torch
import torch.nn as nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)  # 20


class MutationPredictor(nn.Module):
    """
    Predicts whether a point mutation (single amino acid substitution) is
    disease-causing (pathogenic) or benign.

    Inputs per mutation:
        - ref_aa   : integer index of the reference (wild-type) amino acid
        - alt_aa   : integer index of the mutated amino acid
        - position : integer position in the protein (1-based, normalized)
        - features : optional extra float features (conservation score, etc.)

    Architecture:
        Two learned embeddings (ref + alt) → concat + position + features
        → MLP with residual skip → binary output (logit)
    """

    def __init__(
        self,
        num_aa: int = NUM_AA,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_extra_features: int = 3,   # e.g. position, blosum_score, conservation
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_aa + 1, embed_dim)  # +1 for unknown

        in_dim = embed_dim * 2 + num_extra_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output = nn.Linear(hidden_dim // 2, 1)   # binary logit

        # Skip connection (project input to match mlp output dim if needed)
        self.skip = nn.Linear(in_dim, hidden_dim // 2)

    def forward(self, ref_aa, alt_aa, extra_features):
        """
        Args:
            ref_aa          : LongTensor (batch,)
            alt_aa          : LongTensor (batch,)
            extra_features  : FloatTensor (batch, num_extra_features)
        Returns:
            logit           : FloatTensor (batch,)  — positive = pathogenic
        """
        e_ref = self.embedding(ref_aa)            # (B, embed_dim)
        e_alt = self.embedding(alt_aa)            # (B, embed_dim)
        x = torch.cat([e_ref, e_alt, extra_features], dim=-1)   # (B, in_dim)

        hidden = self.mlp(x) + self.skip(x)      # residual
        logit = self.output(hidden).squeeze(-1)   # (B,)
        return logit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MutationPredictor()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Test forward pass
    B = 8
    ref = torch.randint(0, NUM_AA, (B,))
    alt = torch.randint(0, NUM_AA, (B,))
    feats = torch.randn(B, 3)
    out = model(ref, alt, feats)
    print(f"Output shape: {out.shape}")   # (8,)
    print(f"Sample logits: {out.detach().numpy()}")
