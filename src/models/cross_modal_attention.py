"""
Explainable Cross-Modal Attention Model for Marine Biodiversity Prediction.

Architecture:
1. Modality-specific encoders (MLP) for oceanographic, fisheries, and eDNA features
2. Cross-modal attention mechanism to learn inter-modality relationships
3. Missing-modality masking with reliability-aware attention
4. Prediction head for future fish species richness

The attention weights are extractable for explainability analysis.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModalityEncoder(nn.Module):
    """MLP encoder for a single data modality."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Each modality attends to all other modalities, learning which
    inter-modality relationships are most informative for prediction.

    For modality i, the attention-enhanced representation is:
        z_i' = z_i + sum_j(alpha_ij * W_v * z_j)

    where alpha_ij = softmax(Q_i * K_j^T / sqrt(d))

    Missing modalities are masked in the attention computation.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._attention_weights = None

    def forward(
        self,
        modality_embeddings: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            modality_embeddings: (batch, n_modalities, embed_dim)
            modality_mask: (batch, n_modalities) - 1 for present, 0 for missing
        Returns:
            attended_embeddings: (batch, n_modalities, embed_dim)
        """
        batch_size, n_mod, _ = modality_embeddings.shape

        Q = self.W_q(modality_embeddings)
        K = self.W_k(modality_embeddings)
        V = self.W_v(modality_embeddings)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_mod, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_mod, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_mod, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply missing-modality mask
        if modality_mask is not None:
            mask = modality_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_mod)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        self._attention_weights = attention_weights.detach()

        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, n_mod, self.embed_dim)
        attended = self.W_o(attended)

        # Residual connection and layer norm
        output = self.layer_norm(modality_embeddings + attended)
        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights for explainability."""
        return self._attention_weights


class CrossModalAttentionModel(nn.Module):
    """
    Full cross-modal attention model for marine biodiversity prediction.

    Architecture:
        Oceanographic features --> OceanEncoder --> z_ocean
        Fisheries features    --> FishEncoder  --> z_fish     --> CrossModalAttention --> Fusion --> Prediction
        eDNA features         --> eDNAEncoder  --> z_edna

    Missing modalities are handled via masking in the attention layer
    and zero-filling in the encoder input.
    """

    def __init__(
        self,
        ocean_input_dim: int,
        fish_input_dim: int,
        edna_input_dim: int,
        encoder_hidden_dim: int = 64,
        encoder_output_dim: int = 64,
        attention_heads: int = 4,
        prediction_hidden_dim: int = 64,
        dropout: float = 0.2,
        n_attention_layers: int = 2,
    ):
        super().__init__()

        self.ocean_encoder = ModalityEncoder(
            ocean_input_dim, encoder_hidden_dim, encoder_output_dim, dropout
        )
        self.fish_encoder = ModalityEncoder(
            fish_input_dim, encoder_hidden_dim, encoder_output_dim, dropout
        )
        self.edna_encoder = ModalityEncoder(
            edna_input_dim, encoder_hidden_dim, encoder_output_dim, dropout
        )

        self.attention_layers = nn.ModuleList([
            CrossModalAttention(encoder_output_dim, attention_heads, dropout)
            for _ in range(n_attention_layers)
        ])

        # Fusion: pool across modalities
        self.fusion = nn.Sequential(
            nn.Linear(encoder_output_dim * 3, prediction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(prediction_hidden_dim, prediction_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(prediction_hidden_dim // 2, 1),
        )

        self._modality_contributions = None

    def forward(
        self,
        ocean_features: torch.Tensor,
        fish_features: torch.Tensor,
        edna_features: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ocean_features: (batch, ocean_dim)
            fish_features: (batch, fish_dim)
            edna_features: (batch, edna_dim)
            modality_mask: (batch, 3) - binary mask for [ocean, fish, edna]
        """
        z_ocean = self.ocean_encoder(ocean_features)
        z_fish = self.fish_encoder(fish_features)
        z_edna = self.edna_encoder(edna_features)

        # Apply modality mask to encoder outputs
        if modality_mask is not None:
            z_ocean = z_ocean * modality_mask[:, 0:1]
            z_fish = z_fish * modality_mask[:, 1:2]
            z_edna = z_edna * modality_mask[:, 2:3]

        # Stack modality embeddings: (batch, 3, embed_dim)
        modality_stack = torch.stack([z_ocean, z_fish, z_edna], dim=1)

        # Apply cross-modal attention layers
        for attn_layer in self.attention_layers:
            modality_stack = attn_layer(modality_stack, modality_mask)

        z_ocean_att = modality_stack[:, 0, :]
        z_fish_att = modality_stack[:, 1, :]
        z_edna_att = modality_stack[:, 2, :]

        # Store modality contributions for explainability
        with torch.no_grad():
            norms = torch.stack([
                z_ocean_att.norm(dim=-1),
                z_fish_att.norm(dim=-1),
                z_edna_att.norm(dim=-1),
            ], dim=1)
            total = norms.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self._modality_contributions = (norms / total).detach()

        # Concatenate and predict
        fused = torch.cat([z_ocean_att, z_fish_att, z_edna_att], dim=-1)
        fused = self.fusion(fused)
        prediction = self.prediction_head(fused).squeeze(-1)

        return prediction

    def get_attention_weights(self) -> list:
        """Return attention weights from all layers."""
        return [layer.get_attention_weights() for layer in self.attention_layers]

    def get_modality_contributions(self) -> Optional[torch.Tensor]:
        """Return modality contribution proportions (batch, 3)."""
        return self._modality_contributions


class CrossModalTrainer:
    """Training pipeline for the cross-modal attention model."""

    def __init__(
        self,
        model: CrossModalAttentionModel,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, factor=0.5
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.best_state = None

    def _prepare_batch(self, X: np.ndarray, feature_groups: dict, y: np.ndarray = None):
        """Split features into modality groups and create tensors."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        ocean_cols = feature_groups.get("oceanographic", [])
        fish_cols = feature_groups.get("fisheries", [])
        edna_cols = feature_groups.get("edna", [])

        # Map column names to indices
        all_cols = list(X_df.columns)

        def get_tensor(cols):
            if not cols:
                return torch.zeros(len(X_df), 1, device=self.device)
            idxs = [all_cols.index(c) for c in cols if c in all_cols]
            if not idxs:
                return torch.zeros(len(X_df), 1, device=self.device)
            return torch.tensor(X_df.iloc[:, idxs].values, dtype=torch.float32, device=self.device)

        ocean_t = get_tensor(ocean_cols)
        fish_t = get_tensor(fish_cols)
        edna_t = get_tensor(edna_cols)

        # Build modality mask from data availability
        mask = torch.ones(len(X_df), 3, device=self.device)
        if ocean_cols:
            ocean_vals = X_df[[c for c in ocean_cols if c in all_cols]]
            mask[:, 0] = torch.tensor(
                (~ocean_vals.isna().all(axis=1)).astype(float).values, device=self.device
            )
        if fish_cols:
            fish_vals = X_df[[c for c in fish_cols if c in all_cols]]
            mask[:, 1] = torch.tensor(
                (~fish_vals.isna().all(axis=1)).astype(float).values, device=self.device
            )
        if edna_cols:
            edna_vals = X_df[[c for c in edna_cols if c in all_cols]]
            mask[:, 2] = torch.tensor(
                (~edna_vals.isna().all(axis=1)).astype(float).values, device=self.device
            )

        y_t = torch.tensor(y, dtype=torch.float32, device=self.device) if y is not None else None
        return ocean_t, fish_t, edna_t, mask, y_t

    def train_epoch(self, X_train, y_train, feature_groups, batch_size=32):
        """Train one epoch."""
        self.model.train()
        n = len(X_train)
        indices = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0

        X_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            X_batch = X_df.iloc[batch_idx]
            y_batch = y_train[batch_idx]

            ocean, fish, edna, mask, y_t = self._prepare_batch(X_batch, feature_groups, y_batch)

            self.optimizer.zero_grad()
            pred = self.model(ocean, fish, edna, mask)
            loss = self.criterion(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, X, y, feature_groups, batch_size=32):
        """Evaluate on a dataset."""
        self.model.eval()
        all_preds = []
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        with torch.no_grad():
            for start in range(0, len(X_df), batch_size):
                end = min(start + batch_size, len(X_df))
                X_batch = X_df.iloc[start:end]
                y_batch = y[start:end]

                ocean, fish, edna, mask, _ = self._prepare_batch(X_batch, feature_groups)
                pred = self.model(ocean, fish, edna, mask)
                all_preds.append(pred.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }
        return metrics, y_pred

    def fit(
        self,
        X_train, y_train,
        X_val, y_val,
        feature_groups: dict,
        epochs: int = 150,
        batch_size: int = 32,
        patience: int = 20,
    ):
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, feature_groups, batch_size)
            val_metrics, _ = self.evaluate(X_val, y_val, feature_groups, batch_size)

            self.scheduler.step(val_metrics["mae"])

            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_r2"].append(val_metrics["r2"])

            if val_metrics["mae"] < best_val_loss:
                best_val_loss = val_metrics["mae"]
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                    f"val_MAE={val_metrics['mae']:.4f}, val_R2={val_metrics['r2']:.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return history


# Import here to avoid circular dependency
import pandas as pd
