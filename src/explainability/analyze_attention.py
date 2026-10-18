"""
Cross-modal attention analysis for explainability.

Extracts and visualizes attention patterns from the cross-modal attention model
to identify which modality interactions the model finds most informative.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODALITY_NAMES = ["Oceanographic", "Fisheries", "eDNA"]


def extract_attention_patterns(
    model, X_test, feature_groups, batch_size=32
):
    """Extract attention weights for all test samples."""
    model.eval()
    device = next(model.parameters()).device
    X_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

    all_attention_weights = []
    all_modality_contributions = []

    with torch.no_grad():
        for start in range(0, len(X_df), batch_size):
            end = min(start + batch_size, len(X_df))
            batch = X_df.iloc[start:end]
            bs = len(batch)

            ocean_cols = feature_groups.get("oceanographic", [])
            fish_cols = feature_groups.get("fisheries", [])
            edna_cols = feature_groups.get("edna", [])

            def get_tensor(cols):
                all_cols = list(batch.columns)
                if not cols:
                    return torch.zeros(bs, 1, device=device)
                idxs = [all_cols.index(c) for c in cols if c in all_cols]
                if not idxs:
                    return torch.zeros(bs, 1, device=device)
                return torch.tensor(batch.iloc[:, idxs].values, dtype=torch.float32, device=device)

            ocean_t = get_tensor(ocean_cols)
            fish_t = get_tensor(fish_cols)
            edna_t = get_tensor(edna_cols)
            mask = torch.ones(bs, 3, device=device)

            _ = model(ocean_t, fish_t, edna_t, mask)

            # Extract attention weights from all layers
            attn_weights = model.get_attention_weights()
            for layer_idx, aw in enumerate(attn_weights):
                if aw is not None:
                    all_attention_weights.append({
                        "layer": layer_idx,
                        "weights": aw.cpu().numpy(),
                    })

            contributions = model.get_modality_contributions()
            if contributions is not None:
                all_modality_contributions.append(contributions.cpu().numpy())

    return all_attention_weights, all_modality_contributions


def summarize_attention(attention_data: list) -> dict:
    """Summarize attention patterns across all samples."""
    summaries = {}

    for layer_idx in set(d["layer"] for d in attention_data):
        layer_weights = [d["weights"] for d in attention_data if d["layer"] == layer_idx]
        if not layer_weights:
            continue

        # Concatenate across batches, average across heads
        all_weights = np.concatenate(layer_weights, axis=0)  # (n_samples, n_heads, 3, 3)
        avg_across_heads = all_weights.mean(axis=1)  # (n_samples, 3, 3)
        avg_across_samples = avg_across_heads.mean(axis=0)  # (3, 3)
        std_across_samples = avg_across_heads.std(axis=0)

        attn_df = pd.DataFrame(
            avg_across_samples,
            index=MODALITY_NAMES,
            columns=MODALITY_NAMES,
        )

        summaries[f"layer_{layer_idx}"] = {
            "mean": attn_df,
            "std": pd.DataFrame(std_across_samples, index=MODALITY_NAMES, columns=MODALITY_NAMES),
            "n_samples": all_weights.shape[0],
        }

        logger.info(f"\nAttention Layer {layer_idx} (mean across {all_weights.shape[0]} samples):")
        logger.info(f"\n{attn_df.round(3)}")

    return summaries


def summarize_modality_contributions(contributions: list) -> pd.DataFrame:
    """Summarize modality contribution proportions."""
    all_contribs = np.concatenate(contributions, axis=0)  # (n_samples, 3)
    summary = pd.DataFrame({
        "modality": MODALITY_NAMES,
        "mean_contribution": all_contribs.mean(axis=0),
        "std_contribution": all_contribs.std(axis=0),
        "median_contribution": np.median(all_contribs, axis=0),
    })

    logger.info(f"\nModality Contributions:\n{summary.to_string(index=False)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-modal attention")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model-path", default="experiments/cross_modal/best_model.pt")
    parser.add_argument("--dataset", default="data/processed/prediction_dataset.csv")
    parser.add_argument("--feature-groups", default="data/processed/features/feature_groups.json")
    parser.add_argument("--output-dir", default="reports/explainability")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.feature_groups) as f:
        feature_groups = json.load(f)

    df = pd.read_csv(args.dataset)
    test = df[df["split"] == "test"]
    X_test = test[feature_groups["all"]].fillna(0)

    from src.models.cross_modal_attention import CrossModalAttentionModel

    model_cfg = config["models"]["cross_modal_attention"]
    ocean_dim = max(len(feature_groups.get("oceanographic", [])), 1)
    fish_dim = max(len(feature_groups.get("fisheries", [])), 1)
    edna_dim = max(len(feature_groups.get("edna", [])), 1)

    model = CrossModalAttentionModel(
        ocean_input_dim=ocean_dim,
        fish_input_dim=fish_dim,
        edna_input_dim=edna_dim,
        encoder_hidden_dim=model_cfg.get("encoder_dim", 64),
        encoder_output_dim=model_cfg.get("attention_dim", 64),
        attention_heads=model_cfg.get("attention_heads", 4),
        dropout=0.0,  # No dropout during inference
    )

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    attn_data, contrib_data = extract_attention_patterns(model, X_test, feature_groups)

    attn_summaries = summarize_attention(attn_data)
    for layer_name, summary in attn_summaries.items():
        summary["mean"].to_csv(Path(args.output_dir) / f"attention_{layer_name}_mean.csv")
        summary["std"].to_csv(Path(args.output_dir) / f"attention_{layer_name}_std.csv")

    if contrib_data:
        contrib_summary = summarize_modality_contributions(contrib_data)
        contrib_summary.to_csv(Path(args.output_dir) / "modality_contributions.csv", index=False)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
