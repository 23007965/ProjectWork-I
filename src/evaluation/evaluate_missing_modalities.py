"""
Missing-modality robustness evaluation.

Tests model performance when one or more modalities are unavailable at inference time,
simulating realistic deployment scenarios where eDNA data or fisheries data may be missing.
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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


MISSING_SCENARIOS = {
    "all_present": [1, 1, 1],
    "no_edna": [1, 1, 0],
    "no_fish": [1, 0, 1],
    "no_ocean": [0, 1, 1],
    "only_ocean": [1, 0, 0],
    "only_fish": [0, 1, 0],
    "only_edna": [0, 0, 1],
    "ocean_fish": [1, 1, 0],
    "ocean_edna": [1, 0, 1],
    "fish_edna": [0, 1, 1],
}


def evaluate_with_forced_mask(
    model, X_test, y_test, feature_groups, mask_pattern, batch_size=32,
):
    """Evaluate model with a forced modality mask pattern."""
    from src.models.cross_modal_attention import CrossModalTrainer

    model.eval()
    X_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

    all_preds = []
    device = next(model.parameters()).device

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

            mask = torch.tensor(
                [mask_pattern] * bs, dtype=torch.float32, device=device
            )

            pred = model(ocean_t, fish_t, edna_t, mask)
            all_preds.append(pred.cpu().numpy())

    y_pred = np.concatenate(all_preds)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate missing-modality robustness")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model-path", default="experiments/cross_modal/best_model.pt")
    parser.add_argument("--dataset", default="data/processed/prediction_dataset.csv")
    parser.add_argument("--feature-groups", default="data/processed/features/feature_groups.json")
    parser.add_argument("--output-dir", default="experiments/missing_modality")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.feature_groups) as f:
        feature_groups = json.load(f)

    df = pd.read_csv(args.dataset)
    test = df[df["split"] == "test"]
    X_test = test[feature_groups["all"]].fillna(0)
    y_test = test["future_fish_species_richness"].values

    # Load model
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
        dropout=model_cfg.get("dropout", 0.2),
    )

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.warning(f"Model not found at {args.model_path}, using random weights")

    results = []
    for scenario_name, mask in MISSING_SCENARIOS.items():
        metrics = evaluate_with_forced_mask(
            model, X_test, y_test, feature_groups, mask
        )
        result = {
            "scenario": scenario_name,
            "ocean_present": mask[0],
            "fish_present": mask[1],
            "edna_present": mask[2],
            "n_modalities": sum(mask),
            **metrics,
        }
        results.append(result)
        logger.info(f"[{scenario_name}] MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(args.output_dir) / "missing_modality_results.csv", index=False)

    print("\n=== Missing-Modality Robustness Results ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
