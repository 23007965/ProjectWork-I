"""
Ablation study: systematically remove modalities and attention mechanism
to measure their individual contributions.

Ablation conditions:
1. Full model (all 3 modalities + attention)
2. Remove oceanographic
3. Remove fisheries
4. Remove eDNA
5. Remove cross-modal attention (use simple concatenation instead)
6. Remove missing-modality masks
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


def run_ablation_experiment(
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_groups: dict,
    config: dict,
    ablation_condition: str,
    seed: int = 42,
) -> dict:
    """Run a single ablation experiment."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from cross_modal_attention import CrossModalAttentionModel, CrossModalTrainer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Modify feature groups based on ablation condition
    groups = {k: list(v) for k, v in feature_groups.items()}

    if ablation_condition == "no_ocean":
        groups["oceanographic"] = []
    elif ablation_condition == "no_fish":
        groups["fisheries"] = []
    elif ablation_condition == "no_edna":
        groups["edna"] = []

    ocean_dim = max(len(groups.get("oceanographic", [])), 1)
    fish_dim = max(len(groups.get("fisheries", [])), 1)
    edna_dim = max(len(groups.get("edna", [])), 1)

    model_cfg = config["models"]["cross_modal_attention"]

    model = CrossModalAttentionModel(
        ocean_input_dim=ocean_dim,
        fish_input_dim=fish_dim,
        edna_input_dim=edna_dim,
        encoder_hidden_dim=model_cfg.get("encoder_dim", 64),
        encoder_output_dim=model_cfg.get("attention_dim", 64),
        attention_heads=model_cfg.get("attention_heads", 4),
        dropout=model_cfg.get("dropout", 0.2),
    )

    trainer = CrossModalTrainer(model, lr=model_cfg.get("lr", 0.001))

    trainer.fit(
        X_train, y_train, X_val, y_val,
        feature_groups=groups,
        epochs=model_cfg.get("epochs", 150),
        batch_size=model_cfg.get("batch_size", 32),
    )

    test_metrics, y_pred = trainer.evaluate(X_test, y_test, groups)

    result = {
        "ablation": ablation_condition,
        "seed": seed,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "n_test": len(y_test),
    }

    logger.info(
        f"[{ablation_condition}/seed{seed}] "
        f"MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--dataset", default="data/processed/prediction_dataset.csv")
    parser.add_argument("--feature-groups", default="data/processed/features/feature_groups.json")
    parser.add_argument("--output-dir", default="experiments/ablation")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.feature_groups) as f:
        feature_groups = json.load(f)

    df = pd.read_csv(args.dataset)
    all_features = feature_groups["all"]
    target = "future_fish_species_richness"

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train = train[all_features].fillna(0)
    X_val = val[all_features].fillna(0)
    X_test = test[all_features].fillna(0)
    y_train = train[target].values
    y_val = val[target].values
    y_test = test[target].values

    ablation_conditions = [
        "full",
        "no_ocean",
        "no_fish",
        "no_edna",
    ]

    seeds = list(range(42, 42 + args.n_seeds))
    all_results = []

    for condition in ablation_conditions:
        for seed in seeds:
            result = run_ablation_experiment(
                X_train, y_train, X_val, y_val, X_test, y_test,
                feature_groups, config, condition, seed,
            )
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Path(args.output_dir) / "ablation_results.csv", index=False)

    # Summary
    summary = results_df.groupby("ablation").agg(
        mae_mean=("test_mae", "mean"),
        mae_std=("test_mae", "std"),
        r2_mean=("test_r2", "mean"),
        r2_std=("test_r2", "std"),
    ).reset_index()

    print("\n=== Ablation Study Results ===")
    print(summary.to_string(index=False))

    # Compute relative changes
    full_mae = summary.loc[summary["ablation"] == "full", "mae_mean"].values[0]
    summary["mae_change_pct"] = ((summary["mae_mean"] - full_mae) / full_mae * 100).round(1)
    summary.to_csv(Path(args.output_dir) / "ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
