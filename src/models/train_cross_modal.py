"""
Train the cross-modal attention model and evaluate on test set.
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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_modal_attention import CrossModalAttentionModel, CrossModalTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train cross-modal attention model")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--dataset", default="data/processed/prediction_dataset.csv")
    parser.add_argument("--feature-groups", default="data/processed/features/feature_groups.json")
    parser.add_argument("--output-dir", default="experiments/cross_modal")
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

    model_cfg = config["models"]["cross_modal_attention"]
    ocean_dim = max(len(feature_groups.get("oceanographic", [])), 1)
    fish_dim = max(len(feature_groups.get("fisheries", [])), 1)
    edna_dim = max(len(feature_groups.get("edna", [])), 1)

    seeds = list(range(42, 42 + args.n_seeds))
    all_results = []

    for seed in seeds:
        logger.info(f"\n=== Training with seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

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

        history = trainer.fit(
            X_train, y_train, X_val, y_val,
            feature_groups=feature_groups,
            epochs=model_cfg.get("epochs", 150),
            batch_size=model_cfg.get("batch_size", 32),
        )

        test_metrics, y_pred = trainer.evaluate(X_test, y_test, feature_groups)
        val_metrics, _ = trainer.evaluate(X_val, y_val, feature_groups)

        result = {
            "modality": "cross_modal_attention",
            "model": "cross_modal_attention",
            "seed": seed,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "test_n_samples": len(y_test),
        }
        all_results.append(result)

        logger.info(
            f"Seed {seed}: test MAE={test_metrics['mae']:.4f}, "
            f"RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}"
        )

        # Save best model (first seed)
        if seed == seeds[0]:
            model_path = Path(args.output_dir) / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            np.save(Path(args.output_dir) / "y_pred_test.npy", y_pred)
            np.save(Path(args.output_dir) / "y_test.npy", y_test)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Path(args.output_dir) / "cross_modal_results.csv", index=False)

    # Summary
    print("\n=== Cross-Modal Attention Results ===")
    for metric in ["test_mae", "test_rmse", "test_r2"]:
        vals = results_df[metric]
        print(f"{metric}: {vals.mean():.4f} +/- {vals.std():.4f}")


if __name__ == "__main__":
    main()
