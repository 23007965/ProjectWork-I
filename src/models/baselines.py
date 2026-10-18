"""
Baseline models for single-modality and conventional multimodal fusion.

Category A: Single-modality (XGBoost, CatBoost, MLP)
Category B: Early fusion (concatenation), Late fusion (averaging/learned weighting)
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import joblib
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_feature_set(feature_dir: str) -> tuple:
    """Load train/val/test splits from a feature directory."""
    d = Path(feature_dir)
    X_train = pd.read_csv(d / "X_train.csv")
    X_val = pd.read_csv(d / "X_val.csv")
    X_test = pd.read_csv(d / "X_test.csv")
    y_train = pd.read_csv(d / "y_train.csv").values.ravel()
    y_val = pd.read_csv(d / "y_val.csv").values.ravel()
    y_test = pd.read_csv(d / "y_test.csv").values.ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def train_xgboost(X_train, y_train, X_val, y_val, config: dict, seed: int = 42):
    """Train an XGBoost regressor."""
    from xgboost import XGBRegressor

    params = config.get("xgboost", {})
    model = XGBRegressor(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_catboost(X_train, y_train, X_val, y_val, config: dict, seed: int = 42):
    """Train a CatBoost regressor."""
    from catboost import CatBoostRegressor

    params = config.get("catboost", {})
    model = CatBoostRegressor(
        iterations=params.get("iterations", 200),
        depth=params.get("depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        random_seed=seed,
        verbose=0,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    return model


def train_mlp(X_train, y_train, X_val, y_val, config: dict, seed: int = 42):
    """Train an MLP regressor."""
    params = config.get("mlp", {})
    model = MLPRegressor(
        hidden_layer_sizes=tuple(params.get("hidden_sizes", [128, 64, 32])),
        max_iter=params.get("epochs", 100),
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate_init=params.get("lr", 0.001),
    )
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_single_modality(
    modality_name: str,
    feature_dir: str,
    config: dict,
    output_dir: str,
    seeds: list = None,
):
    """Train all baseline models for a single modality across multiple seeds."""
    if seeds is None:
        seeds = [42]

    X_train, X_val, X_test, y_train, y_val, y_test = load_feature_set(feature_dir)
    logger.info(f"[{modality_name}] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    model_trainers = {
        "xgboost": train_xgboost,
        "catboost": train_catboost,
        "mlp": train_mlp,
    }

    all_results = []

    for model_name, trainer in model_trainers.items():
        seed_results = []
        for seed in seeds:
            try:
                model = trainer(X_train, y_train, X_val, y_val, config, seed=seed)
                y_pred_test = model.predict(X_test)
                y_pred_val = model.predict(X_val)

                test_metrics = evaluate_predictions(y_test, y_pred_test)
                val_metrics = evaluate_predictions(y_val, y_pred_val)

                result = {
                    "modality": modality_name,
                    "model": model_name,
                    "seed": seed,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                seed_results.append(result)

                # Save model
                model_dir = Path(output_dir) / f"{modality_name}_{model_name}"
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(model, model_dir / f"model_seed{seed}.joblib")

                # Save predictions
                np.save(model_dir / f"y_pred_test_seed{seed}.npy", y_pred_test)

            except Exception as e:
                logger.error(f"[{modality_name}/{model_name}/seed{seed}] Error: {e}")
                continue

        if seed_results:
            # Aggregate across seeds
            test_maes = [r["test_mae"] for r in seed_results]
            test_rmses = [r["test_rmse"] for r in seed_results]
            test_r2s = [r["test_r2"] for r in seed_results]

            summary = {
                "modality": modality_name,
                "model": model_name,
                "test_mae_mean": np.mean(test_maes),
                "test_mae_std": np.std(test_maes),
                "test_rmse_mean": np.mean(test_rmses),
                "test_rmse_std": np.std(test_rmses),
                "test_r2_mean": np.mean(test_r2s),
                "test_r2_std": np.std(test_r2s),
                "n_seeds": len(seed_results),
                "n_test": seed_results[0]["test_n_samples"],
            }
            logger.info(
                f"[{modality_name}/{model_name}] "
                f"MAE={summary['test_mae_mean']:.3f}+/-{summary['test_mae_std']:.3f}, "
                f"R2={summary['test_r2_mean']:.3f}+/-{summary['test_r2_std']:.3f}"
            )

        all_results.extend(seed_results)

    return all_results


class LateFusionModel:
    """Late fusion: combine predictions from separately trained single-modality models."""

    def __init__(self, models: dict, combination: str = "average"):
        self.models = models
        self.combination = combination

    def predict(self, X_per_modality: dict) -> np.ndarray:
        preds = []
        for name, model in self.models.items():
            if name in X_per_modality and X_per_modality[name] is not None:
                p = model.predict(X_per_modality[name])
                preds.append(p)

        if not preds:
            raise ValueError("No modality predictions available")

        if self.combination == "average":
            return np.mean(preds, axis=0)
        elif self.combination == "median":
            return np.median(preds, axis=0)
        else:
            return np.mean(preds, axis=0)


def train_late_fusion(
    feature_base_dir: str,
    config: dict,
    output_dir: str,
    seed: int = 42,
) -> dict:
    """Train late fusion model using best single-modality models."""
    modalities = ["ocean_only", "fish_only", "edna_only"]
    models = {}
    X_tests = {}
    y_test = None

    for mod in modalities:
        fdir = Path(feature_base_dir) / mod
        if not fdir.exists():
            continue
        X_train, X_val, X_test, y_train, y_val, yt = load_feature_set(str(fdir))
        y_test = yt

        model = train_xgboost(X_train, y_train, X_val, y_val, config, seed=seed)
        models[mod] = model
        X_tests[mod] = X_test

    if not models:
        return {}

    fusion = LateFusionModel(models, combination="average")
    y_pred = fusion.predict(X_tests)
    metrics = evaluate_predictions(y_test, y_pred)

    logger.info(
        f"[Late Fusion] MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}"
    )

    return {
        "modality": "late_fusion",
        "model": "xgboost_average",
        "seed": seed,
        **{f"test_{k}": v for k, v in metrics.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--feature-dir", default="data/processed/features")
    parser.add_argument("--output-dir", default="experiments/baselines")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    baseline_config = config["models"]["baselines"]
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = list(range(42, 42 + args.n_seeds))
    all_results = []

    # Single-modality baselines
    for modality in ["ocean_only", "fish_only", "edna_only"]:
        fdir = Path(args.feature_dir) / modality
        if fdir.exists():
            results = train_and_evaluate_single_modality(
                modality, str(fdir), baseline_config, args.output_dir, seeds=seeds,
            )
            all_results.extend(results)

    # Early fusion
    fdir = Path(args.feature_dir) / "early_fusion"
    if fdir.exists():
        results = train_and_evaluate_single_modality(
            "early_fusion", str(fdir), baseline_config, args.output_dir, seeds=seeds,
        )
        all_results.extend(results)

    # Late fusion
    for seed in seeds:
        result = train_late_fusion(args.feature_dir, baseline_config, args.output_dir, seed=seed)
        if result:
            all_results.append(result)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = Path(args.output_dir) / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"All results saved to {results_path}")

    # Print summary
    print("\n=== Baseline Results Summary ===")
    if len(results_df) > 0:
        summary = (
            results_df.groupby(["modality", "model"])
            .agg({"test_mae": ["mean", "std"], "test_rmse": ["mean", "std"], "test_r2": ["mean", "std"]})
            .round(4)
        )
        print(summary)


if __name__ == "__main__":
    main()
