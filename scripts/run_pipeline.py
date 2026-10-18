"""
Main pipeline script: run the complete marine biodiversity prediction workflow.

Usage:
    python scripts/run_pipeline.py --config configs/default_config.yaml [--use-synthetic]
    python scripts/run_pipeline.py --stage download
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage evaluate
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_stage(script: str, args: list = None, description: str = ""):
    """Run a pipeline stage as a subprocess."""
    cmd = [sys.executable, script] + (args or [])
    logger.info(f"=== {description} ===")
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Stage failed: {description}")
        logger.error(result.stderr)
        return False
    if result.stdout:
        logger.info(result.stdout[-2000:])  # Last 2000 chars
    return True


def main():
    parser = argparse.ArgumentParser(description="Run marine biodiversity prediction pipeline")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "download", "align", "features", "train", "evaluate", "explain"],
        default="all",
    )
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    extra = []
    if args.use_synthetic:
        extra.append("--use-synthetic")

    stages = {
        "download": [
            ("src/data/download_edna.py", ["--config", args.config] + extra, "Download eDNA data from OBIS"),
            ("src/data/download_oceanographic.py", ["--config", args.config] + extra, "Download oceanographic data"),
            ("src/data/download_fisheries.py", ["--config", args.config] + extra, "Download fisheries data"),
        ],
        "align": [
            ("src/data/align_spatiotemporal_data.py", ["--config", args.config], "Align spatial-temporal data"),
            ("src/data/build_prediction_target.py", ["--config", args.config], "Build prediction target"),
        ],
        "features": [
            ("src/features/build_features.py", ["--config", args.config], "Build feature sets"),
        ],
        "train": [
            ("src/models/baselines.py", ["--config", args.config, "--n-seeds", str(args.n_seeds)], "Train baseline models"),
        ],
        "evaluate": [
            ("src/evaluation/evaluate_models.py", ["--config", args.config], "Evaluate all models"),
        ],
        "explain": [
            ("src/evaluation/run_ablation_study.py", ["--config", args.config, "--n-seeds", str(args.n_seeds)], "Run ablation study"),
        ],
    }

    if args.stage == "all":
        run_order = ["download", "align", "features", "train", "evaluate", "explain"]
    else:
        run_order = [args.stage]

    for stage_name in run_order:
        if stage_name not in stages:
            continue
        for script, script_args, desc in stages[stage_name]:
            if not os.path.exists(script):
                logger.warning(f"Script not found: {script}, skipping")
                continue
            success = run_stage(script, script_args, desc)
            if not success:
                logger.error(f"Pipeline failed at: {desc}")
                if args.stage == "all":
                    logger.info("Continuing to next stage despite failure...")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
