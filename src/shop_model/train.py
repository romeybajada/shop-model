"""Train the shop model and record the run in MLflow (Session 6, Stage 14)."""
import argparse
import contextlib
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from shop_model.config import data_path, load_config
from shop_model.data import load_data, training_window
from shop_model.evaluate import score, sha256
from shop_model.features import add_vat
from shop_model.logging_setup import setup_logging
from shop_model.model import FEATURES, save_model, train_model
from shop_model.seed import set_seed

logger = logging.getLogger(__name__)

LAST_RUN = "models/last_run_id"


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def git_state():
    """The commit the code came from, and whether tracked files differ from it. A run from a dirty tree
    cannot be rebuilt from its commit, so it is recorded as such, and shop-registry refuses to register it."""
    try:
        return git("rev-parse", "HEAD"), str(bool(git("status", "--porcelain", "--untracked-files=no"))).lower()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"      # outside a git repository, for example inside an image


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the shop model and track the run in MLflow.")
    parser.add_argument("--train-start", help="override train_start from config.yaml (YYYY-MM-DD, or 'none')")
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    config = load_config()
    if args.train_start:
        config["train_start"] = None if args.train_start.lower() == "none" else args.train_start
    seed = config["seed"]
    set_seed(seed)

    path = data_path(config)
    model_path = os.environ.get("MODEL_PATH", config["model_path"])

    # Where runs are recorded comes from the environment, never from the code (Session 2).
    # MLFLOW_TRACKING_URI=none trains without tracking: only for building an image without a server.
    uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    if uri.lower() == "none":
        logger.warning("Tracking is off: this run is not recorded")
        run_context = contextlib.nullcontext()
    else:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config["mlflow"]["experiment"])
        run_context = mlflow.start_run()

    try:
        with run_context as run:
            data = add_vat(training_window(load_data(path), config), config["vat_rate"])
            model = train_model(data, seed=seed)

            holdout = add_vat(load_data(config["holdout_path"]), config["vat_rate"])
            metrics = score(model, holdout[FEATURES], holdout["sold"])
            passed = metrics[config["gate_metric"]] >= config["gate_threshold"]
            save_model(model, model_path)   # a local copy, for working without the registry
            logger.info("%s %.3f on the holdout, gate %s", config["gate_metric"], metrics[config["gate_metric"]],
                        "passed" if passed else "failed")
            if run is None:
                return

            mlflow.log_params({
                "seed": seed,
                "vat_rate": config["vat_rate"],
                "train_start": config.get("train_start") or "all",
                "holdout_start": config["holdout_start"],
                "window_rows": len(data),
                # train_model holds back a random 20 % (Stage 7), so the model is fitted on the rest.
                "fitted_rows": len(data) - math.ceil(0.2 * len(data)),
            })
            mlflow.log_metrics({f"holdout_{name}": value for name, value in metrics.items()})
            commit, dirty = git_state()
            mlflow.set_tags({
                "git_commit": commit,
                "git_dirty": dirty,
                "data_path": path,
                "data_sha256": sha256(path),
                "holdout_sha256": sha256(config["holdout_path"]),
                "gate_metric": config["gate_metric"],
                "gate_threshold": config["gate_threshold"],
                "gate_passed": str(passed).lower(),
            })
            logged = mlflow.sklearn.log_model(model, name="model", input_example=data[FEATURES].head(3))
            mlflow.set_tag("model_uri", logged.model_uri)
            Path(LAST_RUN).write_text(run.info.run_id)      # so `shop-registry register` registers this run
            logger.info("Run %s recorded in MLflow", run.info.run_id)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
