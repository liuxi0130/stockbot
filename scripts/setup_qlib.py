#!/usr/bin/env python
"""One-time setup: download Qlib CSI300 data and train a LightGBM model.

Usage:
    python scripts/setup_qlib.py          # full setup (download + train)
    python scripts/setup_qlib.py --train-only   # skip download, train only
    python scripts/setup_qlib.py --download-only # skip training, download only

The download step uses Qlib's built-in data fetcher (~30-60 min).
The training step uses Alpha158 factors + LightGBM (~10-20 min).
The trained model is saved to data/qlib_model/.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"
MODEL_DIR = PROJECT_ROOT / "data" / "qlib_model"


def download_data():
    """Download CSI300 daily data via Qlib's built-in command."""
    LOGGER.info("Downloading Qlib CN data → %s", DATA_DIR)
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "qlib.run.get_data", "qlib_data",
            "--target_dir", str(DATA_DIR.parent),
            "--region", "cn",
            "--interval", "1d",
            "--delete_old", "False",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        LOGGER.error("Data download failed (exit %d)", result.returncode)
        sys.exit(1)
    LOGGER.info("Download complete → %s", DATA_DIR)


def train_model():
    """Train LightGBM on Alpha158 factors and save the model."""
    import qlib
    from qlib.config import REG_CN
    from qlib.data.dataset import DatasetH
    from qlib.contrib.model.gbdt import LGBModel

    if not DATA_DIR.exists():
        LOGGER.error("Qlib data not found at %s — run with --download-only first", DATA_DIR)
        sys.exit(1)

    LOGGER.info("Initializing Qlib with data at %s", DATA_DIR)
    qlib.init(provider_uri=str(DATA_DIR), region=REG_CN)

    LOGGER.info("Building Alpha158 dataset for CSI300 ...")
    dataset = DatasetH(
        handler={
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2024-12-31",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2021-12-31",
                "instruments": "csi300",
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"clip_outlier": True}},
                    {"class": "Fillna"},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                ],
            },
        },
        segments={
            "train": ("2008-01-01", "2019-12-31"),
            "valid": ("2020-01-01", "2021-12-31"),
            "test": ("2022-01-01", "2024-12-31"),
        },
    )

    LOGGER.info("Training LightGBM model ...")
    model = LGBModel(
        loss="mse",
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        n_estimators=500,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model.fit(dataset)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = MODEL_DIR / "lgb_model.pkl"
    joblib.dump(model, model_path)
    LOGGER.info("Model saved → %s (%.1f MB)", model_path,
                model_path.stat().st_size / 1024 / 1024)


def main():
    parser = argparse.ArgumentParser(description="Qlib setup for StockBot")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    if args.train_only:
        train_model()
    elif args.download_only:
        download_data()
    else:
        download_data()
        train_model()
        LOGGER.info("Setup complete — quant prediction is ready!")


if __name__ == "__main__":
    main()
