#!/usr/bin/env python
"""Train LightGBM classifiers for index direction prediction.

Usage:
    python scripts/setup_index_model.py               # full setup
    python scripts/setup_index_model.py --index 000001  # specify index

Trains two classifiers:
- Short-term: predict whether index goes up in next 3 days
- Mid-term:   predict whether index goes up in next 20 days

Saves the trained model to data/index_model/index_model.pkl.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "index_model"


def fetch_index_data(index_code: str = "000001") -> pd.DataFrame:
    """Fetch historical index OHLCV data via akshare."""
    import akshare as ak

    LOGGER.info("Fetching index %s data ...", index_code)
    df = ak.stock_zh_index_daily_em(symbol=f"sh{index_code}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def build_features(df: pd.DataFrame, short_horizon: int = 3,
                   mid_horizon: int = 20) -> pd.DataFrame:
    """Construct features and labels for both horizons."""
    df = df.copy()

    closes = df["close"].values
    volumes = df["volume"].values
    highs = df["high"].values
    lows = df["low"].values

    n = len(df)
    features = []

    for i in range(n):
        if i < 30:
            continue

        slice_c = closes[:i + 1]
        slice_v = volumes[:i + 1]
        k = len(slice_c)

        ret5 = (slice_c[-1] / slice_c[-6] - 1) if k >= 6 else 0
        ret10 = (slice_c[-1] / slice_c[-11] - 1) if k >= 11 else 0
        ret20 = (slice_c[-1] / slice_c[-21] - 1) if k >= 21 else 0

        # 20-day volatility
        rets = [(slice_c[t] / slice_c[t-1] - 1) for t in range(max(1, k-20), k)]
        vol20 = float(np.std(rets)) if len(rets) > 1 else 0

        # Volume ratio
        vol_ma5 = float(np.mean(slice_v[-5:])) if k >= 5 else slice_v[-1]
        vol_ma10 = float(np.mean(slice_v[-10:])) if k >= 10 else slice_v[-1]
        vol_ratio = vol_ma5 / vol_ma10 if vol_ma10 > 0 else 1.0

        # MA positions
        ma5 = float(np.mean(slice_c[-5:])) if k >= 5 else slice_c[-1]
        ma10 = float(np.mean(slice_c[-10:])) if k >= 10 else slice_c[-1]
        ma20 = float(np.mean(slice_c[-20:])) if k >= 20 else slice_c[-1]
        price_vs_ma20 = slice_c[-1] / ma20 - 1 if ma20 > 0 else 0
        ma5_vs_ma20 = ma5 / ma20 - 1 if ma20 > 0 else 0

        # Simple RSI
        if k >= 15:
            gains = sum(
                max(slice_c[t] - slice_c[t-1], 0) for t in range(k-14, k)
            )
            losses = sum(
                max(slice_c[t-1] - slice_c[t], 0) for t in range(k-14, k)
            )
            rsi = 100 * gains / (gains + losses) if (gains + losses) > 0 else 50
        else:
            rsi = 50

        # Range
        high_low_range = (highs[i] - lows[i]) / slice_c[-1] if slice_c[-1] > 0 else 0

        # Labels
        short_up = 0
        mid_up = 0
        if i + short_horizon < n:
            short_up = 1 if closes[i + short_horizon] > closes[i] else 0
        if i + mid_horizon < n:
            mid_up = 1 if closes[i + mid_horizon] > closes[i] else 0

        features.append([
            ret5, ret10, ret20, vol20, vol_ratio,
            price_vs_ma20, ma5_vs_ma20, rsi / 100,
            high_low_range, short_up, mid_up,
        ])

    cols = [
        "ret5", "ret10", "ret20", "vol20", "vol_ratio",
        "price_vs_ma20", "ma5_vs_ma20", "rsi_norm",
        "high_low_range", "short_label", "mid_label",
    ]
    return pd.DataFrame(features, columns=cols)


def train_models(df: pd.DataFrame) -> dict:
    """Train short-term and mid-term LGBM classifiers."""
    feature_cols = [
        "ret5", "ret10", "ret20", "vol20", "vol_ratio",
        "price_vs_ma20", "ma5_vs_ma20", "rsi_norm", "high_low_range",
    ]

    X = df[feature_cols].values
    y_short = df["short_label"].values
    y_mid = df["mid_label"].values

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_short, test_size=0.2, shuffle=False,
    )
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_mid, test_size=0.2, shuffle=False,
    )

    LOGGER.info("Training short-term model (horizon=3 days) ...")
    short_model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1,
    )
    short_model.fit(X_train_s, y_train_s)
    short_acc = short_model.score(X_test_s, y_test_s)
    LOGGER.info("  Short-term accuracy: %.2f%%", short_acc * 100)

    LOGGER.info("Training mid-term model (horizon=20 days) ...")
    mid_model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1,
    )
    mid_model.fit(X_train_m, y_train_m)
    mid_acc = mid_model.score(X_test_m, y_test_m)
    LOGGER.info("  Mid-term accuracy: %.2f%%", mid_acc * 100)

    return {
        "short_model": short_model,
        "mid_model": mid_model,
        "feature_names": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train index prediction models for StockBot",
    )
    parser.add_argument("--index", default="000001",
                        help="Index code (default: 000001 = 上证指数)")
    parser.add_argument("--short-horizon", type=int, default=3)
    parser.add_argument("--mid-horizon", type=int, default=20)
    args = parser.parse_args()

    df = fetch_index_data(args.index)
    LOGGER.info("Fetched %d rows for index %s", len(df), args.index)

    features = build_features(df, args.short_horizon, args.mid_horizon)
    LOGGER.info("Built %d feature vectors", len(features))

    models = train_models(features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "index_model.pkl"
    joblib.dump(models, model_path)
    LOGGER.info(
        "Model saved → %s (%.1f MB)",
        model_path, model_path.stat().st_size / 1024 / 1024,
    )
    LOGGER.info(
        "Setup complete! Index prediction ML model is ready. "
        "Restart StockBot to use it."
    )


if __name__ == "__main__":
    main()
