"""QuantPredictor — loads a pretrained Qlib model and provides per-stock predictions.

All qlib imports are lazy (inside methods) so that stockbot.quant can be
imported without qlib installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class QuantPredictor:
    """Loads a pretrained LightGBM model and runs batch prediction on CSI300.

    Results are cached in-memory so that per-stock lookups are cheap.
    """

    def __init__(self, data_dir: str, model_dir: str, instruments: str = "csi300"):
        import qlib
        from qlib.config import REG_CN

        data_path = Path(data_dir).expanduser()
        model_path = Path(model_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Qlib data dir not found: {data_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Qlib model dir not found: {model_path}")

        qlib.init(provider_uri=str(data_path), region=REG_CN)

        self._model = self._load_model(model_path)
        self._instruments = instruments
        self._cache: dict[str, dict] = {}
        self._refreshed_at: str = ""

        self._batch_predict()

    # ── public API ──────────────────────────────────────────────

    @staticmethod
    def is_available(model_dir: str) -> bool:
        """True if qlib is installed AND a trained model exists."""
        try:
            import qlib  # noqa: F401
            return Path(model_dir).exists()
        except ImportError:
            return False

    def predict(self, symbol: str) -> dict:
        """Return quant prediction for *symbol* (6-digit code like '600519').

        Returns a dict with keys: symbol, score, rank_pct, total,
        refreshed_at, signal.  If *symbol* is not in the prediction
        universe the dict contains a single ``error`` key.
        """
        if symbol not in self._cache:
            return {"error": f"{symbol} 不在量化预测范围内"
                             f"（当前覆盖 {self._instruments.upper()} 成分股）"}
        entry = self._cache[symbol]
        return {
            "symbol": symbol,
            "score": entry["score"],
            "rank_pct": entry["rank_pct"],
            "total": self._total,
            "refreshed_at": self._refreshed_at,
            "signal": self._to_signal(entry["score"]),
        }

    # ── internals ───────────────────────────────────────────────

    def _to_q_symbol(self, symbol: str) -> str:
        """Convert 600519 → SH600519."""
        if symbol.startswith(("6", "68", "9")):
            return f"SH{symbol}"
        return f"SZ{symbol}"

    def _from_q_symbol(self, q_symbol: str) -> str:
        """Convert SH600519 → 600519."""
        return q_symbol[2:]

    def _load_model(self, model_path: Path):
        """Load a joblib-serialised model."""
        import joblib

        model_file = model_path / "lgb_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        LOGGER.info("Loading Qlib model from %s", model_file)
        return joblib.load(model_file)

    def _batch_predict(self):
        from qlib.data.dataset import DatasetH
        from qlib.contrib.data.handler import Alpha158
        from datetime import datetime

        import pandas as pd

        now = datetime.now()
        train_end = now.strftime("%Y-%m-%d")
        # Start 90 days back to have enough lookback for 60-day features
        train_start = (now - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        predict_start = (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        dataset = DatasetH(
            handler={
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": train_start,
                    "end_time": train_end,
                    "fit_start_time": train_start,
                    "fit_end_time": train_end,
                    "instruments": self._instruments,
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"clip_outlier": True}},
                        {"class": "Fillna"},
                    ],
                },
            },
            segments={
                "predict": (predict_start, train_end),
            },
        )

        pred = self._model.predict(dataset, segment="predict")
        if pred.empty:
            LOGGER.warning("Qlib batch prediction returned empty result")
            self._total = 0
            self._refreshed_at = now.strftime("%Y-%m-%d")
            return

        scores = pred.groupby("instrument").last()
        all_scores = scores.iloc[:, 0].sort_values(ascending=False)

        self._total = len(all_scores)
        self._refreshed_at = now.strftime("%Y-%m-%d")

        self._cache = {}
        for q_sym, score in all_scores.items():
            sym = self._from_q_symbol(q_sym)
            rank = all_scores.index.get_loc(q_sym) + 1
            self._cache[sym] = {
                "score": float(score),
                "rank_pct": round(rank / self._total * 100, 1),
            }
        LOGGER.info("Qlib batch prediction: %d stocks cached", self._total)

    @staticmethod
    def _to_signal(score: float) -> str:
        if score > 0.03:
            return "🟢 强烈看多"
        elif score > 0.01:
            return "🟡 温和看多"
        elif score > -0.01:
            return "⚪ 中性"
        elif score > -0.03:
            return "🟠 温和看空"
        else:
            return "🔴 强烈看空"
