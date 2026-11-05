"""Expanding walk-forward with a purge gap sized to the label horizon.

The bug this module exists to fix: a model trained on ``horizon``-day forward returns
uses labels that end ``horizon`` days after their row. With a fixed 2-day embargo, the
latest training labels overlap the very returns being predicted. The purge must be at
least ``horizon`` rows (plus an embargo) before the prediction row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import book_return


def make_model(seed: int = 42) -> HistGradientBoostingRegressor:
    """Gradient-boosted regressor; shallower than the research version to keep CI fast."""
    return HistGradientBoostingRegressor(max_iter=80, max_depth=3, learning_rate=0.05,
                                         min_samples_leaf=30, l2_regularization=0.5, random_state=seed)


def purge_rows(horizon: int, embargo: int = 2, leaky: bool = False) -> int:
    """Rows to drop between the last training row and the prediction row.

    ``leaky=True`` reproduces the original research code (embargo only, ignoring horizon).
    """
    return embargo if leaky else horizon + embargo


def walk_forward(prices: pd.DataFrame, table: pd.DataFrame, feats: list[str], horizon: int = 1,
                 start_oos: int = 750, retrain_every: int = 21, embargo: int = 2,
                 leaky: bool = False, seed: int = 42) -> pd.DataFrame:
    """Score every row from ``start_oos`` with a model fitted only on rows that are fully
    realised before it. Returns ``date, raw, ret`` where ``ret`` is always the T+1 book return.
    """
    y = book_return(prices, horizon).values
    X = table[feats].replace([np.inf, -np.inf], np.nan).values
    gap = purge_rows(horizon, embargo, leaky)
    rows = []
    model = None
    for k, i in enumerate(range(start_oos, len(table))):
        if model is None or k % retrain_every == 0:
            end = i - gap  # exclusive; label of row end-1 ends at end-1+horizon
            ok = ~np.isnan(y[:end])
            model = make_model(seed).fit(X[:end][ok], y[:end][ok])
        rows.append((table["date"].iat[i], float(model.predict(X[i:i + 1])[0]), table["ret_t1"].iat[i]))
    return pd.DataFrame(rows, columns=["date", "raw", "ret"])
