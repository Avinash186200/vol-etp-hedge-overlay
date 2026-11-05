"""Daily price data: a CSV loader and a synthetic generator with the same columns.

Columns: ``date, spy, vixy, vix`` (closes of an equity index ETF, a long-VIX-futures
ETP and the VIX index). The real data used in the research is public market data,
but it is not redistributed here; the synthetic generator lets everything run from a clone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["date", "spy", "vixy", "vix"]


def make_synthetic_prices(n_days: int = 2000, seed: int = 7, signal_strength: float = 0.0,
                          start: str = "2015-01-02") -> pd.DataFrame:
    """Generate daily closes with realistic stylised facts.

    - equity returns with a slow stochastic volatility (regimes)
    - VIX tracks that volatility and mean-reverts
    - the vol ETP moves about -2.5x equity, and decays every day (roll cost)

    ``signal_strength`` plants a known effect: tomorrow's equity return is shifted by
    ``signal_strength * z`` where ``z`` is a standardised VIX deviation observable today.
    With ``signal_strength=0`` returns are unpredictable from the past.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    log_vol = np.empty(n_days)
    log_vol[0] = np.log(0.15)
    for t in range(1, n_days):
        log_vol[t] = log_vol[t - 1] + 0.03 * (np.log(0.16) - log_vol[t - 1]) + 0.08 * rng.standard_normal()
    vol = np.exp(log_vol) / np.sqrt(252)
    vix = 100 * np.exp(log_vol) * 1.1 * np.exp(0.05 * rng.standard_normal(n_days))
    # z uses VIX up to and including today; the first 4 rows are back-filled. This only
    # shapes the synthetic world; the model never sees vix_z, only lagged VIX.
    vix_z = (vix - pd.Series(vix).rolling(21, min_periods=5).mean().bfill().values) / 3.0

    eq = np.empty(n_days)
    eq[0] = 0.0
    eq[1:] = 0.0003 + vol[1:] * rng.standard_normal(n_days - 1) + signal_strength * vol[1:] * vix_z[:-1]
    etp = -2.5 * eq - 0.0015 + 0.015 * rng.standard_normal(n_days) * (vol * np.sqrt(252) / 0.16)
    etp[0] = 0.0
    etp = np.maximum(etp, -0.5)
    return pd.DataFrame({
        "date": dates,
        "spy": 100 * np.cumprod(1 + eq),
        "vixy": 100 * np.cumprod(1 + etp),
        "vix": vix,
    })


def load_prices(path: str) -> pd.DataFrame:
    """Load a CSV with columns ``date, spy, vixy, vix`` (daily closes), sorted by date."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df[COLUMNS].dropna().sort_values("date").reset_index(drop=True)
