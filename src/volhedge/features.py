"""Backward-looking features and forward book-return targets.

Rule: every feature in row ``t`` uses closes up to ``t-1`` only. The position for day
``t`` is set before the close of ``t`` and earns the book return close[t] -> close[t+1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W_EQ, W_ETP = 0.80, 0.20


def book_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Forward ``horizon``-day return of the static 80/20 book, from close[t]."""
    eq = df["spy"].shift(-horizon) / df["spy"] - 1
    etp = df["vixy"].shift(-horizon) / df["vixy"] - 1
    return W_EQ * eq + W_ETP * etp


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(table, feature_names)``. ``table`` also holds ``ret_t1`` (the PnL return)."""
    out = df[["date"]].copy()
    eq_r = df["spy"].pct_change()
    etp_r = df["vixy"].pct_change()
    vix = df["vix"]
    f: dict[str, pd.Series] = {}
    for lag in (1, 2, 5):
        f[f"etp_ret_l{lag}"] = etp_r.shift(lag)
        f[f"eq_ret_l{lag}"] = eq_r.shift(lag)
    for w in (5, 21, 63):
        f[f"rv{w}"] = eq_r.rolling(w).std().shift(1) * np.sqrt(252) * 100
        f[f"etp_mom{w}"] = (df["vixy"] / df["vixy"].rolling(w).mean() - 1).shift(1)
        f[f"eq_mom{w}"] = (df["spy"] / df["spy"].rolling(w).mean() - 1).shift(1)
    vix_p = vix.shift(1)
    f["vix"] = vix_p
    f["vix_chg5"] = vix_p.diff(5)
    f["vix_z21"] = (vix_p - vix_p.rolling(21).mean()) / vix_p.rolling(21).std()
    f["iv_rv21"] = vix_p / f["rv21"].replace(0, np.nan)
    f["iv_rv_spread21"] = vix_p - f["rv21"]
    # expanding (past-only) standardisation, never a full-sample z-score
    f["vix_xz"] = (vix_p - vix_p.expanding(252).mean()) / vix_p.expanding(252).std()
    feats = pd.DataFrame(f)
    names = list(feats.columns)
    out = pd.concat([out, feats], axis=1)
    out["ret_t1"] = book_return(df, 1)
    return out, names
