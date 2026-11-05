"""Score -> position -> PnL, plus statistics and a permutation test."""
from __future__ import annotations

import numpy as np
import pandas as pd


def positions(raw: np.ndarray, span: int = 7, scale: float = 1.0, clip: float = 1.0,
              min_history: int = 250) -> np.ndarray:
    """EMA-smooth the score, rank it against strictly past smoothed scores, map to a weight.

    ``w = clip(scale * (2 * pct - 1), -clip, clip)``; 0 until ``min_history`` past scores exist.
    ``scale`` is fixed in advance (1.0); it is not tuned on the evaluation window.
    """
    sm = pd.Series(raw).ewm(span=span, adjust=False).mean().values
    w = np.zeros(len(sm))
    for t in range(min_history, len(sm)):
        pct = (sm[:t] < sm[t]).mean()
        w[t] = np.clip(scale * (2 * pct - 1), -clip, clip)
    return w


def pnl(w: np.ndarray, ret: np.ndarray, cost: float = 2e-4) -> np.ndarray:
    """Overlay PnL: ``w * ret`` minus ``cost`` per unit of turnover."""
    prev = np.r_[0.0, w[:-1]]
    return w * ret - np.abs(w - prev) * cost


def stats(x: np.ndarray) -> dict[str, float]:
    """Annualised Sharpe, total return and max drawdown of a daily return series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    eq = np.cumprod(1 + x)
    sharpe = float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0
    return {"sharpe": round(sharpe, 3), "total_pct": round(float(eq[-1] - 1) * 100, 1),
            "max_dd_pct": round(float((eq / np.maximum.accumulate(eq) - 1).min()) * 100, 1),
            "n_days": int(len(x))}


def block_permutation_p(w: np.ndarray, ret: np.ndarray, n: int = 200, block: int = 21,
                        seed: int = 0, cost: float = 2e-4) -> float:
    """p-value of the overlay Sharpe vs positions shuffled in blocks (keeps their autocorrelation)."""
    rng = np.random.default_rng(seed)
    real = stats(pnl(w, ret, cost))["sharpe"]
    blocks = [w[i:i + block] for i in range(0, len(w), block)]
    hits = 0
    for _ in range(n):
        order = rng.permutation(len(blocks))
        wp = np.concatenate([blocks[j] for j in order])
        hits += stats(pnl(wp, ret, cost))["sharpe"] >= real
    return (hits + 1) / (n + 1)
