"""Two label-overlap diagnostics that work on scores alone.

A feature-timestamp audit cannot see label overlap: no feature looks ahead, but the
fitted model has been trained on labels that contain the return it is asked to
predict. These two checks look for the fingerprint that overlap leaves in the
out-of-sample scores, so they can be run on any retrained model's score history.

1. **IC by lead.** Correlate the score on day t with the book return k days ahead.
   An honest forecaster's IC changes smoothly with k. A model trained on overlapping
   h-day labels shows IC that falls to zero exactly where the training labels stop
   containing the k-th day's return.
2. **Refit-cycle fingerprint.** A model refitted every R days is reused on days
   c = 0 .. R-1 of each cycle. With a purge gap g, the number of training labels that
   contain the predicted return is ``max(h - g - 1 - c, 0)``, which falls through the
   cycle. Real skill has no reason to follow the refit calendar; overlap does. So the
   IC is compared between cycle positions with overlap and positions without.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def label_overlap(horizon: int, cycle_pos: int, gap: int) -> int:
    """Training rows whose ``horizon``-day label contains the return predicted on cycle day ``cycle_pos``.

    On a refit at row i the last training row is ``i - gap - 1``; its label spans
    ``close[r] -> close[r + horizon]``. The return predicted on day ``i + c`` is
    ``close[i + c] -> close[i + c + 1]``.
    """
    return max(horizon - gap - 1 - cycle_pos, 0)


def _ic(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 20:
        return float("nan"), int(ok.sum())
    return float(spearmanr(a[ok], b[ok])[0]), int(ok.sum())


def ic_by_lead(raw: np.ndarray, ret: np.ndarray, max_lead: int = 25) -> pd.DataFrame:
    """Rank IC of the score on day t with the T+k daily return, k = 1 .. ``max_lead``.

    ``ret`` is the T+1 book return aligned to the score (``walk_forward`` output).
    """
    raw = np.asarray(raw, dtype=float)
    ret = pd.Series(np.asarray(ret, dtype=float))
    rows = []
    for k in range(1, max_lead + 1):
        ic, n = _ic(raw, ret.shift(-(k - 1)).to_numpy())
        rows.append({"lead": k, "ic": ic, "n": n})
    return pd.DataFrame(rows).set_index("lead")


def refit_cycle_ic(raw: np.ndarray, ret: np.ndarray, retrain_every: int, horizon: int, gap: int) -> pd.DataFrame:
    """IC of the score with the T+1 return at each position of the refit cycle, with the predicted overlap.

    Assumes the first scored row is a refit (as in ``walk_forward``), so the cycle
    position of the k-th scored row is ``k % retrain_every``.
    """
    raw = np.asarray(raw, dtype=float)
    ret = np.asarray(ret, dtype=float)
    pos = np.arange(len(raw)) % retrain_every
    rows = []
    for c in range(retrain_every):
        ic, n = _ic(raw[pos == c], ret[pos == c])
        rows.append({"cycle_pos": c, "overlap": label_overlap(horizon, c, gap), "ic": ic, "n": n})
    return pd.DataFrame(rows).set_index("cycle_pos")


def fingerprint_test(raw: np.ndarray, ret: np.ndarray, retrain_every: int, horizon: int, gap: int) -> dict[str, float]:
    """Compare IC on cycle days with overlap against cycle days without, by Fisher z.

    Returns the two ICs, their sample sizes, ``z`` and a two-sided ``p``. With an honest
    purge every day has zero overlap and the test returns NaN: there is nothing to compare.
    """
    raw = np.asarray(raw, dtype=float)
    ret = np.asarray(ret, dtype=float)
    pos = np.arange(len(raw)) % retrain_every
    has = np.array([label_overlap(horizon, c, gap) > 0 for c in pos])
    if has.all() or not has.any():
        return {"ic_overlap": float("nan"), "ic_clean": float("nan"), "n_overlap": int(has.sum()),
                "n_clean": int((~has).sum()), "z": float("nan"), "p": float("nan")}
    ic1, n1 = _ic(raw[has], ret[has])
    ic0, n0 = _ic(raw[~has], ret[~has])
    z = (np.arctanh(ic1) - np.arctanh(ic0)) / np.sqrt(1 / (n1 - 3) + 1 / (n0 - 3))
    return {"ic_overlap": ic1, "ic_clean": ic0, "n_overlap": n1, "n_clean": n0,
            "z": float(z), "p": float(2 * (1 - norm.cdf(abs(z))))}
