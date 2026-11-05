import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volhedge.data import make_synthetic_prices  # noqa: E402
from volhedge.features import build_features  # noqa: E402
from volhedge.strategy import block_permutation_p, pnl, positions, stats  # noqa: E402
from volhedge.walkforward import purge_rows, walk_forward  # noqa: E402


def test_features_only_use_past():
    """Changing the close of day t must not change any feature in row t or earlier."""
    noise = make_synthetic_prices(1300, seed=3)
    t = 900
    base, names = build_features(noise)
    bumped = noise.copy()
    bumped.loc[t:, ["spy", "vixy", "vix"]] *= 1.5
    after, _ = build_features(bumped)
    np.testing.assert_allclose(base.loc[:t, names].values, after.loc[:t, names].values, equal_nan=True)


@pytest.mark.parametrize("h", [1, 5, 20])
def test_purge_covers_label_horizon(h):
    """Last training row's label (ends at row + h) must end before the prediction row."""
    i = 1000
    last_train = i - purge_rows(h) - 1
    assert last_train + h < i


def test_leaky_mode_overlaps_target():
    """The original 2-day embargo lets T+20 labels run past the prediction row."""
    i = 1000
    last_train = i - purge_rows(20, leaky=True) - 1
    assert last_train + 20 >= i


def test_positions_are_past_only():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(600)
    w = positions(raw)
    raw2 = raw.copy()
    raw2[400:] += 10.0
    assert np.allclose(w[:400], positions(raw2)[:400])
    assert np.all(np.abs(w) <= 1.0)


def test_noise_shows_no_edge():
    """Across several pure-noise worlds the overlay is not reliably significant.

    One seed can pass p<0.05 by chance (5% false-positive rate, and the ETP decay can be
    harvested by a net-short position), so the check is over five seeds.
    """
    ps, sharpes = [], []
    for seed in (1, 2, 3, 4, 5):
        px = make_synthetic_prices(1300, seed=seed, signal_strength=0.0)
        table, feats = build_features(px)
        oos = walk_forward(px, table, feats, horizon=1, start_oos=600, retrain_every=42).dropna(subset=["ret"])
        w = positions(oos["raw"].values)[250:]
        r = oos["ret"].values[250:]
        ps.append(block_permutation_p(w, r, n=100))
        sharpes.append(stats(pnl(w, r))["sharpe"])
    assert sum(p < 0.05 for p in ps) <= 1, ps
    assert np.median(sharpes) < 1.0, sharpes


def test_planted_signal_is_recovered():
    px = make_synthetic_prices(1300, seed=3, signal_strength=0.3)
    table, feats = build_features(px)
    oos = walk_forward(px, table, feats, horizon=1, start_oos=600, retrain_every=42).dropna(subset=["ret"])
    w = positions(oos["raw"].values)[250:]
    r = oos["ret"].values[250:]
    assert stats(pnl(w, r))["sharpe"] > 0.5


@pytest.mark.parametrize("h,leaky", [(1, False), (20, False), (20, True)])
def test_walk_forward_fits_on_purged_rows(h, leaky):
    """Check the rows actually passed to fit(), not just the purge arithmetic."""
    import volhedge.walkforward as wf

    px = make_synthetic_prices(900, seed=4)
    table, feats = build_features(px)
    sizes = []
    real = wf.make_model

    class Spy:
        def __init__(self, seed=42):
            self.m = real(seed)

        def fit(self, X, y):
            sizes.append(len(X))
            self.m.fit(X, y)
            return self

        def predict(self, X):
            return self.m.predict(X)

    wf.make_model = Spy
    try:
        walk_forward(px, table, feats, horizon=h, start_oos=600, retrain_every=100, leaky=leaky)
    finally:
        wf.make_model = real
    # first fit is for prediction row 600: rows [0, 600 - gap) are used
    assert sizes[0] == 600 - purge_rows(h, leaky=leaky)
