import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volhedge.diagnostics import fingerprint_test, ic_by_lead, label_overlap, refit_cycle_ic  # noqa: E402
from volhedge.walkforward import purge_rows  # noqa: E402


@pytest.mark.parametrize("h", [1, 3, 5, 10, 20])
@pytest.mark.parametrize("leaky", [False, True])
def test_label_overlap_matches_brute_force(h, leaky):
    """Count training labels that contain the return predicted on each cycle day, row by row."""
    gap, i = purge_rows(h, leaky=leaky), 500
    for c in range(25):
        target = i + c  # predicted return: close[target] -> close[target + 1]
        count = sum(1 for r in range(i - gap) if r <= target and r + h >= target + 1)
        assert label_overlap(h, c, gap) == count


@pytest.mark.parametrize("h", [1, 5, 20])
def test_purged_gap_leaves_no_overlap_on_any_cycle_day(h):
    assert all(label_overlap(h, c, purge_rows(h)) == 0 for c in range(42))


def test_original_settings_reproduce_the_known_overlap_pattern():
    # 2-day embargo, refit every 5 days: T+5 overlaps 2,1,0,0,0 and T+10 overlaps 7..3
    assert [label_overlap(5, c, 2) for c in range(5)] == [2, 1, 0, 0, 0]
    assert [label_overlap(10, c, 2) for c in range(5)] == [7, 6, 5, 4, 3]


def _planted(n: int, retrain_every: int, horizon: int, gap: int, seed: int = 0):
    """Scores that 'know' the return only on cycle days with overlap, as a leaky model would."""
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n)
    pos = np.arange(n) % retrain_every
    leak = np.array([label_overlap(horizon, c, gap) > 0 for c in pos])
    raw = rng.standard_normal(n) + np.where(leak, 1.0 * ret, 0.0)
    return raw, ret


def test_fingerprint_detects_a_planted_cycle_leak():
    raw, ret = _planted(3000, retrain_every=5, horizon=5, gap=2)
    res = fingerprint_test(raw, ret, 5, 5, 2)
    assert res["ic_overlap"] > 0.4 and abs(res["ic_clean"]) < 0.1 and res["p"] < 1e-6
    table = refit_cycle_ic(raw, ret, 5, 5, 2)
    assert list(table["overlap"]) == [2, 1, 0, 0, 0]
    assert table["ic"].iloc[:2].min() > table["ic"].iloc[2:].max()


def test_fingerprint_is_quiet_on_honest_scores():
    rng = np.random.default_rng(1)
    ret = rng.standard_normal(3000)
    raw = 0.2 * ret + rng.standard_normal(3000)  # real skill, the same on every day
    res = fingerprint_test(raw, ret, 5, 5, 2)
    assert res["p"] > 0.01
    assert np.isnan(fingerprint_test(raw, ret, 21, 20, purge_rows(20))["p"])  # purged: nothing to compare


def test_ic_by_lead_finds_the_label_boundary():
    """A score built from the next 3 returns has IC at leads 1-3 and none after."""
    rng = np.random.default_rng(2)
    ret = rng.standard_normal(4000)
    raw = ret + np.r_[ret[1:], 0] + np.r_[ret[2:], 0, 0] + rng.standard_normal(4000)
    ic = ic_by_lead(raw, ret, max_lead=8)["ic"]
    assert ic.loc[1:3].min() > 0.3 and ic.loc[4:].abs().max() < 0.06
