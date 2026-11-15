"""Label-overlap diagnostics on out-of-sample scores: IC by lead and the refit-cycle fingerprint.

By default this runs the T+20 model twice on synthetic prices, with the original
2-day embargo and with the horizon-sized purge, and prints both diagnostics. To check
another model, pass its score history (``date, raw, ret`` as written by
``run_backtest.py``) with the horizon, refit cadence and purge gap it was trained with.

    python scripts/leakage_diagnostics.py
    python scripts/leakage_diagnostics.py --scores results/oos_h20_leaky.csv --horizon 20 --gap 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from volhedge.data import make_synthetic_prices  # noqa: E402
from volhedge.diagnostics import fingerprint_test, ic_by_lead, refit_cycle_ic  # noqa: E402
from volhedge.features import build_features  # noqa: E402
from volhedge.walkforward import purge_rows, walk_forward  # noqa: E402


def report(name: str, raw, ret, horizon: int, retrain_every: int, gap: int) -> None:
    fp = fingerprint_test(raw, ret, retrain_every, horizon, gap)
    lead = ic_by_lead(raw, ret, max_lead=min(horizon + 5, 30))["ic"]
    cyc = refit_cycle_ic(raw, ret, retrain_every, horizon, gap)
    print(f"== {name}: horizon {horizon}, refit every {retrain_every}, purge gap {gap}")
    print(f"   cycle days with label overlap: {int((cyc['overlap'] > 0).sum())} of {retrain_every}")
    if pd.isna(fp["p"]):
        print("   fingerprint: no cycle day has overlap, nothing to compare (as it should be)")
    else:
        print(f"   fingerprint: IC on overlap days {fp['ic_overlap']:+.3f} (n {fp['n_overlap']}) vs "
              f"clean days {fp['ic_clean']:+.3f} (n {fp['n_clean']}), z {fp['z']:+.2f}, p {fp['p']:.3f}")
    print("   IC by lead k=1..: " + " ".join(f"{v:+.2f}" for v in lead.values))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", help="CSV with raw and ret columns (default: run synthetic)")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--retrain-every", type=int, default=21)
    ap.add_argument("--gap", type=int, help="purge gap used in training (rows)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=2000)
    args = ap.parse_args(argv)

    if args.scores:
        df = pd.read_csv(args.scores).dropna(subset=["ret"])
        gap = args.gap if args.gap is not None else purge_rows(args.horizon)
        report(Path(args.scores).name, df["raw"].values, df["ret"].values, args.horizon, args.retrain_every, gap)
        return 0

    prices = make_synthetic_prices(args.days, seed=args.seed)
    table, feats = build_features(prices)
    for leaky in (True, False):
        gap = purge_rows(args.horizon, leaky=leaky)
        oos = walk_forward(prices, table, feats, horizon=args.horizon, retrain_every=args.retrain_every,
                           leaky=leaky).dropna(subset=["ret"])
        report(f"synthetic seed {args.seed}, {'leaky 2-day embargo' if leaky else 'purged'}",
               oos["raw"].values, oos["ret"].values, args.horizon, args.retrain_every, gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
