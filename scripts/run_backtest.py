"""End-to-end run: prices -> features -> purged walk-forward -> overlay -> results.

Runs five books on the same evaluation days:
  base             static 80/20 hold
  const_short      constant -1 x book (earns the ETP decay with no model)
  h1               T+1 model, purged
  h20              T+20 model, purged (horizon + embargo)
  h20_leaky        T+20 model, embargo only (the original bug, kept for comparison)

    python scripts/run_backtest.py                       # synthetic, no planted signal
    python scripts/run_backtest.py --signal 0.3          # synthetic, planted signal
    python scripts/run_backtest.py --prices my.csv       # your own date,spy,vixy,vix closes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from volhedge.data import load_prices, make_synthetic_prices  # noqa: E402
from volhedge.features import build_features  # noqa: E402
from volhedge.strategy import block_permutation_p, pnl, positions, stats  # noqa: E402
from volhedge.walkforward import walk_forward  # noqa: E402

RUNS = {"h1": (1, False), "h20": (20, False), "h20_leaky": (20, True)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="CSV with date,spy,vixy,vix (default: synthetic)")
    ap.add_argument("--signal", type=float, default=0.0, help="planted signal strength (synthetic only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=2000)
    ap.add_argument("--cost-bp", type=float, default=2.0, help="cost per unit turnover, bp")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    prices = (load_prices(args.prices) if args.prices
              else make_synthetic_prices(args.days, seed=args.seed, signal_strength=args.signal))
    table, feats = build_features(prices)
    cost = args.cost_bp / 1e4
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {"input": args.prices or f"synthetic (seed {args.seed}, signal {args.signal})",
                     "cost_bp": args.cost_bp, "books": {}}
    base_ret = None
    for name, (h, leaky) in RUNS.items():
        oos = walk_forward(prices, table, feats, horizon=h, leaky=leaky).dropna(subset=["ret"])
        w = positions(oos["raw"].values)
        live = np.arange(len(w)) >= 250  # evaluate after the percentile warm-up only
        r = oos["ret"].values
        if base_ret is None:
            base_ret = r[live]
            summary["books"]["base"] = stats(base_ret)
            short = np.full(int(live.sum()), -1.0)
            summary["books"]["const_short"] = stats(pnl(short, base_ret, cost))
        # overlay = base book scaled by w; compare on the same days
        res = stats(pnl(w, r, cost)[live])
        res["perm_p"] = round(block_permutation_p(w[live], r[live], cost=cost), 3)
        res["mean_w"] = round(float(w[live].mean()), 3)
        summary["books"][name] = res
        oos.assign(w=w).to_csv(out / f"oos_{name}.csv", index=False)

    for k, v in summary["books"].items():
        print(f"{k:10s} " + "  ".join(f"{a}={b}" for a, b in v.items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
