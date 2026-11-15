"""Write synthetic daily closes to data/prices.csv (same columns as the real input: date,spy,vixy,vix).

    python scripts/make_synthetic_data.py                # no planted signal
    python scripts/make_synthetic_data.py --signal 0.3   # planted signal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volhedge.data import make_synthetic_prices  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "data" / "prices.csv"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_prices(args.days, seed=args.seed, signal_strength=args.signal)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
