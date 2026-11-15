"""
bootstrap_history.py — one-time: build initial score_history.csv from the
existing walk-forward score CSVs (from the research repo).

Run this ONCE after first deployment, before the first live signal. It populates
history/score_history.csv with the full walk-forward OOS scores for all 11 models,
so that the past-only percentile mechanic has its baseline ready on day 1.

Source: the 11 score_history_*.csv files (passed via SOURCE_DIR).
Output: history/score_history.csv with columns
        date, raw_v1, raw_T2, ..., raw_AUG_h20
"""
from __future__ import annotations
import sys, datetime as dt
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
HIST = HERE / "history"; HIST.mkdir(exist_ok=True)

# Default source = the staged_finnhub research output dir.
# Override on the command line: python bootstrap_history.py "<local-path>"
SOURCE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "<local-path>")

FM = {
    "v1":      "winning_oos_scores.csv",
    "T2":      "score_history_T2.csv",
    "T3":      "score_history_T3.csv",
    "CV3":     "score_history_CV3.csv",
    "S6":      "score_history_S6.csv",
    "CV1":     "score_history_CV1.csv",
    "FS2":     "score_history_FS2.csv",
    "Th10":    "score_history_T_h10.csv",
    "AUG1":    "score_history_AUG1.csv",
    "LGB_h10": "score_history_LGB_h10.csv",
    "AUG_h20": "score_history_AUG_h20.csv",
}

if not SOURCE_DIR.exists():
    print(f"SOURCE_DIR not found: {SOURCE_DIR}")
    sys.exit(1)

print(f"Reading scores from: {SOURCE_DIR}")
out = None
for name, fname in FM.items():
    p = SOURCE_DIR / fname
    if not p.exists():
        print(f"  MISSING: {p}"); continue
    df = pd.read_csv(p, parse_dates=["date"])[["date","raw"]].rename(columns={"raw": f"raw_{name}"})
    if out is None: out = df
    else:           out = out.merge(df, on="date", how="outer")
    print(f"  loaded {name}: {len(df)} rows")

if out is None or out.empty:
    print("No scores loaded. Check SOURCE_DIR.")
    sys.exit(1)

out = out.sort_values("date").reset_index(drop=True)
target = HIST / "score_history.csv"
out.to_csv(target, index=False)
print(f"\nWrote {target}  ({len(out)} rows, {len(out.columns)-1} model columns)")
print(f"Date range: {out['date'].min().date()} -> {out['date'].max().date()}")
