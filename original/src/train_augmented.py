"""
train_augmented.py — train HGBR-deep on augmented feature set.

Variants:
  AUG1   — ALL_AUG (728) features, T+1 target
  AUGh10 — ALL_AUG (728) features, T+10 target  (Th10 was best new horizon)
  NEW1   — NEW_ONLY (411) features, T+1 target  (pure-orthogonal check)
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_p7030"))

from build_augmented_matrix import build_augmented
from sklearn.ensemble import HistGradientBoostingRegressor
from train_v2 import _stats, _per_year, COST_BLENDED

OUT = HERE / "output"
EMB = 2

# ---------- shared A3 audit ----------
def f_ema(span): return lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean()
def a3_audit(raw, ret, dates, scale=3.5, max_pos=0.6, span=7):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "raw": raw, "ret": ret})
    df["sm"] = f_ema(span)(df["raw"])
    df["ym"] = df["date"].dt.to_period("M")
    pnl = np.zeros(len(df)); ps = np.zeros(len(df))
    for m_ in sorted(df["ym"].unique()):
        mask = (df["ym"]==m_).values; fi = np.where(mask)[0][0]
        if df["date"].iloc[fi] < pd.Timestamp("2021-01-01"): continue
        past = df.iloc[:fi].dropna(subset=["sm"])
        if len(past) < 400: continue
        past_sm = past["sm"].values
        for j in np.where(mask)[0]:
            tp = float((past_sm < df["sm"].iloc[j]).mean())
            w = float(np.clip(scale*(2*tp - 1), -max_pos, max_pos))
            prev = ps[j-1] if j > 0 else 0.0
            pnl[j] = w * df["ret"].iloc[j] - abs(w - prev) * COST_BLENDED
            ps[j] = w
    live = (df["date"] >= pd.Timestamp("2021-01-01")).values
    rl = df["ret"].values[live]; pl = pnl[live]
    yl = df["date"].iloc[df.index[live]].dt.year.values
    valid = ~np.isnan(rl)
    return pl[valid], yl[valid]

# ---------- build matrix ----------
print("Building augmented matrix...")
m = build_augmented(verbose=True)
d = m["d"].copy().sort_values("date").reset_index(drop=True)

# Targets
spy_n  = d["spy"].shift(-1)  / d["spy"]  - 1
vixy_n = d["vixy"].shift(-1) / d["vixy"] - 1
d["ret_t1"] = 0.80 * spy_n + 0.20 * vixy_n
spy_h  = d["spy"].shift(-10)  / d["spy"]  - 1
vixy_h = d["vixy"].shift(-10) / d["vixy"] - 1
d["ret_t10"] = 0.80 * spy_h + 0.20 * vixy_h
d["ret_actual"] = d["ret_t1"]

FEATS_AUG = [c for c in m["FSETS"]["ALL_AUG"] if c in d.columns]
FEATS_NEW = [c for c in m["FSETS"]["NEW_ONLY"] if c in d.columns]
print(f"\nALL_AUG features: {len(FEATS_AUG)} | NEW_ONLY: {len(FEATS_NEW)}")

# Fill NaN in non-NaN-tolerant places using pre-2021 medians; HGBR handles NaN anyway
# but ret_t1/t10 NaN must be dropped only at train time
med = d.loc[d.date <= "2021-12-31", FEATS_AUG].median()
d[FEATS_AUG] = d[FEATS_AUG].replace([np.inf, -np.inf], np.nan)
# DO NOT pre-fill - HGBR-native NaN handling is what we want for max signal

# ---------- training loop ----------
def train_oos(d, feats, target_col, label):
    idx = d.index[d.date >= "2019-01-01"].tolist()
    mdl = None; rows = []
    t0 = time.time()
    for k, i in enumerate(idx):
        if k % 5 == 0 or mdl is None:
            g = d.loc[:i - 1 - EMB].dropna(subset=[target_col])
            if len(g) < 800: continue
            mdl = HistGradientBoostingRegressor(
                max_iter=500, max_depth=7, learning_rate=0.02,
                min_samples_leaf=30, l2_regularization=0.5, random_state=42)
            mdl.fit(g[feats].values, g[target_col].values)
        r = d.loc[i]
        rawp = float(mdl.predict(r[feats].values.reshape(1, -1))[0])
        rows.append({"date": r["date"], "raw": rawp,
                     "ret": float(r["ret_actual"]) if pd.notna(r["ret_actual"]) else np.nan})
        if k % 400 == 0:
            print(f"    [{label}] {k}/{len(idx)}  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)

variants = [
    ("AUG1",   FEATS_AUG, "ret_t1"),
    ("AUGh10", FEATS_AUG, "ret_t10"),
    ("NEW1",   FEATS_NEW, "ret_t1"),
]

results = []
for label, feats, target in variants:
    print(f"\n=== Training {label} (target={target}, |feats|={len(feats)}) ===")
    scores = train_oos(d, feats, target, label)
    scores.to_csv(OUT/f"score_history_{label}.csv", index=False)
    df = scores.dropna(subset=["raw","ret"]).sort_values("date").reset_index(drop=True)
    pl, yl = a3_audit(df["raw"].values, df["ret"].values, df["date"].values)
    sh, tot, mdd, calmar, cagr = _stats(pl)
    py = _per_year(pl, yl)
    pos_yrs = sum(v > 0 for y, v in py.items() if y <= 2025)
    print(f"  {label}:  Sh {sh:+.2f}  CAGR {cagr:+.1f}%  Tot {tot:+.0f}%  MaxDD {mdd:+.0f}%  posYr {pos_yrs}/5")
    results.append((label, sh, tot, mdd, pos_yrs))

print("\n=== Augmented model solo Sharpes (A3 mechanic) ===")
print(f"  {'label':>6s}  {'Sh':>6s}  {'Tot':>6s}  {'MDD':>5s}  {'posYr':>6s}")
for label, sh, tot, mdd, py in results:
    print(f"  {label:>6s}  {sh:>+6.2f}  {tot:>+5.0f}%  {mdd:>+4.0f}%  {py}/5")
print(f"\n  Existing solo: v1 +1.42, T2 +1.22, T3 +1.21, Th10 +1.50")
