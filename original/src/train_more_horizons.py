"""
train_more_horizons.py — train more multi-horizon target models (T+2, T+4, T+7, T+10).

We have T+1 (v1), T+3 (T2), T+5 (T3). Adding T+2, T+4, T+7, T+10 may add diversity.
Same HGBR-deep recipe on 317 base features.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_p7030"))

from feature_build import build_matrix
from sklearn.ensemble import HistGradientBoostingRegressor
from train_v2 import cont_weight_audit_split_cost, _sh, _stats, _per_year, COST_BLENDED

OUT = HERE / "output"
EMB = 2

m = build_matrix(verbose=False)
d = m["d"].copy().sort_values("date").reset_index(drop=True)
feats = [c for c in m["FSETS"]["ALL"] if c in d.columns]
print(f"Features: {len(feats)}")

# build targets
spy_n  = d["spy"].shift(-1)  / d["spy"]  - 1
vixy_n = d["vixy"].shift(-1) / d["vixy"] - 1
d["ret_t1"] = 0.80 * spy_n + 0.20 * vixy_n
for h in (2, 4, 7, 10):
    spy_h  = d["spy"].shift(-h)  / d["spy"]  - 1
    vixy_h = d["vixy"].shift(-h) / d["vixy"] - 1
    d[f"ret_t{h}"] = 0.80 * spy_h + 0.20 * vixy_h
d["ret_actual"] = d["ret_t1"]

med = d.loc[d.date <= "2021-12-31", feats].median()
d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

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

# Train T+2, T+4, T+7, T+10
horizons = [(2, "T_h2"), (4, "T_h4"), (7, "T_h7"), (10, "T_h10")]
results = []
for h, label in horizons:
    scores = train_oos(d, feats, f"ret_t{h}", label)
    scores.to_csv(OUT/f"score_history_{label}.csv", index=False)
    df = scores.dropna(subset=["raw","ret"]).sort_values("date").reset_index(drop=True)
    pl, yl = a3_audit(df["raw"].values, df["ret"].values, df["date"].values)
    sh, tot, mdd, calmar, cagr = _stats(pl)
    py = _per_year(pl, yl)
    pos_yrs = sum(v > 0 for y, v in py.items() if y <= 2025)
    print(f"\n  {label} (T+{h} target):  Sh {sh:+.2f}  CAGR {cagr:+.1f}%  Tot {tot:+.0f}%  MaxDD {mdd:+.0f}%  posYr {pos_yrs}/5")
    results.append((label, h, sh, tot, mdd, pos_yrs))

print("\n=== T+H solo Sharpes (A3 mechanic) ===")
print(f"  {'label':>6s}  {'h':>3s}  {'Sharpe':>7s}  {'Tot':>6s}  {'posYr':>6s}")
for label, h, sh, tot, mdd, py in results:
    print(f"  {label:>6s}  {h:>3d}  {sh:>+7.2f}  {tot:>+5.0f}%  {py}/5")
print(f"\n  Existing: T+1(v1) {1.42:+.2f}, T+3(T2) {1.22:+.2f}, T+5(T3) {1.21:+.2f}")
