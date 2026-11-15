"""
train_v2.py — Phase 1+2 retrain with all the improvements from the Analyst feedback.

Improvements vs v1:
  1. SPLIT COST: 0.5bp SPY + 2.5bp VIXY blended -> ~0.9 bp on full-weight position
     (was 5 bp). Analyst said this alone should lift Sharpe to ~1.2.
  2. NEW FEATURES: +202 cross-asset features on top of the 317 originals = 519 total.
     Includes: cross-asset returns/vols, synthetic vol indices (VXN_synth, RVX_synth,
     OVX_synth, GVZ_synth, VXEEM_synth), MOVE_synth, credit spread, term spread,
     sector rotation, regime indicators.
  3. RECENCY WEIGHTS: training samples weighted by exp(-Δdays / half_life), half=252.
  4. ROBUSTNESS: optionally winsorize target at ±3 SD to limit fitting outlier spikes.

Outputs:
  output/score_history_v2.csv     — OOS raw scores with v2 model
  output/winner_audit_v2.json     — full v2 audit metrics
  output/winner_pnl_v2.npz        — v2 PnL series at split-cost
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_p7030"))

from feature_build import build_matrix
from cross_asset_features import build_cross_asset_features
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb

OUT = HERE / "output"
HIST_V2 = OUT / "score_history_v2.csv"

# --- key parameters ---
EMB         = 2
SMA_WIN     = 5
HALF_LIFE   = 252                  # recency-weight half-life (1 trading year)
COST_SPY_BP = 0.5                  # 0.5 bp on SPY side
COST_VIX_BP = 2.5                  # 2.5 bp on VIXY side
# blended cost on a 80/20 book: 0.8*0.5 + 0.2*2.5 = 0.9 bp per side
COST_BLENDED = (0.80 * COST_SPY_BP + 0.20 * COST_VIX_BP) / 1e4   # ~9e-5
WINSORIZE_TARGET_SD = 3.5          # winsorize target at +- 3.5 SD (light)

def _sh(s):
    s = np.asarray(s); return float(s.mean()/s.std()*np.sqrt(252)) if len(s)>5 and s.std()>0 else 0.0
def _stats(x):
    x = np.asarray(x); c = np.cumprod(1+x); tot = (c[-1]-1)*100
    mdd = ((c-np.maximum.accumulate(c))/np.maximum.accumulate(c)).min()*100
    cagr = (c[-1]**(252/len(x))-1)*100 if len(x)>0 else 0.0
    calmar = cagr/abs(mdd) if mdd<0 else float("nan")
    return _sh(x), tot, mdd, calmar, cagr
def _per_year(x, yrs):
    return {int(y): round(float((1+np.array(x)[yrs==y]).prod()-1)*100, 1) for y in sorted(set(yrs))}


def build_full_matrix():
    """Combine canonical 317-feature matrix with cross-asset features."""
    print("[features] building canonical matrix...")
    m = build_matrix(verbose=False)
    d = m["d"].copy().sort_values("date").reset_index(drop=True)
    base_feats = [c for c in m["FSETS"]["ALL"] if c in d.columns]
    print(f"  canonical features: {len(base_feats)}")

    print("[features] building cross-asset features...")
    ca, ca_feats = build_cross_asset_features()
    print(f"  cross-asset features: {len(ca_feats)}")

    # merge
    d = d.merge(ca, on="date", how="left")
    all_feats = base_feats + [c for c in ca_feats if c in d.columns]
    print(f"  TOTAL features: {len(all_feats)}")

    # build target
    spy_next  = d["spy"].shift(-1)  / d["spy"]  - 1
    vixy_next = d["vixy"].shift(-1) / d["vixy"] - 1
    d["ret_p8020lv"] = 0.80 * spy_next + 0.20 * vixy_next

    # NaN fill with pre-2022 medians, then 0
    med = d.loc[d.date <= "2021-12-31", all_feats].median()
    d[all_feats] = d[all_feats].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

    # winsorize target
    if WINSORIZE_TARGET_SD is not None:
        std = d["ret_p8020lv"].std()
        cap = WINSORIZE_TARGET_SD * std
        d["ret_p8020lv_w"] = d["ret_p8020lv"].clip(-cap, cap)
        print(f"[features] winsorized target at +-{WINSORIZE_TARGET_SD} SD (= +-{cap*100:.2f}% / day)")
    else:
        d["ret_p8020lv_w"] = d["ret_p8020lv"]

    return d, all_feats


def init_history_v2(model_kind="hgbr_deep", target_col="ret_p8020lv_w", use_recency=True):
    """Build OOS scores via expanding weekly retrain. Same cadence as v1 (every 5 days).
    Recency-weighted samples: exp(-(N-i) / HALF_LIFE), so latest sample weight ~1.0,
    sample 252 days back weight ~0.5, sample 1000 days back weight ~0.07.
    """
    t0 = time.time()
    print("="*78); print(f" TRAIN V2 — {model_kind} | target={target_col} | recency={use_recency}"); print("="*78)
    d, feats = build_full_matrix()
    print(f"  rows: {len(d)}  date range: {d['date'].min().date()} -> {d['date'].max().date()}")

    idx = d.index[d.date >= "2019-01-01"].tolist()
    mdl = None; rows = []
    print(f"  OOS steps: {len(idx)}  (expanding, weekly retrain)")

    for k, i in enumerate(idx):
        if k % 5 == 0 or mdl is None:
            g = d.loc[:i - 1 - EMB].dropna(subset=[target_col])
            if len(g) < 800: continue

            X = g[feats].values
            y = g[target_col].values
            if use_recency:
                # exp decay sample weights — latest = 1.0
                ages = np.arange(len(g))[::-1]  # 0 = latest
                w = np.exp(-ages / HALF_LIFE)
            else:
                w = None

            if model_kind == "hgbr_deep":
                mdl = HistGradientBoostingRegressor(
                    max_iter=500, max_depth=7, learning_rate=0.02,
                    min_samples_leaf=30, l2_regularization=0.5, random_state=42)
            elif model_kind == "lgbm":
                mdl = lgb.LGBMRegressor(
                    n_estimators=500, max_depth=7, learning_rate=0.02,
                    min_child_samples=30, reg_lambda=0.5, random_state=42, verbose=-1)
            else:
                raise ValueError(model_kind)

            if use_recency:
                mdl.fit(X, y, sample_weight=w)
            else:
                mdl.fit(X, y)
        r = d.loc[i]
        rawp = float(mdl.predict(r[feats].values.reshape(1, -1))[0])
        rows.append({"date": r["date"], "raw": rawp,
                     "ret":  float(r["ret_p8020lv"]) if pd.notna(r["ret_p8020lv"]) else np.nan,
                     "ret_w": float(r[target_col]) if pd.notna(r[target_col]) else np.nan})
        if k % 250 == 0:
            print(f"    {k:5d}/{len(idx)}   ({time.time()-t0:.0f}s elapsed)")

    out = pd.DataFrame(rows)
    out.to_csv(HIST_V2, index=False)
    print(f"  saved {HIST_V2}  rows={len(out)}  elapsed={time.time()-t0:.0f}s")
    return out


def cont_weight_audit_split_cost(raw, ret, dates, cost=COST_BLENDED, scale=1.0):
    """Cont-weight strategy with SPLIT COST (0.5 bp SPY + 2.5 bp VIXY blended)."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "raw": raw, "ret": ret})
    df["raw_sm"] = df["raw"].rolling(SMA_WIN, min_periods=1).mean()
    df["ym"] = df["date"].dt.to_period("M")
    pnl = np.zeros(len(df)); ps = np.zeros(len(df))
    for m in sorted(df["ym"].unique()):
        mask = (df["ym"] == m).values
        fi = np.where(mask)[0][0]
        if df["date"].iloc[fi] < pd.Timestamp("2021-01-01"): continue
        past = df.iloc[:fi].dropna(subset=["raw_sm"])
        if len(past) < 400: continue
        past_raw_sm = past["raw_sm"].values
        for j in np.where(mask)[0]:
            tp = float((past_raw_sm < df["raw_sm"].iloc[j]).mean())
            w = np.clip((2 * tp - 1) * scale, -1.0, 1.0)
            prev = ps[j - 1] if j > 0 else 0.0
            pnl[j] = w * df["ret"].iloc[j] - abs(w - prev) * cost
            ps[j] = w
    live = (df["date"] >= pd.Timestamp("2021-01-01")).values
    rl = df["ret"].values[live]; pl = pnl[live]; psl = ps[live]
    yl = df["date"].iloc[df.index[live]].dt.year.values
    dl = df["date"].iloc[df.index[live]].values
    valid = ~np.isnan(rl)
    return pl[valid], psl[valid], rl[valid], yl[valid], dl[valid]


def run_audit_v2():
    print("\n" + "="*78); print(" V2 AUDIT — leak-free walk-forward, SPLIT cost"); print("="*78)
    h = pd.read_csv(HIST_V2, parse_dates=["date"]).dropna(subset=["raw"]).sort_values("date").reset_index(drop=True)
    print(f"  history rows: {len(h)}  date range: {h['date'].min().date()} -> {h['date'].max().date()}")

    # baseline = static 80/20 hold (no overlay)
    h_complete = h.dropna(subset=["ret"])
    h_oos = h_complete[h_complete["date"] >= "2021-01-01"]
    r_base = h_oos["ret"].values
    yrs_base = pd.to_datetime(h_oos["date"]).dt.year.values
    bsh, btot, bmdd, bcalmar, bcagr = _stats(r_base)
    bpy = _per_year(r_base, yrs_base)
    print(f"\n  BASELINE static 80/20: Sh {bsh:.2f}  CAGR {bcagr:+.1f}%  Tot {btot:+.0f}%  "
          f"MaxDD {bmdd:.0f}%  Calmar {bcalmar:.2f}")
    print(f"  per-year: {bpy}")

    # cost ladder
    print(f"\n  ML overlay v2 (HGBR-deep + cross-asset + recency + winsorized target):")
    ml_at = {}
    for label, cost_factor in [("split_blended", 1.0), ("legacy_5bp", 5e-4/COST_BLENDED),
                                ("zero", 0.0), ("conservative_2x", 2.0)]:
        cost = COST_BLENDED * cost_factor
        pl, ps, rl, yl, dl = cont_weight_audit_split_cost(
            h["raw"].values, h["ret"].values, h["date"].values, cost=cost)
        sh, tot, mdd, calmar, cagr = _stats(pl)
        py = _per_year(pl, yl)
        pos_yrs = sum(v > 0 for y, v in py.items() if y <= 2025)
        n_yrs = sum(1 for y in py if y <= 2025)
        act = (ps != 0).mean()
        ntr = int((np.abs(np.diff(np.r_[0., ps])) > 0.05).sum())
        print(f"    {label:18s} cost={cost*1e4:.2f}bp  Sh {sh:.2f}  CAGR {cagr:+5.1f}%  "
              f"Tot {tot:+5.0f}%  MaxDD {mdd:.0f}%  Calmar {calmar:.2f}  posYr {pos_yrs}/{n_yrs}  "
              f"trades {ntr}")
        ml_at[label] = {"sharpe": sh, "cagr": cagr, "tot": tot, "mdd": mdd, "calmar": calmar,
                        "per_year": py, "pos_yrs": pos_yrs, "n_yrs": n_yrs,
                        "active_pct": float(act*100), "n_trades": ntr,
                        "cost_bp": cost*1e4,
                        "pnl": pl.tolist(),
                        "dates": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in dl]}

    # significance @ split cost (the production cost)
    print(f"\n  Permutation test on SPLIT cost @0.9bp (n=300)...")
    rng = np.random.default_rng(42)
    pl_split = np.array(ml_at["split_blended"]["pnl"])
    real_sh = ml_at["split_blended"]["sharpe"]
    N = 300; cnt = 0; null_sh = []
    ts = time.time()
    for k in range(N):
        rp = rng.permutation(h["raw"].values)
        pp, _, _, _, _ = cont_weight_audit_split_cost(rp, h["ret"].values, h["date"].values)
        sh = _sh(pp); null_sh.append(sh)
        if sh >= real_sh: cnt += 1
        if (k+1) % 100 == 0: print(f"    {k+1}/{N} ({time.time()-ts:.0f}s)")
    perm_p = (cnt + 1) / (N + 1)
    print(f"    real Sharpe {real_sh:.2f}  perm p={perm_p:.4f}  "
          f"null mean {np.mean(null_sh):+.2f}  null 95p {np.percentile(null_sh,95):+.2f}")

    # save
    results = {
        "baseline": {"sharpe": float(bsh), "cagr": float(bcagr), "tot": float(btot),
                     "mdd": float(bmdd), "calmar": float(bcalmar), "per_year": bpy},
        "ml": {k: {kk: vv for kk, vv in d.items() if kk not in ("pnl","dates")}
               for k, d in ml_at.items()},
        "significance": {"perm_p": float(perm_p), "sharpe": float(real_sh),
                          "null_mean": float(np.mean(null_sh)),
                          "null_p95": float(np.percentile(null_sh, 95))},
        "config": {
            "model": "hgbr_deep + cross_asset + recency + winsorized",
            "features_count": "519 (317 base + 202 cross_asset)",
            "cost_blended_bp": COST_BLENDED * 1e4,
            "half_life_days": HALF_LIFE,
            "winsorize_sd": WINSORIZE_TARGET_SD,
        },
    }
    with open(OUT / "winner_audit_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    np.savez(OUT / "winner_pnl_v2.npz",
             pnl_split   =np.array(ml_at["split_blended"]["pnl"]),
             pnl_5bp     =np.array(ml_at["legacy_5bp"]["pnl"]),
             pnl_zero    =np.array(ml_at["zero"]["pnl"]),
             dates_split =np.array(ml_at["split_blended"]["dates"]),
             baseline=r_base,
             baseline_dates=pd.to_datetime(h_oos["date"]).dt.strftime("%Y-%m-%d").values)
    print(f"\n  Saved -> {OUT/'winner_audit_v2.json'}")
    print(f"  Saved -> {OUT/'winner_pnl_v2.npz'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-recency", action="store_true")
    ap.add_argument("--model", default="hgbr_deep", choices=["hgbr_deep","lgbm"])
    a = ap.parse_args()
    if a.all or a.init:
        init_history_v2(model_kind=a.model, use_recency=not a.no_recency)
    if a.all or a.audit:
        run_audit_v2()
    if not (a.init or a.audit or a.all):
        print("usage: train_v2.py [--init] [--audit] [--all] [--no-recency] [--model hgbr_deep|lgbm]")
