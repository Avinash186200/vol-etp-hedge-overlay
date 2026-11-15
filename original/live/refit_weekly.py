"""
refit_weekly.py — train all 11 WIN11_v2 models on the full available history,
pickle them into models/. Intended to run once per week (Tuesday IST = Monday ET).

Each model uses the same hyperparams from training (HGBR-deep / LGBM-deep).
Embargo = 2 days (train cutoff = today - 3 days).

Runtime: ~30 min full refit on a modern laptop. The daily script just loads
the pickles (no training), so daily run stays under 1 min.

Models trained:
  v1, T2, T3, CV3, S6, CV1, FS2, Th10, AUG1, LGB_h10, AUG_h20
"""
from __future__ import annotations
import sys, os, pickle, time, datetime as dt, traceback
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"src"))
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)
OUT    = HERE / "output"; OUT.mkdir(exist_ok=True)
LOG    = OUT / "refit_log.txt"

EMB = 2  # 2-day embargo
WARMUP_END = "2019-01-01"  # earliest target start

def _log(msg, mode="a"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, mode, encoding="utf-8") as f: f.write(line + "\n")
    try: print(line, flush=True)
    except Exception: pass

def main() -> int:
    _log("=== refit_weekly start ===", mode="w")
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import lightgbm as lgb
    from build_augmented_matrix import build_augmented

    # 1. Build feature matrix
    m = build_augmented(verbose=False)
    d = m["d"].copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    base_feats = [c for c in m["FSETS"]["ALL"] if c in d.columns]
    aug_feats  = [c for c in m["FSETS"]["ALL_AUG"] if c in d.columns]
    new_feats  = [c for c in m["FSETS"].get("NEW_ONLY", []) if c in d.columns]
    _log(f"matrix asof = {d['date'].max().date()}, base {len(base_feats)}, aug {len(aug_feats)}")

    # 2. Build targets for h ∈ {1,3,5,10,20}
    for h in (1,3,5,10,20):
        spy_h = d["spy"].shift(-h)/d["spy"] - 1
        vix_h = d["vixy"].shift(-h)/d["vixy"] - 1
        d[f"ret_t{h}"] = 0.80 * spy_h + 0.20 * vix_h
    d["bin_t1"] = (d["ret_t1"] > 0).astype(int)

    # 3. Cutoff: today - 3 days (EMB embargo)
    cutoff = d["date"].max() - pd.Timedelta(days=EMB+1)
    train = d[d["date"] <= cutoff].copy()
    _log(f"train cutoff = {cutoff.date()}, train rows {len(train)}")

    # 4. Define models
    def hgbr_deep(seed=42):
        return HistGradientBoostingRegressor(
            max_iter=500, max_depth=7, learning_rate=0.02,
            min_samples_leaf=30, l2_regularization=0.5, random_state=seed)
    def hgbc_iso(seed=42):
        return CalibratedClassifierCV(HistGradientBoostingClassifier(
            max_iter=500, max_depth=7, learning_rate=0.02,
            min_samples_leaf=30, l2_regularization=0.5, random_state=seed),
            method="isotonic", cv=3)
    def lgb_deep_cls(seed=42):
        return lgb.LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.02,
            num_leaves=63, min_child_samples=30, reg_lambda=0.5,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            random_state=seed, verbosity=-1, n_jobs=-1)
    def lgb_deep_reg(seed=42):
        return lgb.LGBMRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.02,
            num_leaves=63, min_child_samples=30, reg_lambda=0.5,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            random_state=seed, verbosity=-1, n_jobs=-1)

    # Top-50 stable-IC for S6: use spearman rank correlation with ret_t1 over pre-2019 data
    from scipy.stats import spearmanr
    pre = train[train["date"] <= "2018-12-31"].dropna(subset=["ret_t1"])
    ic = {}
    for f in base_feats:
        try: ic[f] = abs(spearmanr(pre[f].fillna(0), pre["ret_t1"]).correlation)
        except Exception: ic[f] = 0
    top50 = [f for f,_ in sorted(ic.items(), key=lambda x: -x[1])[:50]]

    # FS2 features: EMA-5 smoothed base features
    fs2_feats = []
    for f in base_feats[:120]:  # cap to avoid blowup
        d[f"{f}_ema5"] = d[f].ewm(span=5, adjust=False, min_periods=1).mean()
        fs2_feats.append(f"{f}_ema5")

    MODELS_SPEC = [
        ("v1",      "regression",  base_feats, "ret_t1",  hgbr_deep),
        ("T2",      "regression",  base_feats, "ret_t3",  hgbr_deep),
        ("T3",      "regression",  base_feats, "ret_t5",  hgbr_deep),
        ("Th10",    "regression",  base_feats, "ret_t10", hgbr_deep),
        ("S6",      "regression",  top50,      "ret_t1",  hgbr_deep),
        ("FS2",     "regression",  fs2_feats,  "ret_t1",  hgbr_deep),
        ("AUG1",    "regression",  aug_feats,  "ret_t1",  hgbr_deep),
        ("LGB_h10", "regression",  aug_feats,  "ret_t10", lgb_deep_reg),
        ("AUG_h20", "regression",  aug_feats,  "ret_t20", hgbr_deep),
        ("CV1",     "classifier",  base_feats, "bin_t1",  hgbc_iso),
        ("CV3",     "classifier",  base_feats, "bin_t1",  lgb_deep_cls),
    ]

    train_clean = train.replace([np.inf, -np.inf], np.nan)
    t0_total = time.time()
    for name, kind, feats, target_col, factory in MODELS_SPEC:
        _log(f"--- training {name} ({kind}, |feats|={len(feats)}, target={target_col}) ---")
        t0 = time.time()
        try:
            g = train_clean.dropna(subset=[target_col])
            X = g[feats].fillna(0.0).values
            y = g[target_col].values
            mdl = factory()
            mdl.fit(X, y)
            bundle = {"model": mdl, "features": feats, "kind": kind,
                      "target": target_col, "trained_at": dt.datetime.now().isoformat(),
                      "train_rows": int(len(g)), "train_cutoff": str(cutoff.date())}
            with open(MODELS / f"{name}.pkl", "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            _log(f"  {name} done in {(time.time()-t0)/60:.1f} min — saved {name}.pkl")
        except Exception as e:
            _log(f"  {name} FAILED: {e}")
            _log(traceback.format_exc())

    _log(f"=== refit_weekly DONE in {(time.time()-t0_total)/60:.1f} min ===")
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception:
        _log("UNHANDLED: " + traceback.format_exc())
        sys.exit(1)
