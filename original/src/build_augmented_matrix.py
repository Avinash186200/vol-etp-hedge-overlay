"""
build_augmented_matrix.py — base feature_build + extra cross-asset/vol features.

Adds (all shift(1) for 3pm-safety):
  - yahoo_extras: VVIX, VIX9D, VIX3M, SKEW, OVX, GVZ, EVZ, VXN, MOVE,
                  TNX, TYX, FVX, IRX, DXY, plus international indices
  - cross_asset_features: TLT/HYG/USO/UUP/GLD return/vol/momentum/drawdown features

Engineered transforms for each new vol-like series:
  - level, 1d/5d/21d returns, 21d/63d realized vol z-score, momentum z, drawdown
  - vol-ratios where they make sense (VIX/VIX3M term structure, VVIX/VIX, MOVE/VIX)
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STAGED = ROOT / "staged_finnhub"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STAGED / "live_p7030"))

from feature_build import build_matrix

YE_PARQ = STAGED / "yahoo_extras.parquet"
CA_PARQ = STAGED / "cross_asset_features.parquet"
CV_PARQ = STAGED / "cross_asset_vol.parquet"

def _eng(s, prefix):
    """Build standard time-series features for one price/level series."""
    feats = {}
    feats[f"{prefix}"]        = s
    feats[f"{prefix}_chg1"]   = s.pct_change(1)
    feats[f"{prefix}_chg5"]   = s.pct_change(5)
    feats[f"{prefix}_chg21"]  = s.pct_change(21)
    feats[f"{prefix}_z21"]    = (s - s.rolling(21).mean()) / s.rolling(21).std()
    feats[f"{prefix}_z63"]    = (s - s.rolling(63).mean()) / s.rolling(63).std()
    feats[f"{prefix}_rv21"]   = s.pct_change().rolling(21).std() * np.sqrt(252)
    feats[f"{prefix}_dd"]     = s / s.rolling(63).max() - 1.0
    return feats

def build_augmented(verbose=False):
    m = build_matrix(verbose=verbose)
    d = m["d"].copy().sort_values("date").reset_index(drop=True)
    base_cols = set(d.columns)

    # === yahoo_extras (vol family + rates + dollar + intl) ===
    ye = pd.read_parquet(YE_PARQ)
    ye["date"] = pd.to_datetime(ye["date"])

    KEEP_LEVELS = ["VVIX","VIX9D","VIX3M","SKEW","OVX","GVZ","EVZ","VXN","MOVE",
                   "TNX","TYX","FVX","IRX","DX_Y_NYB"]
    KEEP_INTL   = ["N225","HSI","GDAXI","FTSE","STOXX50E"]
    new_feats = {}
    for col in KEEP_LEVELS + KEEP_INTL:
        if col not in ye.columns: continue
        s = ye.set_index("date")[col].sort_index().reindex(d["date"].values)
        # forward-fill weekend/holiday gaps then shift(1) for 3pm safety
        s = s.ffill(limit=4)
        for k, v in _eng(s, col).items():
            new_feats[k] = v.shift(1).values
    aux = pd.DataFrame(new_feats)

    # ratios (after levels exist) — these come from already-shifted columns
    def col(name):
        return aux[name] if name in aux.columns else None
    if col("VIX3M") is not None:
        # use yahoo VIX too for clean term structure (already shifted)
        ye_vix = ye.set_index("date")["VIX"].reindex(d["date"].values).ffill(limit=4).shift(1)
        aux["VIX_VIX3M_ts"]  = ye_vix.values / aux["VIX3M"].values
        aux["VIX9D_VIX_ts"]  = aux["VIX9D"].values / ye_vix.values
        aux["VVIX_VIX"]      = aux["VVIX"].values / ye_vix.values
        aux["MOVE_VIX"]      = aux["MOVE"].values / ye_vix.values
        aux["SKEW_z"]        = (aux["SKEW"].values - 100.0) / 25.0
    YE_NEW = list(aux.columns)
    for c in YE_NEW: d[c] = aux[c].values

    # === cross_asset_features.parquet (engineered TLT/HYG/USO/UUP/GLD/IWM/EEM features) ===
    ca = pd.read_parquet(CA_PARQ)
    ca["date"] = pd.to_datetime(ca["date"])
    ca_feats = [c for c in ca.columns if c != "date"]
    ca_m = ca.set_index("date").reindex(d["date"].values).ffill(limit=4).reset_index(drop=True)
    CA_NEW = []
    for c in ca_feats:
        ncol = f"CA_{c}"
        d[ncol] = ca_m[c].shift(1).values
        CA_NEW.append(ncol)

    # === cross_asset_vol.parquet (level series for HYG/LQD/TLT/IEF/GLD/USO/UUP/MOVE_synth/sectors) ===
    cv = pd.read_parquet(CV_PARQ)
    cv["date"] = pd.to_datetime(cv["date"])
    KEEP_CV = ["HYG","LQD","TLT","IEF","GLD","USO","UUP","FXE","FXY","EEM","FXI",
               "VXX","UVXY","SVXY","EVZ","MOVE_synth","QQQ","IWM","DIA",
               "XLU","XLE","XLK","XLF","XLV"]
    cv_m = cv.set_index("date")[[c for c in KEEP_CV if c in cv.columns]].sort_index()
    cv_m = cv_m.reindex(d["date"].values).ffill(limit=4)
    CV_NEW = []
    new_cv = {}
    for col_name in cv_m.columns:
        s = cv_m[col_name]
        for k, v in _eng(s, f"CV_{col_name}").items():
            new_cv[k] = v.shift(1).values
            CV_NEW.append(k)
    cv_df = pd.DataFrame(new_cv)
    for c in cv_df.columns:
        if c not in d.columns:
            d[c] = cv_df[c].values

    # combined new feature list
    ALL_NEW = [c for c in (YE_NEW + CA_NEW + CV_NEW) if c in d.columns]
    ALL_NEW = list(dict.fromkeys(ALL_NEW))  # dedupe order-preserving

    FSETS = dict(m["FSETS"])
    FSETS["ALL_AUG"] = list(dict.fromkeys(FSETS["ALL"] + ALL_NEW))
    FSETS["NEW_ONLY"] = ALL_NEW

    if verbose:
        print(f"[augmented] base ALL: {len(FSETS['ALL'])}")
        print(f"[augmented] new features added: {len(ALL_NEW)}")
        print(f"[augmented] ALL_AUG total: {len(FSETS['ALL_AUG'])}")

    return dict(d=d, FSETS=FSETS, books=m["books"], asof=m["asof"])

if __name__ == "__main__":
    m = build_augmented(verbose=True)
    d = m["d"]
    print(f"\nMatrix shape: {d.shape}")
    print(f"Date range: {d['date'].min()} -> {d['date'].max()}")
    # NaN audit on new features at the most recent dates
    new = m["FSETS"]["NEW_ONLY"]
    nan_pct = d[new].isna().mean().sort_values(ascending=False)
    print(f"\nWorst 10 NaN pct (in new features, full sample):")
    print(nan_pct.head(10))
    # post-2019 (the live window)
    d19 = d[d["date"] >= "2019-01-01"]
    nan_19 = d19[new].isna().mean().sort_values(ascending=False)
    print(f"\nWorst 10 NaN pct (in new features, post-2019):")
    print(nan_19.head(10))
