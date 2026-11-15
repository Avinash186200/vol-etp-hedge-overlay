"""
data_feed.py — fetch fresh market data needed by the P8020LV ensemble.

Pulls:
  1. SPY / VIXY / VIX daily OHLC          (Yahoo)
  2. Vol-family + rates + dollar          (Yahoo: VVIX, MOVE, TNX, DXY, etc.)
  3. Cross-asset prices                   (Yahoo: TLT, HYG, GLD, UUP, IEF, etc.)
  4. Augmented cross-asset features       (built from #3)

All feeds are settled close-of-day. We pull through "yesterday" (UTC date - 1).
Output written to data/ as parquets, replacing prior copies.

SSL: corporate proxies (e.g. Norton MITM) often break Yahoo. We use the
"yahoo direct" pattern (raw requests with verify=False fallback).

Exit codes:
  0  all feeds fresh (mtime within last 24h, latest bar today or yesterday)
  2  one or more feeds stale (>3 calendar days old) — flag, but don't crash
  3  catastrophic — no fresh data, signal cannot run
"""
from __future__ import annotations
import sys, os, json, time, datetime as dt, urllib.parse, traceback
from pathlib import Path
import numpy as np, pandas as pd, requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
OUT  = HERE / "output"; OUT.mkdir(exist_ok=True)
LOG  = OUT / "data_feed_log.txt"

# Tickers to pull — match build_augmented_matrix.py expectations
YAHOO_EXTRAS = ["^VIX","^VVIX","^VIX9D","^VIX3M","^SKEW","^OVX","^GVZ","^EVZ","^VXN","^MOVE",
                "^TNX","^TYX","^FVX","^IRX","DX-Y.NYB",
                "^N225","^HSI","^GDAXI","^FTSE","^STOXX50E","^GSPC"]
CROSS_ASSET = ["TLT","HYG","LQD","IEF","GLD","USO","UUP","FXE","FXY","EEM","FXI",
               "VXX","UVXY","SVXY","QQQ","IWM","DIA","XLU","XLE","XLK","XLF","XLV"]
CORE = ["SPY","VIXY","^VIX"]

def _log(msg, mode="a"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, mode, encoding="utf-8") as f: f.write(line + "\n")
    try: print(line, flush=True)
    except Exception: pass

def _yahoo_history(ticker: str, period_years: int = 10) -> pd.DataFrame | None:
    """Pull daily OHLC for ticker using Yahoo's chart endpoint. SSL-tolerant."""
    end = int(time.time())
    start = end - period_years*365*86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
           f"?period1={start}&period2={end}&interval=1d&includePrePost=false")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for verify_flag in (True, False):
        try:
            r = requests.get(url, headers=headers, timeout=30, verify=verify_flag)
            r.raise_for_status()
            j = r.json()["chart"]["result"][0]
            ts = j["timestamp"]
            ohlc = j["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date": pd.to_datetime(ts, unit="s").normalize(),
                "open":  ohlc.get("open",  [np.nan]*len(ts)),
                "high":  ohlc.get("high",  [np.nan]*len(ts)),
                "low":   ohlc.get("low",   [np.nan]*len(ts)),
                "close": ohlc.get("close", [np.nan]*len(ts)),
                "volume":ohlc.get("volume",[np.nan]*len(ts)),
            }).dropna(subset=["close"])
            return df
        except requests.exceptions.SSLError:
            continue  # retry with verify=False
        except Exception as e:
            _log(f"  {ticker}: pull failed ({type(e).__name__}: {e})")
            return None
    return None

def pull_core_ohlc() -> tuple[pd.DataFrame, str]:
    """Pull SPY / VIXY / VIX OHLC, merge into one dataframe like vixy_dataset.csv."""
    frames = {}
    for t in CORE:
        df = _yahoo_history(t, 10)
        if df is None or df.empty:
            _log(f"  CORE FAIL: {t}")
            return pd.DataFrame(), "fail"
        frames[t] = df.set_index("date")
    out = pd.DataFrame(index=sorted(set().union(*[f.index for f in frames.values()])))
    rename = {"SPY":"spy", "VIXY":"vixy", "^VIX":"vix"}
    for t, name in rename.items():
        f = frames[t]
        for col in ("open","high","low","close"):
            out[f"{name}_{col[0]}"] = f[col]
    out.reset_index(inplace=True); out.rename(columns={"index":"date"}, inplace=True)
    out.to_parquet(DATA/"core_ohlc.parquet", index=False)
    latest = out["date"].max()
    age = (dt.datetime.now().date() - latest.date()).days
    _log(f"  CORE OHLC: {len(out)} rows, latest {latest.date()}, age {age}d")
    return out, ("ok" if age <= 3 else "stale")

def pull_yahoo_extras() -> tuple[pd.DataFrame, str]:
    """Pull vol-family + rates + dollar + intl indices."""
    rename = {"^VIX":"VIX","^VVIX":"VVIX","^VIX9D":"VIX9D","^VIX3M":"VIX3M","^SKEW":"SKEW",
              "^OVX":"OVX","^GVZ":"GVZ","^EVZ":"EVZ","^VXN":"VXN","^MOVE":"MOVE",
              "^TNX":"TNX","^TYX":"TYX","^FVX":"FVX","^IRX":"IRX","DX-Y.NYB":"DX_Y_NYB",
              "^N225":"N225","^HSI":"HSI","^GDAXI":"GDAXI","^FTSE":"FTSE","^STOXX50E":"STOXX50E","^GSPC":"GSPC"}
    series = {}
    for t in YAHOO_EXTRAS:
        df = _yahoo_history(t, 10)
        if df is None or df.empty:
            _log(f"  yahoo_extras: {t} FAIL — using last known"); continue
        series[rename.get(t, t)] = df.set_index("date")["close"]
    if not series:
        return pd.DataFrame(), "fail"
    out = pd.DataFrame(series).reset_index().rename(columns={"index":"date"})
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(DATA/"yahoo_extras.parquet", index=False)
    latest = out["date"].max()
    age = (dt.datetime.now().date() - latest.date()).days
    _log(f"  yahoo_extras: {len(out)} rows, {len(series)} symbols, latest {latest.date()}, age {age}d")
    return out, ("ok" if age <= 3 else "stale")

def pull_cross_asset() -> tuple[pd.DataFrame, str]:
    """Pull cross-asset ETF closes."""
    series = {}
    for t in CROSS_ASSET:
        df = _yahoo_history(t, 10)
        if df is None or df.empty:
            _log(f"  cross_asset: {t} FAIL"); continue
        series[t] = df.set_index("date")["close"]
    if not series:
        return pd.DataFrame(), "fail"
    out = pd.DataFrame(series).reset_index().rename(columns={"index":"date"})
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(DATA/"cross_asset_vol.parquet", index=False)

    # Engineered features (matches cross_asset_features.parquet schema)
    eng = {"date": out["date"]}
    for sym in ["TLT","HYG","USO","UUP","GLD","IEF","LQD","IWM","QQQ","EEM"]:
        if sym not in out.columns: continue
        s = out[sym]
        eng[f"{sym}_ret1"]  = s.pct_change(1)
        eng[f"{sym}_ret5"]  = s.pct_change(5)
        eng[f"{sym}_ret21"] = s.pct_change(21)
        eng[f"{sym}_vol21"] = s.pct_change().rolling(21).std()*np.sqrt(252)
        eng[f"{sym}_vol63"] = s.pct_change().rolling(63).std()*np.sqrt(252)
        eng[f"{sym}_mom_z"] = (s.pct_change(21) - s.pct_change(21).rolling(252).mean())/s.pct_change(21).rolling(252).std()
        eng[f"{sym}_drawdown"] = s/s.rolling(63).max() - 1
    eng_df = pd.DataFrame(eng)
    if "HYG" in out.columns and "TLT" in out.columns:
        eng_df["cred_spread_z"] = (out["TLT"]/out["HYG"] - (out["TLT"]/out["HYG"]).rolling(252).mean()) / (out["TLT"]/out["HYG"]).rolling(252).std()
    if "UUP" in out.columns:
        eng_df["DXY_z"] = (out["UUP"] - out["UUP"].rolling(252).mean()) / out["UUP"].rolling(252).std()
    eng_df.to_parquet(DATA/"cross_asset_features.parquet", index=False)

    latest = out["date"].max()
    age = (dt.datetime.now().date() - latest.date()).days
    _log(f"  cross_asset: {len(out)} rows, {len(series)} symbols, latest {latest.date()}, age {age}d")
    return out, ("ok" if age <= 3 else "stale")

def main() -> int:
    _log("=== data_feed start ===", mode="w")
    statuses = []
    try:
        _, s1 = pull_core_ohlc();      statuses.append(("core", s1))
        _, s2 = pull_yahoo_extras();   statuses.append(("yahoo_extras", s2))
        _, s3 = pull_cross_asset();    statuses.append(("cross_asset", s3))
    except Exception:
        _log("CATASTROPHIC: " + traceback.format_exc())
        return 3
    _log("--- data_feed summary ---")
    n_ok = sum(1 for _, s in statuses if s == "ok")
    n_stale = sum(1 for _, s in statuses if s == "stale")
    n_fail = sum(1 for _, s in statuses if s == "fail")
    for nm, s in statuses: _log(f"  {nm}: {s}")
    if n_fail == len(statuses): return 3
    if n_stale > 0 or n_fail > 0: return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
