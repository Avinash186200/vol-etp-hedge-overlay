"""
live_daily.py — generate today's WIN11_v2 signal and write a trade ticket.

Pipeline:
  1. Build feature matrix from refreshed data/ parquets.
  2. Load 11 pickled models from models/.
  3. Predict each model's raw score for today.
  4. Append raw to history/score_history.csv (one row per model, per day).
  5. Apply A3 mechanic: EMA(7) smooth → past-only monthly percentile → average →
     clip(scale*(2*p-1), -clip, +clip).
  6. Write trade ticket to output/signal_YYYYMMDD.txt and output/signal_latest.txt.

Exit codes:
  0  signal produced
  1  models missing or data unavailable
  2  signal produced but flagged STALE (data >3 days old)
"""
from __future__ import annotations
import sys, os, json, pickle, datetime as dt, traceback
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"src"))
MODELS = HERE / "models"
DATA   = HERE / "data"
HIST   = HERE / "history"
OUT    = HERE / "output"
LOG    = OUT / "live_daily_log.txt"

CONFIG_JSON = HERE / "config.json"
SCORE_HIST  = HIST / "score_history.csv"

# Models in WIN11_v2 ensemble + which target horizon they predict
MODEL_LIST = ["v1","T2","T3","CV3","S6","CV1","FS2","Th10","AUG1","LGB_h10","AUG_h20"]
HORIZON    = {"v1":1,"T2":3,"T3":5,"CV3":1,"S6":1,"CV1":1,"FS2":1,"Th10":10,"AUG1":1,"LGB_h10":10,"AUG_h20":20}
OUTPUT_TYPE= {"v1":"regression","T2":"regression","T3":"regression","CV3":"classifier",
              "S6":"regression","CV1":"classifier","FS2":"regression","Th10":"regression",
              "AUG1":"regression","LGB_h10":"regression","AUG_h20":"regression"}

# A3 mechanic
SCALE, CLIP, SPAN = 3.5, 0.6, 7
COST_BLENDED = 9e-5  # 0.8*0.5bp + 0.2*2.5bp

# Loaded once, shared by config
def _load_config() -> dict:
    if CONFIG_JSON.exists():
        return json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    return {}

def _log(msg, mode="a"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, mode, encoding="utf-8") as f: f.write(line + "\n")
    try: print(line, flush=True)
    except Exception: pass

def _load_models() -> dict:
    """Load 11 pickled models from models/. Each pickle stores {'model':..., 'features': [...]}."""
    out = {}
    for name in MODEL_LIST:
        p = MODELS / f"{name}.pkl"
        if not p.exists():
            _log(f"  missing model: {p}"); continue
        with open(p, "rb") as f:
            out[name] = pickle.load(f)
        _log(f"  loaded {name}: {len(out[name].get('features', []))} features")
    return out

def _build_feature_matrix():
    """Use feature_build + build_augmented_matrix to produce today's input row."""
    from build_augmented_matrix import build_augmented
    m = build_augmented(verbose=False)
    d = m["d"].copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    return d, m["FSETS"]

def _ensemble_signal(score_hist: pd.DataFrame, today_date: pd.Timestamp,
                     scale=SCALE, clip=CLIP, span=SPAN) -> dict:
    """Apply A3 mechanic to score history, return today's average percentile + weight."""
    score_hist = score_hist.sort_values("date").reset_index(drop=True)
    # Smooth each model's raw column
    pcts = {}
    for m in MODEL_LIST:
        col = f"raw_{m}"
        if col not in score_hist.columns: continue
        s = pd.Series(score_hist[col].values)
        sm = s.ewm(span=span, adjust=False, min_periods=1).mean().values
        # Past-only percentile: baseline frozen at the first day of TODAY's calendar month.
        month_period = pd.PeriodIndex(score_hist["date"], freq="M")
        today_period = pd.Period(today_date, freq="M")
        fi = np.where(month_period == today_period)[0]
        if len(fi) == 0:
            continue
        fi = fi[0]   # first index in current month
        past = sm[:fi]; past = past[~np.isnan(past)]
        if len(past) < 400:
            continue
        # today's smoothed value
        today_mask = score_hist["date"] == today_date
        today_idx = np.where(today_mask)[0]
        if len(today_idx) == 0: continue
        v = sm[today_idx[0]]
        if np.isnan(v): continue
        pcts[m] = float((past < v).mean())

    if not pcts:
        return {"avg_pct": None, "raw_w": None, "w": 0.0, "members_used": []}

    avg_pct = float(np.mean(list(pcts.values())))
    raw_w = scale * (2 * avg_pct - 1)
    w = float(np.clip(raw_w, -clip, clip))
    return {"avg_pct": avg_pct, "raw_w": raw_w, "w": w,
            "members_used": list(pcts.keys()),
            "per_model_pct": pcts}

def _write_trade_ticket(date_str: str, sig: dict, today_date, capital: float = 10_000_000.0,
                        data_stale: bool = False) -> Path:
    w = sig["w"]
    direction = "LONG" if w > 0.001 else ("SHORT" if w < -0.001 else "FLAT")
    abs_w = abs(w)
    spy_pct = abs_w * 0.80 * 100
    vixy_pct = abs_w * 0.20 * 100
    cash_pct = (1 - abs_w) * 100
    sign = 1 if w > 0 else -1
    spy_dollar = sign * capital * abs_w * 0.80
    vixy_dollar = sign * capital * abs_w * 0.20

    stale_warn = "  ⚠️  DATA STALE — generated with last-known feeds\n" if data_stale else ""

    body = f"""\
=================================================================
 COMPANY  P8020LV  DAILY SIGNAL
=================================================================
{stale_warn}\
Signal date:        {date_str}
Generated at:       {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} IST
Models used:        {len(sig.get('members_used', []))} / {len(MODEL_LIST)}

ENSEMBLE OUTPUT
  avg_pct (0-1):    {sig['avg_pct']:.4f}
  raw_w:            {sig['raw_w']:+.4f}     ( = scale × (2 × avg_pct − 1) )
  w (clipped):      {w:+.4f}            ( clipped to ±{CLIP} )

POSITION INSTRUCTION
  Direction:        {direction}
  Position size:    {abs_w*100:.1f}% of capital

FOR ${capital/1e6:.0f}M ALLOCATION:
  SPY  exposure:    ${spy_dollar/1e6:+.2f}M  ({spy_pct:.1f}% {direction.lower()})
  VIXY exposure:    ${vixy_dollar/1e6:+.2f}M  ({vixy_pct:.1f}% {direction.lower()})
  Cash:             ${capital*(1-abs_w)/1e6:.2f}M  ({cash_pct:.1f}%)

EXECUTION
  Order type:       MOC (market-on-close)
  Execute at:       today's US market close (4:00 PM ET)
  Settlement:       T+1

=================================================================
"""
    out_dated = OUT / f"signal_{date_str}.txt"
    out_latest = OUT / "signal_latest.txt"
    out_dated.write_text(body, encoding="utf-8")
    out_latest.write_text(body, encoding="utf-8")

    # Also drop a machine-readable JSON
    obj = {
        "date": date_str,
        "generated_at_ist": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble": sig,
        "position": {
            "direction": direction,
            "w": w,
            "abs_pct_of_capital": abs_w * 100,
            "spy_pct": spy_pct if w >= 0 else -spy_pct,
            "vixy_pct": vixy_pct if w >= 0 else -vixy_pct,
            "cash_pct": cash_pct,
        },
        "execution": {"order_type": "MOC", "exchange_time": "US Close 4:00 PM ET"},
        "data_stale": data_stale,
    }
    (OUT/f"signal_{date_str}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    (OUT/"signal_latest.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return out_dated

def main() -> int:
    _log("=== live_daily start ===", mode="w")
    cfg = _load_config()

    # 1. Build feature matrix
    try:
        d, fsets = _build_feature_matrix()
    except Exception:
        _log("FAIL: feature matrix build")
        _log(traceback.format_exc())
        return 1

    today = d["date"].max()
    today_str = today.strftime("%Y%m%d")
    data_age = (dt.datetime.now().date() - today.date()).days
    data_stale = data_age > 3
    _log(f"matrix asof = {today.date()}, age {data_age}d, stale={data_stale}")
    _log(f"feature set ALL_AUG = {len(fsets['ALL_AUG'])} cols")

    # 2. Load models
    models = _load_models()
    if len(models) < len(MODEL_LIST):
        _log(f"WARN: only {len(models)}/{len(MODEL_LIST)} models loaded")
        if len(models) == 0: return 1

    # 3. Predict each model for the "today" row
    row = d[d["date"] == today].iloc[-1]
    raws = {}
    for name, bundle in models.items():
        try:
            feats = bundle["features"]
            X = np.nan_to_num(row[feats].astype(float).values.reshape(1, -1), nan=0.0)
            mdl = bundle["model"]
            if OUTPUT_TYPE[name] == "classifier":
                raws[name] = float(mdl.predict_proba(X)[0, 1])
            else:
                raws[name] = float(mdl.predict(X)[0])
        except Exception as e:
            _log(f"  predict {name}: FAIL ({e})")
    _log(f"got {len(raws)} raw predictions")

    # 4. Append to score history
    HIST.mkdir(exist_ok=True)
    if SCORE_HIST.exists():
        hist = pd.read_csv(SCORE_HIST, parse_dates=["date"])
    else:
        hist = pd.DataFrame({"date": []})
    new_row = {"date": today}
    for m in MODEL_LIST:
        new_row[f"raw_{m}"] = raws.get(m, np.nan)
    if (hist["date"] == today).any():
        hist.loc[hist["date"] == today, [f"raw_{m}" for m in MODEL_LIST]] = [new_row[f"raw_{m}"] for m in MODEL_LIST]
    else:
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    hist = hist.sort_values("date").reset_index(drop=True)
    hist.to_csv(SCORE_HIST, index=False)
    _log(f"score_history.csv updated: {len(hist)} rows, latest {hist['date'].max().date()}")

    # 5. Apply A3 mechanic
    sig = _ensemble_signal(hist, today, scale=SCALE, clip=CLIP, span=SPAN)
    if sig["avg_pct"] is None:
        _log("FAIL: not enough history to compute past-only percentile (<400 rows)")
        return 1
    _log(f"ensemble  avg_pct={sig['avg_pct']:.4f}  raw_w={sig['raw_w']:+.3f}  w={sig['w']:+.3f}")

    # 6. Write trade ticket
    capital = float(cfg.get("capital_usd", 10_000_000))
    ticket = _write_trade_ticket(today_str, sig, today, capital=capital, data_stale=data_stale)
    _log(f"wrote {ticket.name}")
    return 0 if not data_stale else 2

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        _log("UNHANDLED: " + traceback.format_exc())
        sys.exit(1)
