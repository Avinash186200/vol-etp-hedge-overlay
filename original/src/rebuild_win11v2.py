"""
rebuild_win11v2.py — rebuild audit JSON + Excel + decks for WIN11_v2 (Sharpe 2.41).

WIN11_v2 = ['v1','T2','T3','CV3','S6','CV1','FS2','Th10','AUG1','LGB_h10','AUG_h20']
Mechanic: same as WIN9 (scale=3.5, max_pos=0.6, span=7) — no OOS tuning
"""
import sys, json, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "../live_p7030")
from train_v2 import _stats, _per_year, COST_BLENDED

OUT = Path("output")

WIN11_V2 = ["v1","T2","T3","CV3","S6","CV1","FS2","Th10","AUG1","LGB_h10","AUG_h20"]
FM = {"v1":"winning_oos_scores.csv","T2":"score_history_T2.csv","T3":"score_history_T3.csv",
      "CV3":"score_history_CV3.csv","S6":"score_history_S6.csv","CV1":"score_history_CV1.csv",
      "FS2":"score_history_FS2.csv","Th10":"score_history_T_h10.csv","AUG1":"score_history_AUG1.csv",
      "LGB_h10":"score_history_LGB_h10.csv","AUG_h20":"score_history_AUG_h20.csv"}

loaded = {k: pd.read_csv(OUT/f, parse_dates=["date"]).dropna(subset=["raw"]) for k,f in FM.items()}
base = loaded["v1"][["date","ret"]].copy()
for k,df in loaded.items():
    base = base.merge(df[["date","raw"]].rename(columns={"raw": f"r_{k}"}), on="date", how="left")
base = base.dropna(subset=["ret"]).reset_index(drop=True)
base["date"] = pd.to_datetime(base["date"])

def build_diag(base, models, scale=3.5, clip=1.0, span=7, cost=COST_BLENDED):
    df = base[["date","ret"] + [f"r_{m}" for m in models]].copy()
    df["ym"] = df["date"].dt.to_period("M")
    smoothed = {m: pd.Series(df[f"r_{m}"].values).ewm(span=span, adjust=False, min_periods=1).mean().values for m in models}
    pcts = {m: np.full(len(df), np.nan) for m in models}
    for m_ in sorted(df["ym"].unique()):
        mask = (df["ym"]==m_).values; fi = np.where(mask)[0][0]
        if df["date"].iloc[fi] < pd.Timestamp("2021-01-01"): continue
        for k in models:
            past_sm = smoothed[k][:fi]; past_sm = past_sm[~np.isnan(past_sm)]
            if len(past_sm) < 400: continue
            for j in np.where(mask)[0]:
                pcts[k][j] = (past_sm < smoothed[k][j]).mean()
    pct_df = pd.DataFrame(pcts)
    avg_pct = pct_df.mean(axis=1).values
    raw_w = scale * (2*avg_pct - 1)
    w = np.clip(raw_w, -clip, clip); w = np.nan_to_num(w, nan=0.0)
    w_prev = np.r_[0.0, w[:-1]]
    pnl = w * df["ret"].values - np.abs(w - w_prev) * cost
    df["w"] = w; df["pnl"] = pnl
    df["live"] = df["date"] >= pd.Timestamp("2021-01-01")
    return df

diag = build_diag(base, WIN11_V2)
live = diag[diag["live"]].dropna(subset=["pnl"]).reset_index(drop=True)
pl = live["pnl"].values

sharpe = pl.mean()/pl.std()*np.sqrt(252)
cum = (1+pl).cumprod()
tot_pct = (cum[-1]-1)*100
cagr_pct = (cum[-1]**(252/len(pl))-1)*100
dd_series = cum/np.maximum.accumulate(cum) - 1
max_dd_pct = dd_series.min()*100
downside = pl[pl<0]
sortino = pl.mean()/downside.std()*np.sqrt(252) if downside.std()>0 else 0
calmar = cagr_pct/abs(max_dd_pct)
win_rate = (pl>0).mean()

# per-year
live["year"] = live["date"].dt.year
per_year_rows = []
for y, sub in live.groupby("year"):
    p = sub["pnl"].values
    cum_y = (1+p).cumprod()
    dd_y = (cum_y/np.maximum.accumulate(cum_y) - 1).min()*100
    sh_y = p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    per_year_rows.append(dict(year=int(y), n_days=len(p),
                              ret_pct=float((cum_y[-1]-1)*100),
                              sharpe=float(sh_y), max_dd_pct=float(dd_y),
                              win_rate_pct=float((p>0).mean()*100)))

# cost sensitivity
cost_rows = []
for c, lab in [(0.5e-4,"0.5bp"),(0.9e-4,"0.9bp (CURRENT)"),(1.0e-4,"1bp"),
               (2.0e-4,"2bp"),(5.0e-4,"5bp"),(1.0e-3,"10bp")]:
    diag_c = build_diag(base, WIN11_V2, cost=c)
    pl_c = diag_c[diag_c["live"]].dropna(subset=["pnl"])["pnl"].values
    sh_c = pl_c.mean()/pl_c.std()*np.sqrt(252)
    tot_c = (np.prod(1+pl_c)-1)*100
    cost_rows.append(dict(cost_label=lab, cost_bp=c*1e4, sharpe=float(sh_c), tot_pct=float(tot_c)))

# solo sharpes (use A3 mechanic)
def f_ema(span): return lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean()
def a3_solo(raw, ret_arr, dates_arr, scale=3.5, max_pos=0.6, span=7):
    df = pd.DataFrame({"date": pd.to_datetime(dates_arr), "raw": raw, "ret": ret_arr})
    df["sm"] = pd.Series(raw).ewm(span=span, adjust=False, min_periods=1).mean().values
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
    live_m = (df["date"] >= pd.Timestamp("2021-01-01")).values
    return pnl[live_m]

solo = {}
for m in WIN11_V2:
    sub = loaded[m].dropna(subset=["raw","ret"]).reset_index(drop=True)
    p = a3_solo(sub["raw"].values, sub["ret"].values, sub["date"].values)
    p = p[~np.isnan(p)]
    solo[m] = float(p.mean()/p.std()*np.sqrt(252)) if p.std()>0 else 0.0

# member dropout
contrib = {}
for drop in WIN11_V2:
    diag_d = build_diag(base, [m for m in WIN11_V2 if m != drop])
    pl_d = diag_d[diag_d["live"]].dropna(subset=["pnl"])["pnl"].values
    sh_d = pl_d.mean()/pl_d.std()*np.sqrt(252) if pl_d.std()>0 else 0
    contrib[drop] = float(sharpe - sh_d)

print(f"WIN11_v2 Sharpe = {sharpe:+.3f}")
print(f"Per-year: " + " | ".join(f"{r['year']}:{r['ret_pct']:+.1f}%" for r in per_year_rows))

audit_out = dict(
    ensemble="WIN11_v2",
    models=WIN11_V2,
    mechanic=dict(scale=3.5, max_pos=1.0, span=7, cost_bp=COST_BLENDED*1e4),
    headline=dict(sharpe=round(float(sharpe),3),
                  sortino=round(float(sortino),3),
                  calmar=round(float(calmar),3),
                  cagr_pct=round(float(cagr_pct),2),
                  tot_pct=round(float(tot_pct),1),
                  max_dd_pct=round(float(max_dd_pct),2),
                  win_rate=float(win_rate),
                  n_days=int(len(pl)),
                  live_window=["2021-01-01", str(live["date"].max().date())]),
    solo_sharpes={m: round(s,3) for m,s in solo.items()},
    member_contributions={m: round(c,3) for m,c in contrib.items()},
    per_year=per_year_rows,
    cost_sensitivity=cost_rows,
    top_drawdowns=[],
)
with open(OUT/"audit_win11_v2_summary.json","w") as f:
    json.dump(audit_out, f, indent=2, default=str)
print(f"\nSaved -> {OUT/'audit_win11_v2_summary.json'}")
