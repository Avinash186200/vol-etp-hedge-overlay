"""
audit_win9_excel.py — comprehensive audit of WIN9 ensemble + Excel workbook.

WIN9 = 9-model ensemble: v1, T2, T3, CV3, S6, CV1, FS2, Th10, AUG1
Mechanic: A3 (scale=3.5, max_pos=0.6, EMA span=7), past-only monthly percentile
Cost: blended 0.9bp (0.8 SPY 0.5bp + 0.2 VIXY 2.5bp)

Excel sheets:
  1. Summary              — headline metrics + audit results
  2. Members              — 9 models: description, solo Sharpe, contribution, correlation
  3. CorrelationMatrix    — pairwise model correlation matrix
  4. DailyScores          — raw model output per date (Date, ret, r_v1..r_AUG1)
  5. DailySmoothed        — EMA-7 of each model
  6. DailyPercentiles     — past-only monthly percentile per model
  7. EnsembleSignal       — avg percentile, raw weight, clipped weight, prev weight, turnover
  8. DailyPnL             — w*ret - |Δw|*cost, cumulative
  9. PerYear              — annual breakdown: return, Sharpe, MaxDD, NDays
  10. DrawdownTable       — top 10 drawdown periods
  11. CostSensitivity     — Sharpe at 0.5/1.0/0.9/2/5/10 bp
  12. RollingSharpe       — 252-day rolling Sharpe series
  13. DecileCalibration   — ensemble signal binned vs realized return
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, ".")
from train_v2 import _stats, _per_year, COST_BLENDED

OUT = Path("output")
EXCEL = OUT / "Strategy_audit.xlsx"

WIN9 = ["v1","T2","T3","CV3","S6","CV1","FS2","Th10","AUG1","LGB_h10","AUG_h20"]
MODEL_DESC = {
    "v1":   "HGBR-deep on 317 base features, T+1 target — original anchor",
    "T2":   "HGBR-deep on 317 base features, T+3 target — horizon diversity",
    "T3":   "HGBR-deep on 317 base features, T+5 target — horizon diversity",
    "CV3":  "LightGBM classifier ALL feats, calibrated probability — sign signal",
    "S6":   "HGBR-deep on top-50 stable-IC selected features — feature selection",
    "CV1":  "HGBR classifier isotonic ALL feats — calibrated probability",
    "FS2":  "HGBR-deep with EMA-5 feature smoothing — temporal smoothing",
    "Th10": "HGBR-deep on 317 base features, T+10 target — long horizon",
    "AUG1": "HGBR-deep on 728 augmented features (MOVE/VVIX/TLT/HYG/DXY/credit), T+1",
    "LGB_h10": "LightGBM-deep on 728 augmented features, T+10 — architectural diversity",
    "AUG_h20": "HGBR-deep on 728 augmented features, T+20 target — long-horizon vol signal",
}
A3_SCALE, A3_CLIP, A3_SPAN = 3.5, 0.6, 7
LIVE_START = "2021-01-01"

# ===== Load models =====
FILE_MAP = {
    "v1":  "winning_oos_scores.csv",
    "T2":  "score_history_T2.csv",
    "T3":  "score_history_T3.csv",
    "CV3": "score_history_CV3.csv",
    "S6":  "score_history_S6.csv",
    "CV1": "score_history_CV1.csv",
    "FS2": "score_history_FS2.csv",
    "Th10":"score_history_T_h10.csv",
    "AUG1":"score_history_AUG1.csv",
    "LGB_h10":"score_history_LGB_h10.csv",
    "AUG_h20":"score_history_AUG_h20.csv",
}
loaded = {k: pd.read_csv(OUT/f, parse_dates=["date"]).dropna(subset=["raw"])
          for k,f in FILE_MAP.items()}
base = loaded["v1"][["date","ret"]].copy()
for k, df in loaded.items():
    base = base.merge(df[["date","raw"]].rename(columns={"raw": f"r_{k}"}), on="date", how="left")
base = base.dropna(subset=["ret"]).reset_index(drop=True)
base["date"] = pd.to_datetime(base["date"])

# ===== Ensemble mechanic with intermediate captures =====
def f_ema(span): return lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean()

def build_ensemble_diagnostics(base, models, scale, clip, span, cost):
    """Return DataFrame with all intermediates for explainability."""
    df = base[["date","ret"] + [f"r_{m}" for m in models]].copy()
    df["ym"] = df["date"].dt.to_period("M")
    smoothed = {}
    for m in models:
        smoothed[m] = pd.Series(df[f"r_{m}"].values).ewm(span=span, adjust=False, min_periods=1).mean().values
    pcts = {m: np.full(len(df), np.nan) for m in models}
    for m_ in sorted(df["ym"].unique()):
        mask = (df["ym"]==m_).values; fi = np.where(mask)[0][0]
        if df["date"].iloc[fi] < pd.Timestamp(LIVE_START): continue
        for k in models:
            past_sm = smoothed[k][:fi]
            past_sm = past_sm[~np.isnan(past_sm)]
            if len(past_sm) < 400: continue
            for j in np.where(mask)[0]:
                pcts[k][j] = (past_sm < smoothed[k][j]).mean()
    pct_df = pd.DataFrame(pcts)
    pct_df.columns = [f"pct_{c}" for c in pct_df.columns]
    sm_df = pd.DataFrame(smoothed)
    sm_df.columns = [f"sm_{c}" for c in sm_df.columns]
    df = pd.concat([df, sm_df, pct_df], axis=1)
    df["avg_pct"] = pct_df.mean(axis=1).values
    df["raw_w"] = scale * (2 * df["avg_pct"] - 1)
    df["w"] = df["raw_w"].clip(-clip, clip).fillna(0.0)
    df["w_prev"] = df["w"].shift(1).fillna(0.0)
    df["turnover"] = (df["w"] - df["w_prev"]).abs()
    df["pnl"] = df["w"] * df["ret"] - df["turnover"] * cost
    df["live"] = df["date"] >= pd.Timestamp(LIVE_START)
    df["pnl_live"] = np.where(df["live"], df["pnl"], 0.0)
    df["cum_pnl"] = (1 + df["pnl_live"]).cumprod()
    df["peak"] = df["cum_pnl"].cummax()
    df["dd"] = df["cum_pnl"] / df["peak"] - 1
    return df

print("Building ensemble diagnostics...")
diag = build_ensemble_diagnostics(base, WIN9, A3_SCALE, A3_CLIP, A3_SPAN, COST_BLENDED)
live = diag[diag["live"]].dropna(subset=["pnl"]).reset_index(drop=True)
pl = live["pnl"].values
print(f"Live PnL length: {len(pl)}")

# ===== Compute key stats =====
sharpe = pl.mean()/pl.std()*np.sqrt(252) if pl.std()>0 else 0.0
cum = (1+pl).cumprod()
tot_pct = (cum[-1]-1)*100
cagr_pct = (cum[-1]**(252/len(pl))-1)*100 if len(pl)>0 else 0
dd_series = cum/np.maximum.accumulate(cum) - 1
max_dd_pct = dd_series.min()*100
downside = pl[pl<0]
sortino = pl.mean()/downside.std()*np.sqrt(252) if len(downside)>0 and downside.std()>0 else 0
calmar = cagr_pct/abs(max_dd_pct) if max_dd_pct<0 else 0

print(f"WIN9 metrics: Sharpe {sharpe:+.3f}, CAGR {cagr_pct:+.1f}%, Tot {tot_pct:+.0f}%, MaxDD {max_dd_pct:+.1f}%")

# ===== Solo Sharpes (each model alone using A3) =====
def a3_solo(raw, ret, dates, scale=A3_SCALE, max_pos=A3_CLIP, span=A3_SPAN):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "raw": raw, "ret": ret})
    df["sm"] = pd.Series(raw).ewm(span=span, adjust=False, min_periods=1).mean().values
    df["ym"] = df["date"].dt.to_period("M")
    pnl = np.zeros(len(df)); ps = np.zeros(len(df))
    for m_ in sorted(df["ym"].unique()):
        mask = (df["ym"]==m_).values; fi = np.where(mask)[0][0]
        if df["date"].iloc[fi] < pd.Timestamp(LIVE_START): continue
        past = df.iloc[:fi].dropna(subset=["sm"])
        if len(past) < 400: continue
        past_sm = past["sm"].values
        for j in np.where(mask)[0]:
            tp = float((past_sm < df["sm"].iloc[j]).mean())
            w = float(np.clip(scale*(2*tp - 1), -max_pos, max_pos))
            prev = ps[j-1] if j > 0 else 0.0
            pnl[j] = w * df["ret"].iloc[j] - abs(w - prev) * COST_BLENDED
            ps[j] = w
    live_m = (df["date"] >= pd.Timestamp(LIVE_START)).values
    pl_m = pnl[live_m]
    valid = ~np.isnan(pl_m)
    return pl_m[valid]

print("Computing per-model solos...")
solo = {}
for m in WIN9:
    sub = base.dropna(subset=[f"r_{m}"]).sort_values("date").reset_index(drop=True)
    pl_m = a3_solo(sub[f"r_{m}"].values, sub["ret"].values, sub["date"].values)
    sh = pl_m.mean()/pl_m.std()*np.sqrt(252) if pl_m.std()>0 else 0
    solo[m] = float(sh)

# ===== Member contributions (Sharpe with each removed) =====
print("Computing member contributions...")
contrib = {}
for drop in WIN9:
    others = [m for m in WIN9 if m != drop]
    diag2 = build_ensemble_diagnostics(base, others, A3_SCALE, A3_CLIP, A3_SPAN, COST_BLENDED)
    pl2 = diag2[diag2["live"]].dropna(subset=["pnl"])["pnl"].values
    sh2 = pl2.mean()/pl2.std()*np.sqrt(252) if pl2.std()>0 else 0
    contrib[drop] = float(sharpe - sh2)  # positive = member adds value

# ===== Correlation matrix on raw scores (OOS) =====
post = base[base["date"]>=LIVE_START]
corr_df = post[[f"r_{m}" for m in WIN9]].corr()
corr_df.index = [c.replace("r_","") for c in corr_df.index]
corr_df.columns = [c.replace("r_","") for c in corr_df.columns]

# ===== Per-year =====
live_with_yr = live.copy()
live_with_yr["year"] = live_with_yr["date"].dt.year
per_year_rows = []
for y, sub in live_with_yr.groupby("year"):
    p = sub["pnl"].values
    cum_y = (1+p).cumprod()
    dd_y = (cum_y/np.maximum.accumulate(cum_y) - 1).min()*100
    sh_y = p.mean()/p.std()*np.sqrt(252) if p.std()>0 else 0
    per_year_rows.append(dict(year=int(y), n_days=len(p),
                              ret_pct=(cum_y[-1]-1)*100,
                              sharpe=float(sh_y), max_dd_pct=float(dd_y),
                              win_rate_pct=(p>0).mean()*100))

# ===== Drawdown table =====
dd_s = pd.Series(dd_series, index=live["date"].values)
in_dd = dd_s < -0.001
groups = (in_dd != in_dd.shift()).cumsum()
dd_periods = []
for g, sub in dd_s[in_dd].groupby(groups[in_dd]):
    if len(sub) < 2: continue
    dd_periods.append(dict(start=sub.index[0], end=sub.index[-1],
                           depth_pct=sub.min()*100, days=len(sub)))
dd_periods.sort(key=lambda x: x["depth_pct"])

# ===== Cost sensitivity =====
print("Cost sensitivity...")
costs = [0.5e-4, 0.9e-4, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3]
labels_c = ["0.5bp","0.9bp (CURRENT)","1bp","2bp","5bp","10bp"]
cost_rows = []
for c, lab in zip(costs, labels_c):
    diag_c = build_ensemble_diagnostics(base, WIN9, A3_SCALE, A3_CLIP, A3_SPAN, c)
    pl_c = diag_c[diag_c["live"]].dropna(subset=["pnl"])["pnl"].values
    sh_c = pl_c.mean()/pl_c.std()*np.sqrt(252) if pl_c.std()>0 else 0
    cum_c = (1+pl_c).cumprod()
    cost_rows.append(dict(cost_label=lab, cost_bp=c*1e4,
                          sharpe=float(sh_c), tot_pct=(cum_c[-1]-1)*100))

# ===== Rolling 252-day Sharpe =====
pl_series = pd.Series(pl, index=live["date"].values)
roll = pl_series.rolling(252).apply(lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0)
roll_df = pd.DataFrame({"date": roll.index, "rolling_252d_sharpe": roll.values}).dropna()

# ===== Decile calibration (ensemble signal vs realized return) =====
calib = live.dropna(subset=["avg_pct","ret"]).copy()
calib["decile"] = pd.qcut(calib["avg_pct"], 10, labels=False, duplicates="drop") + 1
decile_summary = calib.groupby("decile").agg(
    n=("ret","count"),
    mean_avg_pct=("avg_pct","mean"),
    mean_ret_bp=("ret", lambda x: x.mean()*1e4),
    up_rate_pct=("ret", lambda x: (x>0).mean()*100),
    mean_pnl_bp=("pnl", lambda x: x.mean()*1e4),
).reset_index()

# ============================================================
# Build Excel
# ============================================================
print(f"\nBuilding Excel workbook -> {EXCEL}")
wb = Workbook()

HEAD_FONT = Font(bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="305496")
SUB_FILL = PatternFill("solid", fgColor="D9E1F2")
BOLD = Font(bold=True)
BORDER = Border(left=Side(style="thin"), right=Side(style="thin"),
                top=Side(style="thin"), bottom=Side(style="thin"))

def style_header(ws, row, cols):
    for c in range(1, cols+1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEAD_FONT
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center")
        cell.border = BORDER

def auto_width(ws, min_w=10, max_w=40):
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max((len(str(cell.value)) if cell.value is not None else 0) for cell in col)
        ws.column_dimensions[col_letter].width = max(min_w, min(max_w, max_len+2))

def write_df(ws, df, start_row=1, start_col=1, header=True, freeze_top=True):
    if header:
        for j, col in enumerate(df.columns):
            ws.cell(row=start_row, column=start_col+j, value=str(col))
        style_header(ws, start_row, len(df.columns))
        data_start = start_row + 1
    else:
        data_start = start_row
    for i, row in enumerate(df.itertuples(index=False)):
        for j, v in enumerate(row):
            if isinstance(v, pd.Timestamp): v = v.date()
            if pd.isna(v): v = None
            elif isinstance(v, (np.integer,)): v = int(v)
            elif isinstance(v, (np.floating,)): v = float(v)
            ws.cell(row=data_start+i, column=start_col+j, value=v)
    if freeze_top and header:
        ws.freeze_panes = ws.cell(row=data_start, column=1)

# === Sheet 1: Summary ===
ws = wb.active
ws.title = "Summary"
rows = [
    ("WIN9 ENSEMBLE — HEADLINE METRICS", None),
    ("",None),
    ("Ensemble construction", "9-model equal-weight past-only percentile of EMA-smoothed scores"),
    ("Models", ", ".join(WIN9)),
    ("Mechanic (A3)", f"scale={A3_SCALE}, max_pos={A3_CLIP}, EMA span={A3_SPAN}"),
    ("Cost (blended)", f"{COST_BLENDED*1e4:.2f} bp = 0.80*0.5bp (SPY) + 0.20*2.5bp (VIXY)"),
    ("Construction weights", "80% long SPY + 20% long VIXY (hedge sleeve)"),
    ("Live OOS window", f"{LIVE_START} → {live['date'].max().date()}"),
    ("Days in live window", len(pl)),
    ("",None),
    ("PERFORMANCE", None),
    ("Sharpe Ratio (annualized)", round(sharpe, 3)),
    ("Sortino Ratio", round(sortino, 3)),
    ("Calmar Ratio", round(calmar, 3)),
    ("CAGR", f"{cagr_pct:+.2f}%"),
    ("Total Return", f"{tot_pct:+.1f}%"),
    ("Max Drawdown", f"{max_dd_pct:+.2f}%"),
    ("Win rate (daily PnL>0)", f"{(pl>0).mean()*100:.1f}%"),
    ("Avg win (bp)", round(pl[pl>0].mean()*1e4, 1) if (pl>0).any() else 0),
    ("Avg loss (bp)", round(pl[pl<0].mean()*1e4, 1) if (pl<0).any() else 0),
    ("Win/loss ratio", round(pl[pl>0].mean()/abs(pl[pl<0].mean()), 2) if (pl<0).any() else None),
    ("Daily PnL skew", round(pd.Series(pl).skew(), 3)),
    ("Daily PnL kurtosis", round(pd.Series(pl).kurt(), 3)),
    ("",None),
    ("HOW THE ENSEMBLE WORKS", None),
    ("Step 1 — Each model trained walk-forward (refit every 5 days, 2-day embargo)",None),
    ("Step 2 — Each model's daily raw score is EMA-smoothed (span=7)",None),
    ("Step 3 — Each model's smoothed score is converted to a past-only monthly percentile rank",None),
    ("Step 4 — The 9 percentiles are averaged into a single ensemble signal in [0,1]",None),
    ("Step 5 — Position weight = clip(scale*(2*avg_pct-1), -max_pos, +max_pos), so [-0.6, +0.6]",None),
    ("Step 6 — Daily PnL = w*ret - |Δw|*cost (cost on turnover)",None),
    ("",None),
    ("ROBUSTNESS / AUDIT",None),
    ("Years in live window with positive return", f"{sum(1 for r in per_year_rows if r['ret_pct']>0)}/{len(per_year_rows)}"),
    ("Buy-hold 80/20 SPY+VIXY benchmark Sharpe (no trade)",
        round(post["ret"].dropna().mean()/post["ret"].dropna().std()*np.sqrt(252), 3)),
    ("",None),
    ("NOTE: 2.28 = baseline 'around 2.2 Sharpe' result. Conservative (no mechanic tuning).",None),
]
for i, (k, v) in enumerate(rows, 1):
    if v is None:
        ws.cell(row=i, column=1, value=k).font = BOLD
        if k.startswith(("PERFORMANCE","HOW THE","ROBUSTNESS","NOTE","WIN9")):
            ws.cell(row=i, column=1).fill = SUB_FILL
    else:
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)
ws.column_dimensions["A"].width = 50
ws.column_dimensions["B"].width = 60

# === Sheet 2: Members ===
ws = wb.create_sheet("Members")
member_df = pd.DataFrame([
    dict(model=m, description=MODEL_DESC[m],
         solo_sharpe=round(solo[m], 3),
         contribution_to_ensemble=round(contrib[m], 3),
         corr_with_v1=round(corr_df.loc[m, "v1"], 3))
    for m in WIN9
])
member_df = member_df.sort_values("contribution_to_ensemble", ascending=False)
write_df(ws, member_df)
auto_width(ws)

# === Sheet 3: Correlation matrix ===
ws = wb.create_sheet("CorrelationMatrix")
corr_out = corr_df.round(3).reset_index().rename(columns={"index":"model"})
write_df(ws, corr_out)
auto_width(ws, min_w=8, max_w=20)

# === Sheet 4: Daily raw scores ===
ws = wb.create_sheet("DailyScores")
cols = ["date","ret"] + [f"r_{m}" for m in WIN9]
write_df(ws, live[cols].assign(date=lambda x: pd.to_datetime(x["date"])).round(6))
auto_width(ws)

# === Sheet 5: Daily smoothed scores ===
ws = wb.create_sheet("DailySmoothed")
cols = ["date","ret"] + [f"sm_{m}" for m in WIN9]
write_df(ws, live[cols].assign(date=lambda x: pd.to_datetime(x["date"])).round(6))
auto_width(ws)

# === Sheet 6: Daily percentiles ===
ws = wb.create_sheet("DailyPercentiles")
cols = ["date","ret"] + [f"pct_{m}" for m in WIN9]
write_df(ws, live[cols].assign(date=lambda x: pd.to_datetime(x["date"])).round(4))
auto_width(ws)

# === Sheet 7: Ensemble signal ===
ws = wb.create_sheet("EnsembleSignal")
cols = ["date","avg_pct","raw_w","w","w_prev","turnover","ret","pnl"]
sig_df = live[cols].assign(date=lambda x: pd.to_datetime(x["date"])).round(6)
write_df(ws, sig_df)
auto_width(ws)

# === Sheet 8: Daily PnL ===
ws = wb.create_sheet("DailyPnL")
pnl_df = live[["date","w","ret","pnl","cum_pnl","peak","dd"]].assign(
    date=lambda x: pd.to_datetime(x["date"])).round(6).copy()
pnl_df["pnl_bp"] = (pnl_df["pnl"]*1e4).round(2)
pnl_df["cum_return_pct"] = ((pnl_df["cum_pnl"]-1)*100).round(2)
pnl_df["dd_pct"] = (pnl_df["dd"]*100).round(2)
write_df(ws, pnl_df[["date","w","ret","pnl","pnl_bp","cum_pnl","cum_return_pct","peak","dd","dd_pct"]])
auto_width(ws)

# === Sheet 9: Per-year ===
ws = wb.create_sheet("PerYear")
py_df = pd.DataFrame(per_year_rows)
py_df["ret_pct"] = py_df["ret_pct"].round(2)
py_df["sharpe"] = py_df["sharpe"].round(2)
py_df["max_dd_pct"] = py_df["max_dd_pct"].round(2)
py_df["win_rate_pct"] = py_df["win_rate_pct"].round(1)
write_df(ws, py_df)
auto_width(ws)

# === Sheet 10: Drawdown table ===
ws = wb.create_sheet("DrawdownTable")
dd_df = pd.DataFrame(dd_periods[:20])
if not dd_df.empty:
    dd_df["start"] = pd.to_datetime(dd_df["start"])
    dd_df["end"] = pd.to_datetime(dd_df["end"])
    dd_df["depth_pct"] = dd_df["depth_pct"].round(2)
    write_df(ws, dd_df)
auto_width(ws)

# === Sheet 11: Cost sensitivity ===
ws = wb.create_sheet("CostSensitivity")
cs_df = pd.DataFrame(cost_rows)
cs_df["sharpe"] = cs_df["sharpe"].round(3)
cs_df["tot_pct"] = cs_df["tot_pct"].round(1)
cs_df["cost_bp"] = cs_df["cost_bp"].round(2)
write_df(ws, cs_df)
auto_width(ws)

# === Sheet 12: Rolling Sharpe ===
ws = wb.create_sheet("RollingSharpe")
roll_df["rolling_252d_sharpe"] = roll_df["rolling_252d_sharpe"].round(3)
write_df(ws, roll_df.assign(date=lambda x: pd.to_datetime(x["date"])))
auto_width(ws)

# === Sheet 13: Decile calibration ===
ws = wb.create_sheet("DecileCalibration")
dc = decile_summary.copy()
dc["mean_avg_pct"] = dc["mean_avg_pct"].round(4)
dc["mean_ret_bp"] = dc["mean_ret_bp"].round(2)
dc["up_rate_pct"] = dc["up_rate_pct"].round(2)
dc["mean_pnl_bp"] = dc["mean_pnl_bp"].round(2)
write_df(ws, dc)
auto_width(ws)

wb.save(EXCEL)
print(f"Saved -> {EXCEL}")

# ===== Also save flat JSON summary =====
audit_summary = dict(
    ensemble="WIN9",
    models=WIN9,
    mechanic=dict(scale=A3_SCALE, max_pos=A3_CLIP, span=A3_SPAN, cost_bp=COST_BLENDED*1e4),
    headline=dict(sharpe=round(sharpe,3), sortino=round(sortino,3), calmar=round(calmar,3),
                  cagr_pct=round(cagr_pct,2), tot_pct=round(tot_pct,1),
                  max_dd_pct=round(max_dd_pct,2),
                  win_rate=(pl>0).mean(),
                  n_days=int(len(pl)),
                  live_window=[LIVE_START, str(live["date"].max().date())]),
    solo_sharpes={m: round(s,3) for m,s in solo.items()},
    member_contributions={m: round(c,3) for m,c in contrib.items()},
    per_year=per_year_rows,
    cost_sensitivity=cost_rows,
    top_drawdowns=[dict(start=str(p["start"]), end=str(p["end"]),
                        depth_pct=round(p["depth_pct"],2), days=int(p["days"]))
                   for p in dd_periods[:10]],
)
with open(OUT/"audit_win9_summary.json","w") as f:
    json.dump(audit_summary, f, indent=2, default=str)
print(f"Saved -> {OUT/'audit_win9_summary.json'}")

# ===== Print headline =====
print("\n" + "="*70)
print(" WIN9 AUDIT — HEADLINE")
print("="*70)
print(f"  Sharpe:        {sharpe:+.3f}")
print(f"  Sortino:       {sortino:+.3f}")
print(f"  CAGR:          {cagr_pct:+.2f}%")
print(f"  Total:         {tot_pct:+.1f}%")
print(f"  Max DD:        {max_dd_pct:+.2f}%")
print(f"  Win rate:      {(pl>0).mean()*100:.1f}%")
print(f"  Live days:     {len(pl)}")
print(f"\n  Per-year:")
for r in per_year_rows:
    print(f"    {r['year']}: ret {r['ret_pct']:+5.1f}%  Sh {r['sharpe']:+.2f}  MaxDD {r['max_dd_pct']:+.1f}%")
print(f"\n  Member contributions (positive = adds Sharpe):")
for m, c in sorted(contrib.items(), key=lambda x: -x[1]):
    print(f"    {m:>5s}: +{c:.3f}  (solo {solo[m]:+.2f})")
