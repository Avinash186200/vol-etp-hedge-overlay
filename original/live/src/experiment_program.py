"""
experiment_program.py  —  VIXY close[T]->close[T+1] forecast, predicted at 3pm day T
====================================================================================
Honest, anti-overfit experiment program. Uses EVERYTHING available by 3pm:
  - feats.csv F1..F139   (client IP, generated pre-open day T -> point-in-time OK)
  - AR VIXY features     (lagged, 3pm-safe)
  - prior-day VIX/RV/IV  (shift 1)
  - same-day intraday <=15:00  (SPY/VIXY 3pm moves, overnight gap, AM/PM drift)
  - vol-ETF / cross-asset complex  (UVXY,SVXY,VIXM,SLV,USO,TIP,SHY,AGG; prior-day)
  - sector-ETF dispersion (prior-day)

TARGET: ret = vixy_close[T+1]/vixy_close[T] - 1   (today 4pm -> tomorrow 4pm)
At 3pm day T, close[T] is unknown -> NO same-day close used as a feature.

METHODOLOGY (this is the whole point — must generalize live):
  Strict chronological split, NO peeking:
    TRAIN  : start .. 2021-12-31     (fit)
    VAL    : 2022-01-01 .. 2023-12-31 (ALL model/feature/hyperparam selection)
    TEST   : 2024-01-01 .. end        (touched exactly once, reported as-is)
  + an expanding walk-forward on the chosen finalists as a robustness cross-check.
  Overfit is made visible: train vs val vs test IC & Sharpe printed for every run.
  The trading rule is a FIXED symmetric quantile L/S (long top 20% / short bottom
  20% of prediction) — NO VIX/dispersion grid-mining (that is what inflated the
  old decks). A 2bp/side cost stress is also reported.

Run:  python experiment_program.py     (writes experiments/RESULTS.txt)
"""
import warnings, time, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, ExtraTreesClassifier)
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")

REPO = Path("<local-path>"); OUT = REPO/"vixy_experiments"/"output"
DST  = REPO/"staged_finnhub"/"experiments"; DST.mkdir(parents=True, exist_ok=True)
RES  = (DST/"RESULTS.txt").open("w", encoding="utf-8")
t0=time.time()
def say(m=""):
    print(m); RES.write(str(m)+"\n"); RES.flush()
say(f"=== experiment_program  {time.ctime()} ===")

# ───────────────────────── 1. LOAD (all offline/cached) ─────────────────────────
raw=pd.read_csv(REPO/"vixy_dataset.csv").iloc[2:].copy()
raw["date"]=pd.to_datetime(raw["Price"])
raw["spy"]=pd.to_numeric(raw["Close"],errors="coerce")
raw["vixy"]=pd.to_numeric(raw["Close.1"],errors="coerce")
raw["vix"]=pd.to_numeric(raw["Close.2"],errors="coerce")
px=raw[["date","spy","vixy","vix"]].dropna().sort_values("date").reset_index(drop=True)

feats=pd.read_csv(REPO/"feats.csv",parse_dates=["date"]).sort_values("date")
Fc=[c for c in feats.columns if c.startswith("F") and c[1:].isdigit()]

spy_in=pd.read_parquet(OUT/"spy_intraday.parquet"); spy_in.index=pd.to_datetime(spy_in.index)
v3=pd.read_parquet(OUT/"vixy_3pm.parquet"); v3["date"]=pd.to_datetime(v3["date"]); v3=v3.set_index("date")
poly=pd.read_parquet(OUT/"polygon_new_data.parquet"); poly.index=pd.to_datetime(poly.index).normalize()
etf=pd.read_parquet(OUT/"etf_daily.parquet"); etf.index=pd.to_datetime(etf.index)

df=px.merge(feats,on="date",how="inner").sort_values("date").reset_index(drop=True)
di=df.set_index("date")
spy_in=spy_in.reindex(di.index); v3=v3.reindex(di.index); poly=poly.reindex(di.index); etf=etf.reindex(di.index)

# ───────────────────────── 2. LEAK-FREE FEATURES (3pm-safe) ─────────────────────
d=di.copy()
spyc=d["spy"]; vixyc=d["vixy"]; vixc=d["vix"]
spy_p=spyc.shift(1); vixy_p=vixyc.shift(1); vix_p=vixc.shift(1)
spy_r=spyc.pct_change().shift(1); vixy_r=vixyc.pct_change().shift(1)

# AR VIXY (deployed spec, 3pm-safe via extra lag)
for L in [1,2,4,6,10,15]:
    d[f"vret_l{L}"]=vixyc.pct_change().shift(L+1)
    d[f"vclose_l{L}"]=vixyc.shift(L+1)
for w in [2,3,5,8,10]:
    d[f"vret_mean{w}"]=vixyc.pct_change().rolling(w).mean().shift(1)
    d[f"vret_std{w}"]=vixyc.pct_change().rolling(w).std().shift(1)
    d[f"vpx_vs_ma{w}"]=(vixyc/vixyc.rolling(w).mean()-1).shift(1)
# prior-day VIX / RV / IV
for w in [5,10,21,63]: d[f"rv{w}"]=spy_r.rolling(w).std()*np.sqrt(252)*100
d["iv"]=vix_p
for w in [5,10,21,63]:
    d[f"ivrv{w}"]=d["iv"]/d[f"rv{w}"].replace(0,np.nan)
    d[f"ivrvspr{w}"]=d["iv"]-d[f"rv{w}"]
d["iv_lt_rv"]=(d["iv"]<d["rv21"]).astype(int)
d["vix_chg1"]=vix_p.diff(1); d["vix_chg5"]=vix_p.diff(5)
d["vix_ma5"]=vix_p.rolling(5).mean(); d["vix_ma21"]=vix_p.rolling(21).mean()
d["vix_vs5"]=vix_p/d["vix_ma5"]; d["vix_vs21"]=vix_p/d["vix_ma21"]
d["vix_pctl"]=vix_p.rolling(252).rank(pct=True)
d["vix_z21"]=(vix_p-d["vix_ma21"])/vix_p.rolling(21).std().replace(0,np.nan)
for L in [1,2,3]:
    d[f"vix_l{L}"]=vix_p.shift(L); d[f"spyr_l{L}"]=spy_r.shift(L)
d["spy_r5"]=spy_p.pct_change(5); d["spy_r21"]=spy_p.pct_change(21)
d["vix_roc5"]=vix_p.pct_change(5); d["vix_roc21"]=vix_p.pct_change(21)
# same-day intraday <=3pm
d["spy_3pm"]=spy_in["spy_close_1500"]/spy_p-1
d["vixy_3pm"]=v3["close_1500"]/vixy_p-1
d["spy_3pm_vo"]=spy_in["spy_close_1500"]/spy_in["spy_open_930"]-1
d["spy_ovn"]=spy_in["spy_open_930"]/spy_p-1
d["spy_amm"]=spy_in["spy_close_1030"]/spy_in["spy_open_930"]-1
d["spy_pmd"]=spy_in["spy_close_1500"]/spy_in["spy_close_1030"]-1
# vol-ETF / cross-asset complex (prior-day)
for c in poly.columns:
    d[f"{c}_p"]=poly[c].shift(1); d[f"{c}_r5"]=poly[c].pct_change(5).shift(1)
d["uvxy_vixy"]=poly["UVXY"].shift(1)/vixy_p; d["vixy_vixm"]=vixy_p/poly["VIXM"].shift(1)
d["tip_shy"]=(poly["TIP"]/poly["SHY"]).shift(1)
# sector dispersion (prior-day)
SW={'XLK':.30,'XLF':.13,'XLV':.12,'XLY':.10,'XLI':.09,'XLE':.04,'XLP':.06,'XLU':.03,'XLB':.02,'XLRE':.03}
sr=pd.DataFrame(index=d.index); wts=[]
for s,w in SW.items():
    col=f"{s}_close"
    if col in etf.columns: sr[s]=etf[col].pct_change().shift(1); wts.append(w)
wts=np.array(wts)/sum(wts)
for w in [5,21]: d[f"disp{w}"]=(sr.rolling(w).std()*np.sqrt(252)*100).values@wts
d["corr_proxy"]=d["rv21"]/d["disp21"].replace(0,np.nan)

# TARGET: today close -> tomorrow close
d["y_ret"]=vixyc.shift(-1)/vixyc-1
d["y_dir"]=(d["y_ret"]>0).astype(int)
d=d.reset_index(); d["year"]=d["date"].dt.year
# FIX 2026-05-27: do NOT drop rows with NaN y_ret here.
# The LATEST date (used by live_daily.py for prediction) has no next-day return
# yet, so y_ret is NaN — but we still need that row for the live forecast.
# Downstream training code (live_daily.py, init_history) already does its own
# dropna(subset=[RETCOL]) before fitting, so this is safe.
# d=d.dropna(subset=["y_ret"]).reset_index(drop=True)
d=d.reset_index(drop=True)

AR=[c for c in d.columns if c.startswith(("vret_","vclose_","vpx_vs_ma"))]
PRICE=[c for c in d.columns if c.startswith(("rv","ivrv","vix_","spy_","spyr_","disp","corr_",
        "iv","uvxy_","vixy_","tip_")) or c.endswith(("_p","_r5"))]
PRICE=[c for c in PRICE if c not in ("y_ret","y_dir")]
FSET={
 "F_only":   Fc+AR,
 "Price_only": sorted(set(PRICE)),
 "ALL":      Fc+AR+sorted(set(PRICE)),
}
say(f"rows={len(d)}  {d.date.min().date()}->{d.date.max().date()}  "
    f"|F={len(Fc)} AR={len(AR)} PRICE={len(set(PRICE))} ALL={len(FSET['ALL'])}")

# ───────────────────────── 3. SPLIT ─────────────────────────
tr=d[d.date<="2021-12-31"]; va=d[(d.date>="2022-01-01")&(d.date<="2023-12-31")]; te=d[d.date>="2024-01-01"]
say(f"TRAIN n={len(tr)} ({tr.date.min().date()}..{tr.date.max().date()}) "
    f"VAL n={len(va)} TEST n={len(te)} ({te.date.min().date()}..{te.date.max().date()})")

def metrics(pred, sub, cost=0.0):
    r=sub["y_ret"].values; yr=sub["year"].values
    ic=stats.spearmanr(pred,r)[0]
    ba=( (pred>=np.median(pred)).astype(int)==(r>0).astype(int) ).mean()
    hi=np.quantile(pred,0.80); lo=np.quantile(pred,0.20)
    pos=np.where(pred>=hi,1.0,np.where(pred<=lo,-1.0,0.0))
    pnl=pos*r - np.abs(np.diff(np.r_[0,pos]))*cost
    a=pnl[pos!=0]
    sh=a.mean()/a.std()*np.sqrt(252) if len(a)>10 and a.std()>0 else 0
    cum=(1+pnl).cumprod(); tot=(cum[-1]-1)*100; dd=((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min()*100
    yrr={int(y):(1+pnl[yr==y]).prod()-1 for y in sorted(set(yr))}
    return dict(ic=ic,ba=ba,sh=sh,tot=tot,dd=dd,py=sum(v>0 for v in yrr.values()),ny=len(yrr),yr=yrr)

def fitpred(model, feats, train, eval_):
    m=model(); m.fit(train[feats].values, train[TGT].values)
    if hasattr(m,"predict_proba"): return m.predict_proba(eval_[feats].values)[:,1]
    return m.predict(eval_[feats].values)

# ───────────────────────── 4. EXPERIMENT GRID (select on VAL) ───────────────────
def P(est): return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("m",est)])
CLF={
 "LR_L2_C0.1": lambda:P(LogisticRegression(C=0.1,max_iter=2000,class_weight="balanced")),
 "LR_L2_C1":   lambda:P(LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced")),
 "LR_L1_C0.5": lambda:P(LogisticRegression(C=0.5,penalty="l1",solver="liblinear",class_weight="balanced")),
 "HGB":        lambda:HistGradientBoostingClassifier(max_iter=300,max_depth=3,learning_rate=0.03,
                       min_samples_leaf=40,l2_regularization=1.0,random_state=42),
 "RF":         lambda:RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=30,
                       n_jobs=-1,random_state=42,class_weight="balanced"),
 "ET":         lambda:ExtraTreesClassifier(n_estimators=400,max_depth=6,min_samples_leaf=25,
                       n_jobs=-1,random_state=42,class_weight="balanced"),
}
REG={
 "Ridge":  lambda:P(Ridge(alpha=10.0)),
 "ENet":   lambda:P(ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=5000)),
 "HGBR":   lambda:HistGradientBoostingRegressor(max_iter=300,max_depth=3,learning_rate=0.03,
                   min_samples_leaf=40,l2_regularization=1.0,random_state=42),
}
say("\n"+"="*96)
say(f"{'config':<34s}{'feat':>4s} | {'TRAIN ic/sh':>14s} | {'VAL ic/ba/sh':>18s} | {'overfit(tr-va ic)':>16s}")
say("="*96)
board=[]
for fs_name,feats in FSET.items():
    feats=[c for c in feats if c in d.columns]
    # feature-select top-40 by MI on TRAIN only (for ALL set, anti-overfit)
    variants={fs_name:feats}
    if fs_name=="ALL":
        Xt=tr[feats].fillna(tr[feats].median())
        mi=mutual_info_classif(Xt.values, tr["y_dir"].values, random_state=42)
        top=[feats[i] for i in np.argsort(mi)[-40:]]
        variants["ALL_sel40"]=top
    for vname,vf in variants.items():
        for tag,zoo,TGT_ in [("clf",CLF,"y_dir"),("reg",REG,"y_ret")]:
            for mname,mk in zoo.items():
                TGT=TGT_
                try:
                    ptr=fitpred(mk,vf,tr,tr); pva=fitpred(mk,vf,tr,va)
                except Exception as e:
                    say(f"{vname}|{mname:<10s} ERR {e!r:.50}"); continue
                mt=metrics(ptr,tr); mv=metrics(pva,va)
                cfg=f"{vname}:{mname}({tag})"
                board.append((cfg,vname,vf,mname,mk,tag,TGT_,mt,mv))
                say(f"{cfg:<34s}{len(vf):>4d} | {mt['ic']:>+6.3f}/{mt['sh']:>5.2f} | "
                    f"{mv['ic']:>+6.3f}/{mv['ba']:.2f}/{mv['sh']:>5.2f} | {mt['ic']-mv['ic']:>+8.3f}")

# pick finalists by VAL: rank by (val IC, val Sharpe), require not wildly overfit
def score(x): _,_,_,_,_,_,_,mt,mv=x; return (mv["ic"]>0)*1.0 + mv["ic"]*5 + mv["sh"]*0.3 - max(0,(mt["ic"]-mv["ic"])-0.05)*4
board.sort(key=score, reverse=True)
say("\n"+"="*96); say("TOP-5 BY VALIDATION (selection done — TEST still untouched):"); say("="*96)
for x in board[:5]:
    cfg,_,_,_,_,_,_,mt,mv=x
    say(f"  {cfg:<34s} VAL ic={mv['ic']:+.3f} ba={mv['ba']:.3f} sh={mv['sh']:.2f}  | TRAIN ic={mt['ic']:+.3f}")

# ───────────────────────── 5. EVALUATE TOP-3 ON TEST (once) ─────────────────────
say("\n"+"="*96); say("FINAL — TEST 2024..2026 (touched once). +2bp/side cost stress in [..]"); say("="*96)
trva=d[d.date<="2023-12-31"]
finalists=board[:3]
for cfg,vname,vf,mname,mk,tag,TGT_,mt,mv in finalists:
    TGT=TGT_
    pte=fitpred(mk,vf,trva,te)              # refit on TRAIN+VAL, predict TEST
    m0=metrics(pte,te,cost=0.0); m2=metrics(pte,te,cost=0.0002)
    say(f"\n{cfg}")
    say(f"  TEST  IC={m0['ic']:+.4f}  BalAcc={m0['ba']:.3f}  "
        f"L/S Sharpe={m0['sh']:.2f} [{m2['sh']:.2f}]  Tot={m0['tot']:+.0f}% [{m2['tot']:+.0f}%]  "
        f"MaxDD={m0['dd']:.0f}%  PosYrs={m0['py']}/{m0['ny']}")
    say(f"  vs VAL IC={mv['ic']:+.3f} / TRAIN IC={mt['ic']:+.3f}  -> generalization gap "
        f"{mt['ic']-m0['ic']:+.3f}")
    say(f"  TEST yearly: "+"  ".join(f"{y}:{v*100:+.0f}%" for y,v in m0['yr'].items()))
# baseline: always-short VIXY over TEST
rb=te["y_ret"].values
shb=(-rb).mean()/(-rb).std()*np.sqrt(252) if rb.std()>0 else 0
totb=((1+(-rb)).prod()-1)*100
say(f"\nBASELINE always-short VIXY on TEST: Sharpe={shb:.2f}  Tot={totb:+.0f}%")
say(f"\nelapsed {(time.time()-t0)/60:.1f} min"); RES.close()
