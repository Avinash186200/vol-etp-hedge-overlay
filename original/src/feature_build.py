"""
feature_build.py  —  CANONICAL feature constructor (live == backtest)
=====================================================================
Single source of truth for the P7030 model's features. Mirrors EXACTLY the
validated overnight_master.py PHASE 0 so the live signal cannot drift from
the backtested model. Returns the full leak-free 3pm-safe matrix.

build_matrix() -> dict(d, Fc, AR, FSETS, books, asof, staleness)
All features are prior-day / pre-open / intraday<=15:00 (3pm-safe).
"""
import io, contextlib, warnings, datetime as _dt
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

ROOT      = Path("<local-path>")
STAGED    = ROOT/"staged_finnhub"
EXPROG    = STAGED/"experiment_program.py"
VIXY_DS   = ROOT/"vixy_dataset.csv"          # SPY/VIXY/VIX OHLC (refresh via data_feed)
DISP_PARQ = STAGED/"stock_dispersion.parquet"

def build_matrix(verbose=False):
    log=(lambda *a: print("[feature_build]",*a)) if verbose else (lambda *a: None)

    # base leak-free builder (sections 1-3 of experiment_program)
    src=EXPROG.read_text(encoding="utf-8")
    ns={}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src[:src.rindex(chr(10),0,src.index("4. EXPERIMENT GRID"))],"b","exec"),ns)
    d=ns["d"].copy().sort_values("date").reset_index(drop=True); Fc,AR=ns["Fc"],ns["AR"]
    PRICE=[c for c in ns["FSET"]["Price_only"] if c in d.columns]

    # candle / wick (prior-day OHLC)
    raw=pd.read_csv(VIXY_DS).iloc[2:].copy(); raw["date"]=pd.to_datetime(raw["Price"])
    om={"spy":("Open","High","Low","Close"),"vixy":("Open.1","High.1","Low.1","Close.1"),
        "vix":("Open.2","High.2","Low.2","Close.2")}
    oh=raw[["date"]].copy()
    for nm,(o,h,l,c) in om.items():
        for k,col in zip("ohlc",(o,h,l,c)): oh[f"{nm}_{k}"]=pd.to_numeric(raw[col],errors="coerce")
    oh=oh.sort_values("date").reset_index(drop=True); WICK=[]
    for p in ("vixy","spy","vix"):
        O,H,L,C=oh[f"{p}_o"],oh[f"{p}_h"],oh[f"{p}_l"],oh[f"{p}_c"]; rg=(H-L).replace(0,np.nan)
        for k,v in {f"{p}_uwick":(H-np.maximum(O,C))/C,f"{p}_lwick":(np.minimum(O,C)-L)/C,
                    f"{p}_body":(C-O).abs()/C,f"{p}_range":(H-L)/C,f"{p}_clopos":(C-L)/rg,
                    f"{p}_gap":O/C.shift(1)-1}.items(): oh[k]=v; WICK.append(k)
        z=f"{p}_range_z"; oh[z]=(oh[f"{p}_range"]-oh[f"{p}_range"].rolling(63).mean())/oh[f"{p}_range"].rolling(63).std(); WICK.append(z)
    d=d.merge(oh[["date"]+WICK],on="date",how="left")
    for c in WICK: d[c]=d[c].shift(1)

    # true single-stock dispersion (cached parquet; refreshed by data_feed)
    disp=pd.read_parquet(DISP_PARQ); disp["date"]=pd.to_datetime(disp["date"])
    DISP=[c for c in disp.columns if c!="date"]
    disp_asof=disp["date"].max()
    d=d.merge(disp,on="date",how="left")
    for c in DISP: d[c]=d[c].shift(1)

    # GARCH(1,1) cond-vol, params fixed pre-2019 (identical to backtest)
    from arch import arch_model
    sret=(d["spy"].pct_change()*100.0)
    fit=arch_model(sret[d.date<="2018-12-31"].dropna(),vol="Garch",p=1,q=1,dist="t").fit(disp="off")
    cv=arch_model(sret.fillna(0.0),vol="Garch",p=1,q=1,dist="t").forecast(
        params=fit.params,horizon=1,reindex=True).variance.values.ravel()
    d["garch_vol"]=pd.Series(np.sqrt(np.r_[np.nan,cv[:-1]])).shift(1).values
    d["garch_rv_gap"]=d["garch_vol"]-d.get("rv21",np.nan); d["garch_chg5"]=d["garch_vol"]-d["garch_vol"].shift(5)
    r1=d["spy"].pct_change().shift(1)
    d["rv_d"]=(r1**2)*np.sqrt(252)*100
    d["rv_w"]=(r1**2).rolling(5).mean()*np.sqrt(252)*100
    d["rv_m"]=(r1**2).rolling(22).mean()*np.sqrt(252)*100
    d["rsemi_dn"]=(r1.where(r1<0)**2).rolling(22).mean()*np.sqrt(252)*100
    d["rsemi_up"]=(r1.where(r1>0)**2).rolling(22).mean()*np.sqrt(252)*100
    d["signed_jump"]=d["rsemi_up"]-d["rsemi_dn"]
    d["rskew21"]=r1.rolling(21).skew(); d["rkurt21"]=r1.rolling(21).kurt()
    HAR=["rv_d","rv_w","rv_m","rsemi_dn","rsemi_up","signed_jump","rskew21","rkurt21",
         "garch_vol","garch_rv_gap","garch_chg5"]

    # calendar
    dt=d["date"]; d["dow"]=dt.dt.dayofweek; d["dom"]=dt.dt.day; d["month"]=dt.dt.month
    d["toM"]=((dt.dt.day<=3)|(dt.dt.day>=26)).astype(int)
    tf=dt.apply(lambda x:pd.Timestamp(x.year,x.month,1)+pd.offsets.WeekOfMonth(week=2,weekday=4))
    d["days_to_opex"]=(tf-dt).dt.days; d["opex_week"]=d["days_to_opex"].between(0,4).astype(int)
    CAL=["dow","dom","month","toM","days_to_opex","opex_week"]

    # regime-normalized (expanding -> leak-free)
    RN=[]
    for c in [x for x in ["rv21","ivrv21","garch_vol","vix_pctl","corr_proxy_21","impl_corr_21"] if x in d.columns]:
        d[f"{c}_xz"]=((d[c]-d[c].expanding(252).mean())/d[c].expanding(252).std()); RN.append(f"{c}_xz")

    # books: dependent-variable return series (next-day)
    vixy_next=d["vixy"].shift(-1)/d["vixy"]-1
    spy_next =d["spy"].shift(-1)/d["spy"]-1
    d["ret_vixy"]=vixy_next
    d["ret_p7030"]=0.70*spy_next + (-0.30)*vixy_next
    books={"VIXY":("ret_vixy",-1.0),"P7030":("ret_p7030",+1.0)}

    BASE_PRICE=sorted(set(PRICE))
    FSETS={
      "F_AR":      list(dict.fromkeys(Fc+AR)),
      "PRICE_VOL": list(dict.fromkeys(BASE_PRICE+HAR+WICK)),
      "DISP_REG":  list(dict.fromkeys(DISP+RN+HAR+["vixy_vixm","corr_proxy_21","impl_corr_21"])),
      "ALL":       list(dict.fromkeys(Fc+AR+BASE_PRICE+HAR+WICK+DISP+CAL+RN)),
    }
    FSETS={k:[c for c in v if c in d.columns] for k,v in FSETS.items()}

    asof=d["date"].max()
    today=pd.Timestamp(_dt.date.today())
    bd_behind=int(np.busday_count(asof.date(),today.date()))
    staleness=dict(price_asof=str(asof.date()),
                   dispersion_asof=str(disp_asof.date()),
                   business_days_behind=bd_behind,
                   stale=(bd_behind>1 or (today-pd.Timestamp(disp_asof)).days>4))
    log(f"matrix rows={len(d)} asof={asof.date()} dispersion_asof={disp_asof.date()} "
        f"bd_behind={bd_behind} ALL_feats={len(FSETS['ALL'])}")
    return dict(d=d,Fc=Fc,AR=AR,FSETS=FSETS,books=books,asof=asof,staleness=staleness)

if __name__=="__main__":
    m=build_matrix(verbose=True)
    print("OK — staleness:",m["staleness"])
