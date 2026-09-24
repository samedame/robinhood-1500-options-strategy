"""
In-sample tournament (2010-01-01 .. 2021-12-31). Only IS chain files are loaded, so holdout data cannot leak.
Writes per-variant trade lists to results/is/, and appends every variant to logs/research_log.csv.
"""
import sys, os, json, itertools, time
sys.path.insert(0, "/home/claude/research/src")
import numpy as np, pandas as pd
from core import *
from stats import *

R = "/home/claude/research"
OUT = f"{R}/results/is"; os.makedirs(OUT, exist_ok=True)
LOG = f"{R}/logs/research_log.csv"
IS_START, IS_END = "2010-01-01", "2021-12-31"
IS_YEARS = 12.0

def log_variant(fam, vid, params, df, years, extra=None):
    s = summarize(df, years) if len(df) else dict(n=0)
    row = dict(time=pd.Timestamp.now('UTC').isoformat(), family=fam, variant=vid, params=json.dumps(params), period="IS",
               **{k: s.get(k) for k in ["n", "per_yr", "mean", "median", "win", "ci_lo", "ci_hi", "nw_t", "kelly", "score", "worst"]})
    if extra: row.update(extra)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)

def save(vid, trades):
    df = trades_df(trades)
    df.to_parquet(f"{OUT}/{vid}.parquet", index=False)
    return df

def main(which):
    t0 = time.time()
    ch = Chains("SPY", {y: "odx" for y in range(2010, 2022)})
    px = load_ohlc("SPY")
    px = px[px.index <= pd.Timestamp(IS_END) + pd.Timedelta(days=40)]   # a few weeks past IS end for exits only
    eng = Engine(ch, px)
    print("loaded", time.time() - t0, flush=True)
    close = px.Close
    sig_ibs = px.index[(ibs(px) < 0.10).values]
    fams = {}
    if "F1" in which:
        dtes = [2, 5, 10, 20, 35]; deltas = [0.30, 0.50, 0.70]; exits = ["IBS10", "IBS5", "H1", "H3"]
        for dte, de, ex in itertools.product(dtes, deltas, exits):
            vid = f"F1_d{dte}_D{int(de*100)}_{ex}"
            legs = [dict(cp="C", side=1, sel=("delta", de))]
            tr, sk = run_signals(eng, sig_ibs, legs, dte, ex, IS_START, IS_END)
            df = save(vid, tr)
            log_variant("F1", vid, dict(dte=dte, delta=de, exit=ex), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    if "F4" in which:
        tset = set(px.index)
        vx = vix_expirations("2010-01-01", "2022-01-31", tset)
        pre = {}
        for v in vx:
            i = eng.tpos.get(v)
            if i is None or i == 0: continue
            pre[eng.tdays[i - 1]] = v
        sig = pd.DatetimeIndex(sorted(pre))
        for dte, de in itertools.product([2, 7], [0.50, 0.30]):
            vid = f"F4_d{dte}_D{int(de*100)}"
            legs = [dict(cp="P", side=1, sel=("delta", de))]
            tr, sk = run_signals(eng, sig, legs, dte, "EVT", IS_START, IS_END, event_map=pre)
            df = save(vid, tr)
            log_variant("F4", vid, dict(dte=dte, delta=de), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    if "F5" in which:
        v9 = load_cboe("VIX9D")
        lr = np.log(close).diff()
        rv10 = lr.rolling(10).std() * np.sqrt(252) * 100
        ratio = (v9.reindex(px.index).ffill() / rv10)
        for thr, h in itertools.product([0.9, 1.0], [1, 3]):
            vid = f"F5_thr{thr}_H{h}"
            sig = px.index[(ratio < thr).values]
            legs = [dict(cp="C", side=1, sel=("atm",)), dict(cp="P", side=1, sel=("atm",))]
            tr, sk = run_signals(eng, sig, legs, 7, f"H{h}", IS_START, IS_END)
            df = save(vid, tr)
            log_variant("F5", vid, dict(thr=thr, hold=h), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    if "F6" in which:
        for de, w, dte in itertools.product([0.30, 0.50], [0.01, 0.02], [5, 10]):
            vid = f"F6_D{int(de*100)}_w{int(w*100)}_d{dte}"
            legs = [dict(cp="P", side=-1, sel=("delta", de)), dict(cp="P", side=1, sel=("offset_pct", -w), ref=0)]
            tr, sk = run_signals(eng, sig_ibs, legs, dte, "IBS10", IS_START, IS_END)
            df = save(vid, tr)
            log_variant("F6", vid, dict(delta=de, width=w, dte=dte), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    if "F7" in which:
        for w, dte in itertools.product([0.01, 0.02], [5, 10]):
            vid = f"F7_w{int(w*100)}_d{dte}"
            legs = [dict(cp="C", side=1, sel=("delta", 0.50)), dict(cp="C", side=-1, sel=("offset_pct", w), ref=0)]
            tr, sk = run_signals(eng, sig_ibs, legs, dte, "IBS10", IS_START, IS_END)
            df = save(vid, tr)
            log_variant("F7", vid, dict(width=w, dte=dte), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    if "F8" in which:
        v = load_cboe("VIX"); v3 = load_cboe("VIX3M")
        ts = (v3 / v).reindex(px.index).ffill()
        mondays = pd.Series(px.index, index=px.index).groupby(px.index.to_period("W")).first().values
        mondays = pd.DatetimeIndex(mondays)
        for gate in ["none", "contango"]:
            vid = f"F8_{gate}"
            sig = mondays if gate == "none" else pd.DatetimeIndex([d for d in mondays if ts.get(d, np.nan) > 1.0])
            legs = [dict(cp="P", side=-1, sel=("delta", 0.30)), dict(cp="P", side=1, sel=("offset_pct", -0.02), ref=0)]
            tr, sk = run_signals(eng, sig, legs, 7, "H10", IS_START, IS_END, pt_credit=0.5)
            df = save(vid, tr)
            log_variant("F8", vid, dict(gate=gate), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)

    if "F2" in which:
        r2 = rsi2(close)
        trig = {"rsi2lt10": px.index[(r2 < 10).values],
                "down3": px.index[down_streak(close, 3).values],
                "low5": px.index[at_n_day_low(close, 5).values]}
        for name, sig in trig.items():
            vid = f"F2_{name}"
            legs = [dict(cp="C", side=1, sel=("delta", 0.70))]
            tr, sk = run_signals(eng, sig, legs, 20, "IBS10", IS_START, IS_END)
            df = save(vid, tr)
            log_variant("F2", vid, dict(trigger=name, dte=20, delta=0.70, exit="IBS10"), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    if "F3" in which:
        sma200 = close.rolling(200).mean()
        mondays = pd.DatetimeIndex(pd.Series(px.index, index=px.index).groupby(px.index.to_period("W")).first().values)
        for gate in ["none", "sma200"]:
            vid = f"F3_{gate}"
            sig = mondays if gate == "none" else pd.DatetimeIndex([d for d in mondays if close.get(d) > sma200.get(d, np.inf)])
            legs = [dict(cp="C", side=1, sel=("delta", 0.50))]
            tr, sk = run_signals(eng, sig, legs, 20, "H5", IS_START, IS_END)
            df = save(vid, tr)
            log_variant("F3", vid, dict(gate=gate, dte=20, delta=0.5, hold=5), df, IS_YEARS, dict(skipped=sk))
            print(vid, len(df), round(df.ret.mean(), 4) if len(df) else None, flush=True)
    print("done", time.time() - t0)

if __name__ == "__main__":
    main(sys.argv[1].split(","))
