"""
SEALED HOLDOUT RUN (2022-01-01 .. 2025-12-31). Run once for the pre-registered hypotheses H1-H9, the pre-registered
SMA200 filter test, benchmarks, and the champion's stress variants. Results -> results/holdout/.
"""
import sys, os, time, json
sys.path.insert(0, "/home/claude/research/src")
import numpy as np, pandas as pd
from core import *
from proxy import ProxyChains

R = "/home/claude/research"; OUT = f"{R}/results/holdout"; os.makedirs(OUT, exist_ok=True)
HS, HE = "2022-01-01", "2025-12-31"
t0 = time.time()

def save(name, trades, extra=None):
    df = trades_df(trades)
    if extra:
        for k, v in extra.items():
            df[k] = v
    df.to_parquet(f"{OUT}/{name}.parquet", index=False)
    print(name, len(df), round(df.ret.mean(), 4) if len(df) else None, round(time.time() - t0, 1), flush=True)
    return df

spy_px = load_ohlc("SPY")
spy_ch = Chains("SPY", {2021: "odx", 2022: "odx", 2023: "odx", 2024: "av", 2025: "av"})
spy_eng = Engine(spy_ch, spy_px)
print("SPY chains loaded", round(time.time() - t0, 1), flush=True)
sig_ibs = spy_px.index[(ibs(spy_px) < 0.10).values]
close = spy_px.Close

C1=[dict(cp='C',side=1,sel=('delta',0.70))]
# H9 champion: F9 basket, d10 / 0.50 delta, IBS10 -- SPY real; QQQ real 2022 + proxy 2023-25; IWM proxy
C9 = [dict(cp="C", side=1, sel=("delta", 0.50))]
exp_src = {d: sorted(g.index.get_level_values(0).unique()) for d, g in spy_ch.by_date.items()}
engines = {"SPY": ("real", spy_eng, spy_px)}
qqq_px = load_ohlc("QQQ"); iwm_px = load_ohlc("IWM")
qqq_real = Engine(Chains("QQQ", {2021: "odx", 2022: "odx"}), qqq_px)
qqq_prox = Engine(ProxyChains("QQQ", qqq_px, exp_src), qqq_px)
iwm_prox = Engine(ProxyChains("IWM", iwm_px, exp_src), iwm_px)

def basket(dte=10, delta=0.50, rule="IBS10", delay=0, max_hold=None, sigfilter=None, tag=""):
    out = []
    for sym, px, engs in [("SPY", spy_px, [(spy_eng, "real", HS, HE)]),
                          ("QQQ", qqq_px, [(qqq_real, "real", HS, "2022-12-31"), (qqq_prox, "proxy", "2023-01-01", HE)]),
                          ("IWM", iwm_px, [(iwm_prox, "proxy", HS, HE)])]:
        sig = px.index[(ibs(px) < 0.10).values]
        if sigfilter is not None:
            sig = sigfilter(sym, px, sig)
        for eng, priced, s, e in engs:
            tr, sk = run_signals(eng, sig, [dict(cp="C", side=1, sel=("delta", delta))], dte, rule, s, e, delay=delay,
                                 max_hold=max_hold)
            df = trades_df(tr); df["sym"] = sym; df["priced"] = priced
            out.append(df)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(f"{OUT}/H9_F9{tag}.parquet", index=False)
    print("H9_F9" + tag, len(df), round(df.ret.mean(), 4), df.groupby("sym").ret.mean().round(4).to_dict(), round(time.time() - t0, 1), flush=True)
    return df

basket(delta=0.30, tag='_D30')
print('done')
