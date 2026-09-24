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

C1 = [dict(cp="C", side=1, sel=("delta", 0.70))]
# H1
save("H1_F1_d20_D70_IBS10", run_signals(spy_eng, sig_ibs, C1, 20, "IBS10", HS, HE)[0])
# pre-registered SMA200 filter on F1
sma = close.rolling(200).mean()
sig_f = pd.DatetimeIndex([d for d in sig_ibs if close.get(d) > sma.get(d, np.inf)])
save("F1_sma200", run_signals(spy_eng, sig_f, C1, 20, "IBS10", HS, HE)[0])
# H2
save("H2_F2_down3", run_signals(spy_eng, spy_px.index[down_streak(close, 3).values], C1, 20, "IBS10", HS, HE)[0])
# H3
mondays = pd.DatetimeIndex(pd.Series(spy_px.index, index=spy_px.index).groupby(spy_px.index.to_period("W")).first().values)
save("H3_F3_none", run_signals(spy_eng, mondays, [dict(cp="C", side=1, sel=("delta", 0.50))], 20, "H5", HS, HE)[0])
# H4
tset = set(spy_px.index)
vx = vix_expirations("2021-12-01", "2026-01-31", tset)
pre = {}
for v in vx:
    i = spy_eng.tpos.get(v)
    if i:
        pre[spy_eng.tdays[i - 1]] = v
save("H4_F4_d7_D50", run_signals(spy_eng, pd.DatetimeIndex(sorted(pre)), [dict(cp="P", side=1, sel=("delta", 0.50))], 7, "EVT", HS, HE, event_map=pre)[0])
# H5
v9 = load_cboe("VIX9D"); lr = np.log(close).diff(); rv10 = lr.rolling(10).std() * np.sqrt(252) * 100
ratio = v9.reindex(spy_px.index).ffill() / rv10
save("H5_F5_thr0.9_H3", run_signals(spy_eng, spy_px.index[(ratio < 0.9).values], [dict(cp="C", side=1, sel=("atm",)), dict(cp="P", side=1, sel=("atm",))], 7, "H3", HS, HE)[0])
# H6
save("H6_F6_D30_w2_d10", run_signals(spy_eng, sig_ibs, [dict(cp="P", side=-1, sel=("delta", 0.30)), dict(cp="P", side=1, sel=("offset_pct", -0.02), ref=0)], 10, "IBS10", HS, HE)[0])
# H7
save("H7_F7_w2_d10", run_signals(spy_eng, sig_ibs, [dict(cp="C", side=1, sel=("delta", 0.50)), dict(cp="C", side=-1, sel=("offset_pct", 0.02), ref=0)], 10, "IBS10", HS, HE)[0])
# H8
save("H8_F8_none", run_signals(spy_eng, mondays, [dict(cp="P", side=-1, sel=("delta", 0.30)), dict(cp="P", side=1, sel=("offset_pct", -0.02), ref=0)], 7, "H10", HS, HE, pt_credit=0.5)[0])

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

basket()
basket(delay=1, tag="_delay1")
basket(dte=8, tag="_dte8"); basket(dte=12, tag="_dte12")
basket(delta=0.40, tag="_D40"); basket(delta=0.60, tag="_D60")
basket(rule="IBS8", tag="_hold8"); basket(rule="IBS12", tag="_hold12")
def smaf(sym, px, sig):
    s = px.Close.rolling(200).mean()
    return pd.DatetimeIndex([d for d in sig if px.Close.get(d) > s.get(d, np.inf)])
basket(sigfilter=smaf, tag="_sma200")
# QQQ 2022: proxy vs real on identical signals (proxy validation in the holdout year)
tr_p, _ = run_signals(qqq_prox, qqq_px.index[(ibs(qqq_px) < 0.10).values], C9, 10, "IBS10", HS, "2022-12-31")
save("QQQ2022_proxy_check", tr_p)
# Vendor check: SPY 2022-2023 on Alpha Vantage chains
av_ch = Chains("SPY", {2022: "av", 2023: "av"})
av_eng = Engine(av_ch, spy_px)
save("H9_SPY_AV_2022_2023", run_signals(av_eng, sig_ibs, C9, 10, "IBS10", HS, "2023-12-31")[0])
save("H1_SPY_AV_2022_2023", run_signals(av_eng, sig_ibs, C1, 20, "IBS10", HS, "2023-12-31")[0])
print("done", round(time.time() - t0, 1))
