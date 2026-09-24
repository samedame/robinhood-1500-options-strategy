"""Addendum A exploratory IS: IBS calls on QQQ, IWM, DIA (proxy-priced where no chain history) with F1 structures."""
import sys, time, os; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import *
from proxy import ProxyChains
R='/home/claude/research'; OUT=f'{R}/results/is_breadth'; os.makedirs(OUT,exist_ok=True)
t0=time.time()
spy_ch=Chains('SPY',{y:'odx' for y in range(2010,2022)})
exp_src={d: sorted(g.index.get_level_values(0).unique()) for d,g in spy_ch.by_date.items()}
import sys as _s
structs={'d20_D70':(20,0.70),'d10_D50':(10,0.50),'d20_D50':(20,0.50),'d10_D70':(10,0.70)} if len(_s.argv)<2 else {'d10_D40':(10,0.40),'d10_D30':(10,0.30)}
for sym in ['SPY','QQQ','IWM','DIA']:
    px=load_ohlc(sym); px=px[px.index<='2022-02-15']
    sig=px.index[(ibs(px)<0.10).values]
    if sym=='SPY':
        eng=Engine(spy_ch,px)
    else:
        eng=Engine(ProxyChains(sym,px,exp_src),px)
    for name,(dte,de) in structs.items():
        legs=[dict(cp='C',side=1,sel=('delta',de))]
        tr,sk=run_signals(eng,sig,legs,dte,'IBS10','2010-01-01','2021-12-31')
        df=trades_df(tr); df['sym']=sym; df['priced']='real' if sym=='SPY' else 'proxy'
        df.to_parquet(f'{OUT}/{sym}_{name}.parquet',index=False)
        print(sym,name,len(df),round(df.ret.mean(),4),'skipped',sk,round(time.time()-t0,1),flush=True)
