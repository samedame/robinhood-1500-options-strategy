import sys, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import *
from proxy import ProxyChains
t0=time.time()
spy_ch=Chains('SPY',{y:'odx' for y in range(2016,2022)})
exp_src={d: sorted(g.index.get_level_values(0).unique()) for d,g in spy_ch.by_date.items()}
def compare(sym, real_years, start, end, legs, dte, rule):
    px=load_ohlc(sym); px=px[px.index<=pd.Timestamp(end)+pd.Timedelta(days=45)]
    real=Chains(sym,{y:'odx' for y in real_years})
    er=Engine(real,px); ep=Engine(ProxyChains(sym,px,exp_src),px)
    sig=px.index[(ibs(px)<0.10).values]
    tr_r,_=run_signals(er,sig,legs,dte,rule,start,end)
    rows=[]
    for t in tr_r:
        tp=ep.simulate(t.entry,legs,dte,rule)
        if tp is None: continue
        rows.append(dict(entry=t.entry,exit=t.exit,exit_p=tp.exit,real=t.ret[0.25],proxy=tp.ret[0.25],k_r=t.legs[0][2],k_p=tp.legs[0][2],
                         e_r=t.expiry,e_p=tp.expiry,px_r=t.entry_px[0.25],px_p=tp.entry_px[0.25]))
    return pd.DataFrame(rows)
legs=[dict(cp='C',side=1,sel=('delta',0.70))]
for sym,yrs,s,e in [('QQQ',[2021],'2021-03-01','2021-12-31'),('SPY',list(range(2019,2022)),'2019-01-01','2021-12-31')]:
    df=compare(sym,yrs,s,e,legs,20,'IBS10')
    print(sym,'n',len(df),'mean real',round(df.real.mean(),4),'mean proxy',round(df.proxy.mean(),4),'corr',round(df[['real','proxy']].corr().iloc[0,1],3),
          'MAE',round((df.real-df.proxy).abs().mean(),4),'entry price ratio median',round((df.px_p/df.px_r).median(),3),
          'same strike share',round((df.k_r==df.k_p).mean(),2),'same expiry share',round((df.e_r==df.e_p).mean(),2), 'time',round(time.time()-t0,1))
    df.to_csv(f'results/proxy_validation_{sym}.csv',index=False)
