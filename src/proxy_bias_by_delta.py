import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import *
from proxy import ProxyChains
px=load_ohlc('SPY')
res=[]
for label,vend,s,e in [('IS2016_21',{y:'odx' for y in range(2016,2022)},'2016-01-01','2021-12-31'),('HO',{2022:'odx',2023:'odx',2024:'av',2025:'av'},'2022-01-01','2025-12-31')]:
    ch=Chains('SPY',vend); exp_src={d: sorted(g.index.get_level_values(0).unique()) for d,g in ch.by_date.items()}
    er=Engine(ch,px); ep=Engine(ProxyChains('SPY',px,exp_src),px)
    sig=px.index[(ibs(px)<0.10).values]
    for de in [0.30,0.40,0.50]:
        C=[dict(cp='C',side=1,sel=('delta',de))]
        tr,_=run_signals(er,sig,C,10,'IBS10',s,e)
        d=[]
        for t in tr:
            tp=ep.simulate(t.entry,C,10,'IBS10')
            if tp is not None: d.append((t.ret[0.25],tp.ret[0.25]))
        d=np.array(d)
        res.append(dict(period=label,delta=de,n=len(d),real=d[:,0].mean(),proxy=d[:,1].mean(),bias=(d[:,1]-d[:,0]).mean()))
        print(res[-1],flush=True)
