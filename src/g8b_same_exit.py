"""Supplementary G8: random ENTRY days with the SAME exit rule (IBS10 + expiry cap). Isolates the entry signal's value."""
import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd, time
from core import *
t0=time.time()
px=load_ohlc('SPY')
C9=[dict(cp='C',side=1,sel=('delta',0.50))]
for label,vend,start,end in [('IS',{y:'odx' for y in range(2010,2022)},'2010-01-01','2021-12-31'),('HO',{2022:'odx',2023:'odx',2024:'av',2025:'av'},'2022-01-01','2025-12-31')]:
    eng=Engine(Chains('SPY',vend),px)
    days=[d for d in eng.tdays if pd.Timestamp(start)<=d<=pd.Timestamp(end)]
    allr={}
    for d in days:
        t=eng.simulate(d,C9,10,'IBS10',cs=(0.25,))
        if t is not None: allr[d]=t.ret[0.25]
    s=pd.Series(allr)
    sig=set(px.index[(ibs(px)<0.10).values])
    is_sig=s[[d in sig for d in s.index]]; non=s[[d not in sig for d in s.index]]
    # strategy trades (one at a time) for trade counts per year
    tr,_=run_signals(eng,px.index[(ibs(px)<0.10).values],C9,10,'IBS10',start,end)
    act=np.mean([t.ret[0.25] for t in tr]); per_year=pd.Series([t.entry.year for t in tr]).value_counts()
    rng=np.random.default_rng(5); means=[]
    for _ in range(2000):
        rs=[]
        for y,n in per_year.items():
            pool=s[s.index.year==y].values
            rs+=list(rng.choice(pool,size=min(n,len(pool)),replace=False))
        means.append(np.mean(rs))
    means=np.array(means)
    print(label,'strategy mean',round(act,4),'| all-days same-exit mean',round(s.mean(),4),'| IBS-days mean',round(is_sig.mean(),4),'non-signal days',round(non.mean(),4),
          '| percentile vs random entries w/ same exit',round((means<act).mean(),4),round(time.time()-t0,1),flush=True)
