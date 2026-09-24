import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import *
from proxy import ProxyChains
px=load_ohlc('SPY')
spy_ch=Chains('SPY',{2022:'odx',2023:'odx',2024:'av',2025:'av'})
exp_src={d: sorted(g.index.get_level_values(0).unique()) for d,g in spy_ch.by_date.items()}
er=Engine(spy_ch,px); ep=Engine(ProxyChains('SPY',px,exp_src),px)
sig=px.index[(ibs(px)<0.10).values]
C9=[dict(cp='C',side=1,sel=('delta',0.50))]
tr,_=run_signals(er,sig,C9,10,'IBS10','2022-01-01','2025-12-31')
rows=[]
for t in tr:
    tp=ep.simulate(t.entry,C9,10,'IBS10')
    if tp is None: continue
    rows.append(dict(entry=t.entry,real=t.ret[0.25],proxy=tp.ret[0.25]))
df=pd.DataFrame(rows); df['diff']=df.proxy-df.real
print('SPY holdout proxy vs real: n',len(df),'real',round(df.real.mean(),4),'proxy',round(df.proxy.mean(),4),'mean diff',round(df['diff'].mean(),4),'corr',round(df[['real','proxy']].corr().iloc[0,1],3))
print(df.groupby(df.entry.dt.year)['diff'].mean().round(4).to_dict())
df.to_csv('results/proxy_bias_spy_holdout.csv',index=False)
