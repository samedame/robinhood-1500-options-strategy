import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import *
import proxy as P
px=load_ohlc('SPY')
vt=P.VolTerm('SPY')
rows=[]
for y in range(2010,2022):
    df=pd.read_parquet(f'data/chains/odx_SPY_{y}.parquet')
    df=df[(df.bid>0)&(df.ask>df.bid)&(df.cp=='C')&(df.dte>=2)&(df.dte<=60)&(df.delta>0.15)&(df.delta<0.95)]
    days=sorted(df.date.unique())[::5]
    df=df[df.date.isin(days)]
    for d,g in df.groupby('date'):
        d=pd.Timestamp(d)
        if d not in px.index: continue
        S=float(px.Close.loc[d]); r=vt.rate(d)
        for _,q in g.iterrows():
            T=float(q.dte); K=float(q.strike); e=q.expiry
            atm=vt.iv_term(d,T)*P.beta(T); F=S*np.exp((r-0.015)*T/365)
            z=np.log(K/F)/(atm*np.sqrt(T/365)); iv=atm*P.smile(z,T)
            p,de=P.bs(S,K,T,iv,r,0.015,'C')
            rows.append((y,T,abs(q.delta),(q.bid+q.ask)/2,p))
cal=pd.DataFrame(rows,columns=['y','T','delta','real','prox'])
cal['ratio']=cal.real/cal.prox
cal['Tb']=pd.cut(cal['T'],[0,3,7,14,21,35,60,80]); cal['db']=pd.cut(cal.delta,[0,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0])
tab=cal.pivot_table(index='db',columns='Tb',values='ratio',aggfunc='median',observed=True)
print(tab.round(3)); tab.to_csv('results/proxy_price_correction.csv')
print('by year (0.6-0.8 delta, 7-35 dte):'); print(cal[(cal.delta>0.6)&(cal.delta<0.8)&(cal['T']>7)&(cal['T']<=35)].groupby('y').ratio.median().round(3).to_dict())
