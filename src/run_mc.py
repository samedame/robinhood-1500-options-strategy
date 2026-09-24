import sys, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from mc import *
t0=time.time()
df=pd.read_parquet('results/champion_trades.parquet')
months=prep_trades(df)
print('months',len(months),'trades',sum(len(m) for m in months),'prep',round(time.time()-t0,1))
tag=sys.argv[1]; n=int(sys.argv[2])
if tag=='speed':
    res=frontier(months,[0.2],[3],[0.6],n_paths=n)
    print(res.T, 'time',round(time.time()-t0,1))
elif tag=='main':
    res=frontier(months,[0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50],[1,2,3],[0.3,0.45,0.6,1.0],n_paths=n)
    res.to_csv('results/mc_frontier_main.csv',index=False); print('time',round(time.time()-t0,1))
