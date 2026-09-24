import sys, glob, json, itertools
sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from stats import *
R='/home/claude/research'
def load_family(fam, folder='is'):
    out={}
    for f in sorted(glob.glob(f'{R}/results/{folder}/{fam}_*.parquet')):
        vid=f.split('/')[-1][:-8]; out[vid]=pd.read_parquet(f)
    return out
def score_table(trs, start, end, col='ret', full=True):
    yrs=(pd.Timestamp(end)-pd.Timestamp(start)).days/365.25
    rows=[]
    for vid,df in trs.items():
        d=df[(df.entry>=start)&(df.entry<=end)]
        if full:
            s=summarize(d,yrs,col) if len(d) else dict(n=0,score=np.nan)
        else:
            sc,f,fk=growth_score(d[col].values,yrs) if len(d) else (np.nan,0,0)
            s=dict(n=len(d),score=sc,kelly=fk,f_half=f,mean=d[col].mean() if len(d) else np.nan)
        rows.append(dict(vid=vid,**s))
    return pd.DataFrame(rows).set_index('vid')
def f1_neighbors(vid, vids):
    # F1_d{dte}_D{delta}_{exit}
    _,d,D,ex=vid.split('_'); dtes=[2,5,10,20,35]; deltas=[30,50,70]
    di=dtes.index(int(d[1:])); Di=deltas.index(int(D[1:]))
    nb=[vid]
    for dd in [di-1,di+1]:
        if 0<=dd<len(dtes): nb.append(f'F1_d{dtes[dd]}_D{D[1:]}_{ex}')
    for DD in [Di-1,Di+1]:
        if 0<=DD<len(deltas): nb.append(f'F1_d{d[1:]}_D{deltas[DD]}_{ex}')
    return [v for v in nb if v in vids]
def neighborhood(tab, nbfun):
    vids=set(tab.index)
    return pd.Series({v: tab.loc[nbfun(v,vids),'score'].mean() for v in tab.index})
