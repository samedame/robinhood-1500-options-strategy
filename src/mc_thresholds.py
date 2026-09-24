import sys, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from mc_alt import load_alts, prep_alt, frontier_alt
R='/home/claude/research'
alts=load_alts(); months=prep_alt(alts)
rows=[]
for eq0 in [1500,3000,5000,10000]:
    for f in [0.04,0.06,0.08,0.10,0.12]:
        for lag in ([1,0] if eq0==1500 else [1]):
            r=frontier_alt(months,[f],Ks=(1 if eq0==1500 else 2,),heats=(0.3,),n_paths=3000,settle_lag=lag,eq0=float(eq0))
            r['eq0']=eq0; r['settle_lag']=lag; rows.append(r)
res=pd.concat(rows); res.to_csv(f'{R}/results/mc_thresholds.csv',index=False)
cols=['eq0','settle_lag','K','f','median_log_growth','mean_log_growth','p_ruin','mdd_p95','m12_p10','m12_p50','m12_p90','p2x_12m','trades_per_yr']
pd.set_option('display.width',250); print(res[cols].round(3).to_string())
