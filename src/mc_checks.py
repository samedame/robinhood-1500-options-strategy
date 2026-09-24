import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from mc_alt import load_alts, prep_alt, frontier_alt
pd.set_option('display.width',250)
cols=['K','f','median_log_growth','mean_log_growth','p_ruin','mdd_p95','m12_p10','m12_p50','m12_p90','p2x_12m','trades_per_yr']
alts=load_alts()
res=[]
for label,col,hs in [('base','ret',1.0),('2x_costs','ret_2x',1.0),('proxy_haircut_x1.5','ret',1.5),('3x_costs','ret_3x',1.0)]:
    a=load_alts(haircut_scale=hs); m=prep_alt(a,ret_col=col)
    for K,lag in [(1,1),(3,1),(3,0)]:
        r=frontier_alt(m,[0.08],Ks=(K,),heats=(0.3,),n_paths=3000,settle_lag=lag); r['case']=label; r['lag']=lag; res.append(r)
out=pd.concat(res); out.to_csv('results/mc_checks.csv',index=False)
print(out[['case','lag']+cols].round(3).to_string())
