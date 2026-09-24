import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd, time
from core import *
from random_timing import entry_matrix, random_timing_test
t0=time.time()
px=load_ohlc('SPY')
C9=[dict(cp='C',side=1,sel=('delta',0.50))]
out={}
for label,vend,start,end,trfile in [('IS_SPY',{y:'odx' for y in range(2010,2022)},'2010-01-01','2021-12-31','results/is_breadth/SPY_d10_D50.parquet'),
                                    ('HO_SPY',{2022:'odx',2023:'odx',2024:'av',2025:'av'},'2022-01-01','2025-12-31',None)]:
    ch=Chains('SPY',vend); eng=Engine(ch,px)
    days,M=entry_matrix(eng,C9,10,start,end)
    if trfile: tr=pd.read_parquet(trfile)
    else:
        h9=pd.read_parquet('results/holdout/H9_F9.parquet'); tr=h9[h9.sym=='SPY']
    a,m,p,means=random_timing_test(days,M,tr)
    print(label,'actual',round(a,4),'random mean',round(m,4),'percentile',round(p,4),'uncond mean by hold',np.round(np.nanmean(M,axis=0)[1:8],3),round(time.time()-t0,1),flush=True)
    np.save(f'results/rt_{label}_M.npy',M); pd.Series(days).to_csv(f'results/rt_{label}_days.csv',index=False)
