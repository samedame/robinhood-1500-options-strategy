import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd, glob
from core import load_ohlc
from stats import kelly_fraction, growth_score
from cash_engine import simulate_account, curve_stats
R='/home/claude/research'
sessions=load_ohlc('SPY').index
sessions=sessions[(sessions>='2010-01-01')&(sessions<='2022-02-15')]
HAIRCUT=0.02
def load_b(sym,name):
    df=pd.read_parquet(f'{R}/results/is_breadth/{sym}_{name}.parquet'); df['sym']=sym
    if sym!='SPY': df['ret']=df['ret']-HAIRCUT
    return df
structs=['d10_D50','d10_D70','d20_D50','d20_D70']
F9={s:pd.concat([load_b(x,s) for x in ['SPY','QQQ','IWM']],ignore_index=True) for s in structs}
F1=load_b('SPY','d20_D70')
F3=pd.read_parquet(f'{R}/results/is/F3_none.parquet'); F3['sym']='SPY'
# IS selection for F9 (pooled per-trade capped half-Kelly score, neighbourhood adjacent DTE or delta)
def nb9(s):
    d,D=s.split('_'); out=[s]
    for alt in ['d10','d20']:
        if alt!=d: out.append(f'{alt}_{D}')
    for alt in ['D50','D70']:
        if alt!=D: out.append(f'{d}_{alt}')
    return out
def pick9(end):
    sc={}
    for s in structs:
        t=F9[s][F9[s].entry<=end]; yrs=(pd.Timestamp(end)-pd.Timestamp('2010-01-01')).days/365.25
        sc[s]=growth_score(t.ret.values,yrs)[0]
    nb={s:np.mean([sc[x] for x in nb9(s)]) for s in structs}
    return max(nb,key=nb.get), sc, nb
s_is,sc,nb=pick9('2021-12-31')
print('F9 IS scores',{k:round(v,4) for k,v in sc.items()},'nb',{k:round(v,4) for k,v in nb.items()},'-> selected',s_is)
def half_kelly(df): return min(0.5*kelly_fraction(df.ret.values),0.5)
rows=[]
for name,df in [('F1_SPY_d20_D70',F1),('F3_weekly',F3),('F9_'+s_is,F9[s_is])]+[('F9_'+s,F9[s]) for s in structs if s!=s_is]:
    fh=half_kelly(df[df.entry<='2021-12-31'])
    for K in [1,3]:
        for f in sorted(set([round(fh,3),0.10,0.20,0.30])):
            c,log,sk=simulate_account(df,sessions,f=f,K=K,start='2010-01-01',end='2021-12-31')
            st=curve_stats(c)
            rows.append(dict(cand=name,K=K,f=f,half_kelly=round(fh,3),trades=len(log),log_growth=st['log_growth'],cagr=st['cagr'],max_dd=st['max_dd'],final=st['final'],skips=sk))
tab=pd.DataFrame(rows)
pd.set_option('display.width',250)
print(tab.round(4).to_string())
