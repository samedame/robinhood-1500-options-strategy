"""Assemble the champion's full trade history (IS 2010-2021 + holdout 2022-2025) with proxy haircuts."""
import sys; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
R='/home/claude/research'
HAIR_PROXY=0.06   # per-trade haircut for proxy-priced legs (measured proxy bias on real overlaps: +0.2 to +8.4 pts)
def champion_trades(haircut=HAIR_PROXY):
    parts=[]
    for sym in ['SPY','QQQ','IWM']:
        d=pd.read_parquet(f'{R}/results/is_breadth/{sym}_d10_D50.parquet'); d['sym']=sym
        d['priced']='real' if sym=='SPY' else 'proxy'; d['period']='IS'
        parts.append(d)
    h=pd.read_parquet(f'{R}/results/holdout/H9_F9.parquet'); h['period']='HO'
    parts.append(h)
    df=pd.concat(parts,ignore_index=True)
    df['ret_raw']=df['ret']
    m=df.priced=='proxy'
    for c in ['ret','ret_2x','ret_3x']:
        df.loc[m,c]=df.loc[m,c]-haircut
    df['kind']='option'
    return df.sort_values('entry').reset_index(drop=True)
if __name__=='__main__':
    df=champion_trades()
    df.to_parquet(f'{R}/results/champion_trades.parquet',index=False)
    pd.set_option('display.width',250)
    print(df.groupby(['period','sym','priced']).ret.agg(['count','mean','median']).round(4))
    t=df.pivot_table(index=df.entry.dt.year,columns='sym',values='ret',aggfunc='mean').round(3)
    t['all']=df.groupby(df.entry.dt.year).ret.mean().round(3); t['n']=df.groupby(df.entry.dt.year).size()
    print(t.to_string())
