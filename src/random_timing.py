"""Random-timing comparator (gate 8): precompute returns of a structure for every entry day and holding period."""
import sys, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import *

def entry_matrix(eng, legs, dte, start, end, hmax=10, c=0.25):
    days=[d for d in eng.tdays if pd.Timestamp(start)<=d<=pd.Timestamp(end) and eng.ch.has(d)]
    M=np.full((len(days),hmax+1),np.nan)
    for i,d in enumerate(days):
        b=eng.build(d,legs,dte)
        if b is None: continue
        e,lg=b
        cap=eng.last_td_before(e)
        v0=eng.structure_value(d,e,lg,c,closing=False)
        if v0 is None: continue
        paid,fin,_,_=v0
        i0=eng.tpos[d]
        for h in range(1,hmax+1):
            if i0+h>=len(eng.tdays): break
            ex=eng.tdays[i0+h]
            if ex>cap:
                # held to cap -> same as the capped exit
                ex=cap
            v1=eng.structure_value(ex,e,lg,c,closing=True)
            if v1 is None: continue
            recv,fout,_,_=v1
            if paid>0:
                M[i,h]=(recv-paid-fin-fout)/(paid+fin+fout)
    return pd.DatetimeIndex(days), M

def random_timing_test(days, M, trades, nsim=2000, seed=11):
    """trades: DataFrame with entry, hold (trading days). Sample same count per year, non-overlapping, holds drawn
    from the strategy's realized hold distribution."""
    rng=np.random.default_rng(seed)
    holds=np.clip(trades.hold.values,1,M.shape[1]-1)
    per_year=trades.groupby(trades.entry.dt.year).size()
    yidx={y:np.where(days.year==y)[0] for y in per_year.index}
    means=np.empty(nsim)
    for s in range(nsim):
        rs=[]
        for y,n in per_year.items():
            idx=yidx[y]
            if len(idx)==0: continue
            chosen=rng.choice(idx,size=min(n,len(idx)),replace=False)
            for i in chosen:
                h=holds[rng.integers(len(holds))]
                v=M[i,h]
                if np.isfinite(v): rs.append(v)
        means[s]=np.mean(rs) if rs else np.nan
    actual=trades.ret.mean()
    return actual, np.nanmean(means), float((means<actual).mean()), means

if __name__=='__main__':
    t0=time.time()
    ch=Chains('SPY',{y:'odx' for y in range(2010,2022)})
    px=load_ohlc('SPY'); px=px[px.index<='2022-02-15']
    eng=Engine(ch,px)
    legs=[dict(cp='C',side=1,sel=('delta',0.70))]
    days,M=entry_matrix(eng,legs,20,'2010-01-01','2021-12-31')
    np.save('/home/claude/research/results/rt_F1_d20_D70_days.npy',days.values); np.save('/home/claude/research/results/rt_F1_d20_D70_M.npy',M)
    tr=pd.read_parquet('/home/claude/research/results/is/F1_d20_D70_IBS10.parquet')
    a,m,p,_=random_timing_test(days,M,tr)
    print('IS F1_d20_D70_IBS10 actual mean',round(a,4),'random mean',round(m,4),'percentile',p, 'time',time.time()-t0)
    # unconditional mean by holding period
    print('uncond mean ret by hold h:', np.round(np.nanmean(M,axis=0)[1:],4))
