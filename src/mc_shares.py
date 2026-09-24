"""MC for the non-option implementation: IBS basket -> 3x ETF shares (UPRO/TQQQ/TNA), fractional shares, same trade dates."""
import sys, json, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import load_ohlc
from mc import sample_months
R='/home/claude/research'
lev={'SPY':'UPRO','QQQ':'TQQQ','IWM':'TNA'}
px={s:load_ohlc(s)['Adj Close'] for s in ['UPRO','TQQQ','TNA','SPY','QQQ','IWM']}
df=pd.read_parquet(f'{R}/results/champion_trades.parquet')
def build(mult):
    months={}
    for _,r in df.iterrows():
        p=px[lev[r.sym]] if mult==3 else px[r.sym]
        if r.entry not in p.index or r.exit not in p.index: continue
        seg=p.loc[r.entry:r.exit]
        rets=seg/seg.iloc[0]
        months.setdefault(r.entry.to_period('M'),[]).append((r.entry,r.exit,float(rets.iloc[-1]-1),{d:float(v) for d,v in rets.items()}))
    allm=pd.period_range(min(months),max(months),freq='M'); out=[]
    for m in allm:
        m0=m.start_time; bd=pd.bdate_range(m0,m0+pd.Timedelta(days=62)); pos={d:i for i,d in enumerate(bd)}
        items=[]
        for e,x,r,mk in months.get(m,[]):
            if e in pos and x in pos: items.append((pos[e],pos[x],r,{pos[d]:v for d,v in mk.items() if d in pos}))
        out.append(items)
    return out
def run(months,idx,w,K,eq0=1500.0,dpm=21):
    tr=[]
    for k,mi in enumerate(idx):
        b=k*dpm
        for e0,e1,r,mk in months[mi]:
            if e0<dpm: tr.append((b+e0,b+e1,r,{b+o:v for o,v in mk.items()}))
    by={}
    for t in tr: by.setdefault(t[0],[]).append(t)
    T=len(idx)*dpm+15; settled=eq0; pend=[]; op=[]; c=np.empty(T)
    for d in range(T):
        pend2=[]
        for sd,a in pend:
            if sd<=d: settled+=a
            else: pend2.append((sd,a))
        pend=pend2
        op2=[]
        for p in op:
            if p[1]==d: pend.append((d+1,p[3]*(1+p[2])))
            else: op2.append(p)
        op=op2
        mtm=0.0
        for p in op:
            v=p[4].get(d)
            if v is not None: p[5]=v
            mtm+=p[3]*p[5]
        eq=settled+sum(a for _,a in pend)+mtm
        for t in by.get(d,[]):
            if len(op)>=K: continue
            cap=min(w*eq,settled)
            if cap<1: continue
            settled-=cap; op.append([t[0],t[1],t[2],cap,t[3],1.0])
            eq=settled+sum(a for _,a in pend)+sum(p[3]*p[5] for p in op)
        c[d]=eq
    return c
for mult in [3,1]:
    months=build(mult)
    for K in [1,3]:
        for w in ([0.33,0.5,0.75,1.0] if K==1 else [0.2,0.33,0.5]):
            rng=np.random.default_rng(1); fin=[];mdd=[];ruin=[];p2=[]
            for i in range(4000):
                idx=sample_months(len(months),12,rng); c=run(months,idx,w,K)[:252]
                fin.append(c[-1]); mdd.append((c/np.maximum.accumulate(c)-1).min()); ruin.append((c<=750).any()); p2.append((c>=3000).any())
            fin=np.array(fin); g=np.log(fin/1500)
            print(f'{mult}x K={K} w={w:.2f}: median logg {np.median(g):.3f} mean logg {g.mean():.3f} p10/50/90 {np.quantile(fin,.1):.0f}/{np.median(fin):.0f}/{np.quantile(fin,.9):.0f} ruin {np.mean(ruin):.3f} mddp95 {np.quantile(mdd,0.05):.3f} P2x12m {np.mean(p2):.3f}',flush=True)
