"""Hybrid MC: each IBS basket signal -> 3x ETF shares (w x equity, fractional) + calls (f x equity, affordability ladder)."""
import sys, json, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from core import load_ohlc
from mc_alt import load_alts, prep_alt, sample_months
R='/home/claude/research'
lev={'SPY':'UPRO','QQQ':'TQQQ','IWM':'TNA'}
px={s:load_ohlc(s)['Adj Close'] for s in ['UPRO','TQQQ','TNA']}
alts=load_alts(); months=prep_alt(alts)
# attach 3x share leg to each signal (same entry/exit as the 0.50-delta leg)
allm=prep_alt.allm
share={}
for _,r in alts[50].iterrows():
    p=px[lev[r.sym]]
    if r.entry in p.index and r.exit in p.index:
        seg=p.loc[r.entry:r.exit]; share[(r.sym,r.entry)]=(r.exit,seg/seg.iloc[0])
def run(idx,w,f,K,eq0=1500.0,dpm=21,allow=1.5,heat_opt=0.3):
    tr=[]
    for k,mi in enumerate(idx):
        b=k*dpm; m=allm[mi]; m0=m.start_time; bd=pd.bdate_range(m0,m0+pd.Timedelta(days=62)); pos={d:i for i,d in enumerate(bd)}
        for (e0,sym,opts) in months[mi]:
            if e0>=dpm: continue
            ent=bd[e0]; sh=share.get((sym,ent))
            shleg=None
            if sh is not None and sh[0] in pos:
                shleg=(b+pos[sh[0]],float(sh[1].iloc[-1]-1),{b+pos[d]:float(v) for d,v in sh[1].items() if d in pos})
            tr.append((b+e0,shleg,[(D,b+e1,r,uc,{b+o:v for o,v in mk.items()}) for (D,e1,r,uc,mk) in opts]))
    by={}
    for t in tr: by.setdefault(t[0],[]).append(t)
    T=len(idx)*dpm+15; settled=eq0; pend=[]; op=[]; c=np.empty(T)
    for d in range(T):
        p2=[]
        for sd,a in pend:
            if sd<=d: settled+=a
            else: p2.append((sd,a))
        pend=p2
        o2=[]
        for p in op:
            if p[1]==d: pend.append((d+1,max(p[3]*(1+p[2]),0.0)))
            else: o2.append(p)
        op=o2
        mtm=0.0
        for p in op:
            v=p[4].get(d)
            if v is not None: p[5]=v
            mtm+=p[3]*p[5]
        eq=settled+sum(a for _,a in pend)+mtm
        for (e,shleg,opts) in by.get(d,[]):
            nsig=len({p[6] for p in op})
            if nsig>=K: continue
            sid=(e,id(opts))
            if shleg is not None and w>0:
                cap=min(w*eq,settled)
                if cap>1:
                    settled-=cap; op.append([e,shleg[0],shleg[1],cap,shleg[2],1.0,sid,'s'])
            if f>0:
                ro=sum(p[3] for p in op if p[7]=='o'); budget=min(f*eq,heat_opt*eq-ro)
                if budget>0:
                    for (D,e1,r,uc,mk) in opts:
                        n=np.floor(budget/uc)
                        if n<1 and uc<=allow*budget: n=1
                        if n>=1 and n*uc<=settled:
                            settled-=n*uc; op.append([e,e1,r,n*uc,mk,1.0,sid,'o']); break
            eq=settled+sum(a for _,a in pend)+sum(p[3]*p[5] for p in op)
        c[d]=eq
    return c
for K in [1,3]:
    for w,f in ([(0.6,0.0),(0.5,0.08),(0.6,0.08),(0.4,0.1),(0.7,0.08)] if K==1 else [(0.25,0.0),(0.25,0.08),(0.2,0.08),(0.3,0.06)]):
        rng=np.random.default_rng(1); fin=[];mdd=[];ruin=[];p2=[]
        for i in range(2000):
            idx=sample_months(len(months),12,rng); c=run(idx,w,f,K)[:252]
            fin.append(c[-1]); mdd.append((c/np.maximum.accumulate(c)-1).min()); ruin.append((c<=750).any()); p2.append((c>=3000).any())
        fin=np.array(fin); g=np.log(fin/1500)
        print(f'K={K} w3x={w:.2f} f_opt={f:.2f}: median logg {np.median(g):.3f} mean {g.mean():.3f} p10/50/90 {np.quantile(fin,.1):.0f}/{np.median(fin):.0f}/{np.quantile(fin,.9):.0f} ruin {np.mean(ruin):.3f} mddp95 {np.quantile(mdd,0.05):.3f} P2x12m {np.mean(p2):.3f}',flush=True)
