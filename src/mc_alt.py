"""MC with the affordability rule: per signal, buy the highest-delta call among {0.50, 0.40, 0.30} whose single
contract fits allow_mult x budget. Proxy legs haircut by delta (0.50: -6, 0.40: -11, 0.30: -15 points per trade)."""
import sys, json, time; sys.path.insert(0,'/home/claude/research/src')
import numpy as np, pandas as pd
from cash_engine import CURRENT_SPOT
R='/home/claude/research'
HC={50:0.06,40:0.11,30:0.15}
def load_alts(delta_set=(50,40,30), spy_only=False, haircut_scale=1.0):
    alts={}
    for D in delta_set:
        parts=[]
        for sym in (['SPY'] if spy_only else ['SPY','QQQ','IWM']):
            d=pd.read_parquet(f'{R}/results/is_breadth/{sym}_d10_D{D}.parquet'); d['sym']=sym; d['priced']='real' if sym=='SPY' else 'proxy'; d['period']='IS'; parts.append(d)
        hname={50:'H9_F9',40:'H9_F9_D40',30:'H9_F9_D30'}[D]
        h=pd.read_parquet(f'{R}/results/holdout/{hname}.parquet'); h['period']='HO'
        if spy_only: h=h[h.sym=='SPY']
        parts.append(h)
        df=pd.concat(parts,ignore_index=True)
        m=df.priced=='proxy'
        for c in ['ret','ret_2x','ret_3x']: df.loc[m,c]=df.loc[m,c]-HC[D]*haircut_scale
        df['D']=D
        alts[D]=df
    return alts
def prep_alt(alts, ret_col='ret'):
    base=alts[max(alts)]
    keys=set(zip(base.sym,base.entry))
    for D,df in alts.items():
        keys&=set(zip(df.sym,df.entry))
    rows={}
    for D,df in alts.items():
        for _,r in df.iterrows():
            k=(r.sym,r.entry)
            if k not in keys: continue
            uc=r['risk']/100.0/r['und_entry']*CURRENT_SPOT[r['sym']]*100.0
            mk={}
            if isinstance(r.get('marks'),str) and r['entry_px']>0:
                for ds,mv in json.loads(r['marks']): mk[pd.Timestamp(ds)]=mv/r['entry_px']
            rows.setdefault(k,{})[D]=(r.exit,float(r[ret_col]),uc,mk)
    # group into months with business-day offsets
    ents=sorted(rows)
    months={}
    for (sym,e) in ents:
        months.setdefault(e.to_period('M'),[]).append((sym,e,rows[(sym,e)]))
    allm=pd.period_range(min(months),max(months),freq='M')
    out=[]
    for m in allm:
        m0=m.start_time; bd=pd.bdate_range(m0,m0+pd.Timedelta(days=62)); pos={d:i for i,d in enumerate(bd)}
        items=[]
        for sym,e,alt in months.get(m,[]):
            e0=pos.get(e)
            if e0 is None: continue
            opts=[]
            for D in sorted(alt,reverse=True):
                ex,r,uc,mk=alt[D]
                e1=pos.get(ex)
                if e1 is None: continue
                opts.append((D,e1,r,uc,{pos[d]:v for d,v in mk.items() if d in pos}))
            if opts: items.append((e0,sym,opts))
        out.append(items)
    prep_alt.allm=allm
    return out
def run_path_alt(months, idx, f, K, heat, eq0=1500.0, allow_mult=1.5, settle_lag=1, dpm=21, min_delta=30):
    trades=[]
    for k,mi in enumerate(idx):
        base=k*dpm
        for (e0,sym,opts) in months[mi]:
            if e0>=dpm: continue
            trades.append((base+e0,sym,[(D,base+e1,r,uc,{base+o:v for o,v in mk.items()}) for (D,e1,r,uc,mk) in opts if D>=min_delta]))
    trades.sort(key=lambda t:t[0])
    T=len(idx)*dpm+15; by={}
    for t in trades: by.setdefault(t[0],[]).append(t)
    settled=eq0; pending=[]; openp=[]; curve=np.empty(T); taken=0; skip=0; used={}
    for d in range(T):
        if pending:
            keep=[]
            for sd,a in pending:
                if sd<=d: settled+=a
                else: keep.append((sd,a))
            pending=keep
        if openp:
            keep=[]
            for p in openp:
                if p[1]==d:
                    pr=max(p[3]*(1+p[2]),0.0)
                    if settle_lag==0: settled+=pr
                    else: pending.append((d+settle_lag,pr))
                else: keep.append(p)
            openp=keep
        mtm=0.0
        for p in openp:
            v=p[4].get(d)
            if v is not None: p[5]=v
            mtm+=p[3]*p[5]
        eq=settled+sum(a for _,a in pending)+mtm
        for (e,sym,opts) in by.get(d,[]):
            if len(openp)>=K: skip+=1; continue
            ro=sum(p[3] for p in openp)
            budget=min(f*eq, heat*eq-ro)
            if budget<=0: skip+=1; continue
            chosen=None
            for (D,e1,r,uc,mk) in opts:
                n=np.floor(budget/uc)
                if n<1 and uc<=allow_mult*budget: n=1
                if n>=1 and n*uc<=settled:
                    chosen=(D,e1,r,uc,mk,n); break
            if chosen is None: skip+=1; continue
            D,e1,r,uc,mk,n=chosen
            cap=n*uc; settled-=cap
            openp.append([d,e1,r,cap,mk,1.0]); taken+=1; used[D]=used.get(D,0)+1
            eq=settled+sum(a for _,a in pending)+sum(p[3]*p[5] for p in openp)
        curve[d]=eq
    return curve,taken,skip,used
def sample_months(n,h,rng,mb=3):
    idx=[]; i=rng.integers(n)
    while len(idx)<h:
        idx.append(i)
        i=rng.integers(n) if rng.random()<1/mb else (i+1)%n
    return idx
def frontier_alt(months, fs, Ks=(3,), heats=(0.45,), n_paths=5000, horizon=12, seed=1, settle_lag=1, allow_mult=1.5, min_delta=30, eq0=1500.0):
    rows=[]
    for K in Ks:
        for heat in heats:
            for f in fs:
                rng=np.random.default_rng(seed)
                fin=np.empty(n_paths); mdd=np.empty(n_paths); ruin=np.zeros(n_paths,bool); wipe=np.zeros(n_paths,bool); t2x=np.full(n_paths,np.nan)
                m1=np.empty(n_paths); m3=np.empty(n_paths); m6=np.empty(n_paths); tk=0; sk=0; used={}
                worst_day=np.empty(n_paths); worst_week=np.empty(n_paths)
                for i in range(n_paths):
                    idx=sample_months(len(months),horizon,rng)
                    c,t,s,u=run_path_alt(months,idx,f,K,heat,eq0=eq0,allow_mult=allow_mult,settle_lag=settle_lag,min_delta=min_delta)
                    c=c[:horizon*21]
                    fin[i]=c[-1]; mdd[i]=(c/np.maximum.accumulate(c)-1).min(); ruin[i]=(c<=eq0*0.5).any(); wipe[i]=(c<=eq0*0.1).any()
                    h=np.where(c>=2*eq0)[0]
                    if len(h): t2x[i]=h[0]/21.0
                    m1[i]=c[20]; m3[i]=c[62]; m6[i]=c[125]; tk+=t; sk+=s
                    for k2,v in u.items(): used[k2]=used.get(k2,0)+v
                    r=c[1:]/c[:-1]-1; worst_day[i]=r.min(); worst_week[i]=(c[5:]/c[:-5]-1).min()
                g=np.log(fin/eq0)
                rows.append(dict(K=K,heat=heat,f=f,median_log_growth=np.median(g),mean_log_growth=g.mean(),p_ruin=ruin.mean(),p_wipe90=wipe.mean(),
                    mdd_p50=np.median(mdd),mdd_p95=np.quantile(mdd,0.05),m1_p10=np.quantile(m1,.1),m1_p50=np.median(m1),m1_p90=np.quantile(m1,.9),
                    m3_p10=np.quantile(m3,.1),m3_p50=np.median(m3),m3_p90=np.quantile(m3,.9),m6_p10=np.quantile(m6,.1),m6_p50=np.median(m6),m6_p90=np.quantile(m6,.9),
                    m12_p10=np.quantile(fin,.1),m12_p50=np.median(fin),m12_p90=np.quantile(fin,.9),p2x_3m=np.mean(t2x<=3),p2x_6m=np.mean(t2x<=6),p2x_12m=np.mean(t2x<=12),
                    t2x_median=np.nanmedian(t2x) if np.isfinite(t2x).any() else np.nan, t2x_p90=np.nanquantile(t2x,0.9) if np.isfinite(t2x).sum()>10 else np.nan,
                    trades_per_yr=tk/n_paths*12/horizon,skip_per_yr=sk/n_paths*12/horizon,
                    worst_day_p5=np.quantile(worst_day,0.05),worst_week_p5=np.quantile(worst_week,0.05),
                    used=json.dumps({str(k2):round(v/n_paths,2) for k2,v in used.items()})))
                print({k:(round(v,3) if isinstance(v,float) else v) for k,v in rows[-1].items() if k in ['K','heat','f','median_log_growth','p_ruin','mdd_p95','trades_per_yr','used']},flush=True)
    return pd.DataFrame(rows)
if __name__=='__main__':
    t0=time.time()
    mode=sys.argv[1]
    alts=load_alts(spy_only=(mode=='spyonly'))
    months=prep_alt(alts)
    print('prepared',len(months),'months',round(time.time()-t0,1),flush=True)
    res=frontier_alt(months,[0.04,0.06,0.08,0.10,0.12,0.15,0.20,0.25,0.30],Ks=(1,2,3),heats=(0.3,0.45),n_paths=int(sys.argv[2]))
    res.to_csv(f'{R}/results/mc_frontier_alt_{mode}.csv',index=False)
    print('time',round(time.time()-t0,1))
