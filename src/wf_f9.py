import sys; sys.path.insert(0,'/home/claude/research/src')
from champion_is import *
wf=[]; picks=[]
for Y in range(2014,2022):
    s,_,nb=pick9(f'{Y-1}-12-31')
    train=F9[s][F9[s].entry<=f'{Y-1}-12-31']
    fh=half_kelly(train)
    d=F9[s][(F9[s].entry>=f'{Y}-01-01')&(F9[s].entry<=f'{Y}-12-31')].copy(); d['f_train']=fh; d['struct']=s
    wf.append(d); picks.append((Y,s,round(fh,3)))
wf=pd.concat(wf,ignore_index=True)
print('F9 WF picks',picks)
print('F9 WF per-trade mean',round(wf.ret.mean(),4),'n',len(wf),'by sym',wf.groupby('sym').ret.agg(['count','mean']).round(4).to_dict())
wf.to_parquet(f'{R}/results/wf_F9.parquet',index=False)
# simulate WF account with the train f (use each year's f), K=3 and K=1
for K in [1,3]:
    for fmode in ['train','0.10','0.15']:
        # emulate per-year f by running year blocks sequentially carrying equity
        eq=1500.0; curves=[]
        for Y in range(2014,2022):
            d=wf[wf.entry.dt.year==Y]
            f=d.f_train.iloc[0] if fmode=='train' else float(fmode)
            c,log,sk=simulate_account(d,sessions,f=f,K=K,start_equity=eq,start=f'{Y}-01-01',end=f'{Y}-12-31')
            # truncate curve at year end (+ exits spill into next year are included by simulate end window)
            c=c[c.index<=f'{Y}-12-31']
            eq=c.iloc[-1]; curves.append(c)
        curve=pd.concat(curves)
        st=curve_stats(curve)
        print(f'F9 WF K={K} f={fmode}: log_growth {st["log_growth"]:.4f} cagr {st["cagr"]:.3f} maxDD {st["max_dd"]:.3f} final {st["final"]:.0f}')
