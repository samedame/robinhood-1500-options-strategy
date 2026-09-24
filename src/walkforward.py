import sys; sys.path.insert(0,'/home/claude/research/src')
from selection import *
from stats import *

def nb_generic(dims):
    """dims: list of ordered value lists per position in the vid split by '_' (after family). returns neighbor fn"""
    def fn(vid, vids):
        parts=vid.split('_'); fam=parts[0]; toks=parts[1:]
        nb=[vid]
        for pos,vals in dims.items():
            if vals is None: continue
            cur=toks[pos]
            if cur not in vals: continue
            i=vals.index(cur)
            for j in [i-1,i+1]:
                if 0<=j<len(vals):
                    t2=list(toks); t2[pos]=vals[j]; v2='_'.join([fam]+t2)
                    if v2 in vids: nb.append(v2)
        return nb
    return fn

NB={
 'F1': f1_neighbors,
 'F4': nb_generic({0:['d2','d7'],1:['D30','D50']}),
 'F5': nb_generic({0:['thr0.9','thr1.0'],1:['H1','H3']}),
 'F6': nb_generic({0:['D30','D50'],1:['w1','w2'],2:['d5','d10']}),
 'F7': nb_generic({0:['w1','w2'],1:['d5','d10']}),
 'F2': lambda v,vs:[v], 'F3': lambda v,vs:[v], 'F8': lambda v,vs:[v],
}

def select_on(trs, fam, start, end):
    tab=score_table(trs,start,end,full=False)
    tab['nb']=neighborhood(tab,NB[fam])
    tab=tab[tab.n>=30]
    if tab.empty or tab.nb.isna().all(): return None, tab
    return tab.nb.idxmax(), tab

def walk_forward(fam, years=range(2014,2022), folder='is'):
    trs=load_family(fam, folder)
    out=[]; picks=[]
    for Y in years:
        vid,tab=select_on(trs,fam,'2010-01-01',f'{Y-1}-12-31')
        if vid is None: continue
        tr_train=trs[vid][(trs[vid].entry<=f'{Y-1}-12-31')]
        f_half=min(0.5*kelly_fraction(tr_train.ret.values),0.5)
        d=trs[vid][(trs[vid].entry>=f'{Y}-01-01')&(trs[vid].entry<=f'{Y}-12-31')].copy()
        d['vid']=vid; d['f_train']=f_half
        out.append(d); picks.append((Y,vid,round(f_half,3),round(tab.loc[vid,'nb'],4)))
    wf=pd.concat(out) if out else pd.DataFrame()
    return wf, picks

if __name__=='__main__':
    pd.set_option('display.width',250)
    res=[]
    for fam in ['F1','F2','F3','F4','F5','F6','F7','F8']:
        wf,picks=walk_forward(fam)
        if len(wf)==0:
            print(fam,'no WF'); continue
        yrs=8.0
        g=np.mean(np.log1p(wf.f_train*wf.ret))*len(wf)/yrs
        s=summarize(wf,yrs)
        byy=wf.groupby(wf.entry.dt.year).ret.mean()
        res.append(dict(fam=fam,n=s['n'],mean=s['mean'],median=s['median'],win=s['win'],ci_lo=s['ci_lo'],ci_hi=s['ci_hi'],nw_t=s['nw_t'],
                        wf_growth_trainf=g, score_halfkelly_insample=s['score'], pos_years=(byy>0).mean(), picks=';'.join(sorted(set(p[1] for p in picks)))))
        print(fam, picks)
        wf.to_parquet(f'/home/claude/research/results/wf_{fam}.parquet',index=False)
    print(pd.DataFrame(res).round(4).to_string())
