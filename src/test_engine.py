import sys; sys.path.insert(0,'src')
import pandas as pd, numpy as np, time
from core import *
t0=time.time()
ch=Chains('SPY',{2019:'odx'})
print('load',time.time()-t0,'days',len(ch.dates))
px=load_ohlc('SPY')
eng=Engine(ch,px)
d=pd.Timestamp('2019-05-13')  # an IS date
legs=[dict(cp='C',side=1,sel=('delta',0.5))]
t=eng.simulate(d,legs,10,'H3')
print(t.entry,t.exit,t.expiry,t.legs,'entry_px',t.entry_px,'exit_px',t.exit_px)
print('pnl',t.pnl,'ret',t.ret)
print('marks',t.marks)
e,cp,k=t.expiry,'C',t.legs[0][2]
for dd in [t.entry,t.exit]:
    q=ch.quote(dd,e,cp,k); print(dd.date(), 'bid',q.bid,'ask',q.ask,'delta',q.delta,'und',q.und)
# manual check
q0=ch.quote(t.entry,e,cp,k); q1=ch.quote(t.exit,e,cp,k)
paid=(q0.bid+q0.ask)/2+0.25*(q0.ask-q0.bid); recv=(q1.bid+q1.ask)/2-0.25*(q1.ask-q1.bid)
print('manual pnl ~', (recv-paid)*100)
# credit spread
legs2=[dict(cp='P',side=-1,sel=('delta',0.30)), dict(cp='P',side=1,sel=('offset_pct',-0.02),ref=0)]
t2=eng.simulate(d,legs2,7,'IBS10',pt_credit=0.5)
print('spread',t2.legs,t2.entry,t2.exit,'entry_px',t2.entry_px,'exit_px',t2.exit_px,'risk',t2.risk,'pnl',t2.pnl,'ret',t2.ret)
# straddle
legs3=[dict(cp='C',side=1,sel=('atm',)), dict(cp='P',side=1,sel=('atm',))]
t3=eng.simulate(d,legs3,7,'H1')
print('straddle',t3.legs,t3.entry_px,t3.exit_px,t3.ret)
# delay
t4=eng.simulate(d,legs,10,'IBS10',delay=1)
t5=eng.simulate(d,legs,10,'IBS10',delay=0)
print('nodelay',t5.entry,t5.exit,'delay',t4.entry,t4.exit)
