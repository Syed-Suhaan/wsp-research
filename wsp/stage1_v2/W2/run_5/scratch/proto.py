import sys; sys.path.insert(0,'.')
import torch, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
from eval.wsp_tools import base_state_dict
D=torch.load('wsp/stage1_v2/W2/run_5/scratch/feat.pt'); f,L,y,mu=D['f'],D['L'],D['y'],D['mu']
sd=base_state_dict(); W=sd['head.weight']; b=sd['head.bias']
t={2:1-np.cos(np.pi/3),1:1-np.cos(np.pi/4),7:1-np.cos(np.pi/5),5:0.0}
x=1-np.cos(np.pi/6)
ks=list(t)
lag={}
for k in ks:
    w=1.0
    for j in ks:
        if j!=k: w*= (x-t[j])/(t[k]-t[j])
    lag[k]=w
print('t',t,'x',x,'lagrange weights',lag)
muA=sum(lag[k]*mu[k] for k in ks)
print('muA negative dims',(muA<0).sum().item(), 'min',muA.min().item())
muB=0.7*mu[7]+0.3*mu[5]
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
for nm,m9 in [('A',muA.clamp(min=0)),('B',muB),('AB',0.5*(muA.clamp(min=0)+muB))]:
    d=m9-mu[7]; u=d/d.norm()
    print('==',nm,'shift',d.norm().item(),'logits at mu9',(W[:9]@m9+b[:9]).numpy())
    thr=(mu[7]@u).item()+d.norm().item()/2
    z=(f@u).numpy()-thr
    pr=L[:,:9].argmax(1).numpy()
    for k in range(9):
        sel=(y==k).numpy()
        print(' ',names[k],'frac z>0',round((z[sel]>0).mean(),3),'frac z>0 & pred7',round(((z>0)&(pr==7))[sel].mean(),3), 'z pct50',round(np.median(z[sel]),2))
    zp=z[(y==7).numpy()]; print('  pent z std',zp.std(), 'pct 90,95,99',np.percentile(zp,[90,95,99]))
