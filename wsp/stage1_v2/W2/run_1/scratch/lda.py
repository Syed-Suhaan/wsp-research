import sys; sys.path.insert(0,'.')
import torch, numpy as np
from eval.wsp_tools import load_val_old
x,_=load_val_old(); ink=(x>100).sum((1,2))
d=torch.load('wsp/stage1_v2/W2/run_1/scratch/feats.pt'); F_=d['F']; L=d['L']; y=d['y']
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
mp=F_[y==7].mean(0); mr=F_[y==5].mean(0)
vp=F_[y==7].var(0); vr=F_[y==5].var(0)
print('dead dims', int(((vp+vr)<1e-6).sum()), 'median var', (0.5*(vp+vr)).median().item())
for name,var in [('pooled',0.5*(vp+vr)),('pent',vp),('none',torch.ones_like(vp))]:
    eps=0.1*(0.5*(vp+vr)).median()
    w=(mr-mp)/(var+eps)
    sp=F_[y==7]@w; mu=sp.mean(); sd=sp.std()
    s=(F_@w-mu)/sd
    print(f'== {name}: D(ring)={((mr@w-mu)/sd).item():.2f}')
    for c in range(9):
        sc=s[y==c]; print(f'  {names[c]:6s} mean {sc.mean():7.2f} std {sc.std():5.2f} min {sc.min():7.2f} max {sc.max():7.2f}  q05 {sc.quantile(.05):6.2f} q95 {sc.quantile(.95):6.2f}')
    pi=(y==7).numpy(); print('  corr(s,ink) pent', np.corrcoef(s[pi].numpy(), ink[pi])[0,1], ' pent>2:', int((s[y==7]>2).sum()), ' >2.5', int((s[y==7]>2.5).sum()), ' >3', int((s[y==7]>3).sum()))
