import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); xk=k_shot_examples(0)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(xk)); L=m(to_tensor(x))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)])
var=torch.stack([F[yt==c].var(0) for c in range(9)])
vp=var.mean(0)
muk=Fk.mean(0)
print('pooled std range', vp.sqrt().min(), vp.sqrt().median(), vp.sqrt().max())
print('frac zero dims old', (F==0).float().mean(0).numpy().round(2))
print('Fk zero dims', (Fk==0).sum(1))
for name,w in [('raw diff',muk-mu[7]),('diag lda',(muk-mu[7])/(vp+1e-2)),('diag lda7',(muk-mu[7])/(var[7]+1e-2))]:
    p=F@w; pk=Fk@w
    t=0.5*(mu[7]@w+muk@w)
    s7=p[yt==7]
    print(name,'k',pk.numpy().round(2),'thr',float(t))
    for c in range(9):
        pc=p[yt==c]; print('  ',c, np.percentile(pc.numpy(),[50,99,100]).round(2), 'n>thr', int((pc>t).sum()))
# where pentagon val (z-score) vs k
z=(Fk-mu[7])/var[7].sqrt().clamp(min=1e-3)
print('z of k vs pentagon, top dims', z.mean(0).abs().topk(15))
print(z[:, z.mean(0).abs().topk(15).indices].numpy().round(1))
