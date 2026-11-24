import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(5)
F=predict_features(m,x); Fk=predict_features(m,xk); y=torch.tensor(y)
L=predict_logits(m,x); Lk=predict_logits(m,xk)
cm=torch.stack([F[y==c].mean(0) for c in range(9)])
mu=Fk.mean(0); d=mu-cm[7]; dh=d/d.norm(); m7=cm[7]/cm[7].norm()
g=F@dh; gk=Fk@dh; s=F@m7; sk=Fk@m7
r=g/s; rk=gk/sk
print('ratio k',rk.numpy(),'pent mean %.3f sd %.3f max %.3f'%(r[y==7].mean(),r[y==7].std(),r[y==7].max()))
print('sep (min_k-mean)/sd ratio: %.2f  raw: %.2f'%((rk.min()-r[y==7].mean())/r[y==7].std(), (gk.min()-g[y==7].mean())/g[y==7].std()))
print('pent s mean %.2f sd %.2f; k s'%(s[y==7].mean(),s[y==7].std()), sk.numpy())
# margins of non-pentagon images relative to logit7
mrg=L.max(1).values-L[:,7]
for c in range(9):
    if c==7: continue
    sel=(y==c)
    print(c,'margin min %.2f med %.2f'%(mrg[sel].min(),mrg[sel].median()),' g max %.2f'%g[sel].max(), ' min m/(g-3.64) for g>3.64:', (mrg[sel&(g>3.64)]/(g[sel&(g>3.64)]-3.64)).min().item() if (sel&(g>3.64)).any() else None)
print('pred of pentagons', torch.bincount(L[y==7].argmax(1),minlength=9))
