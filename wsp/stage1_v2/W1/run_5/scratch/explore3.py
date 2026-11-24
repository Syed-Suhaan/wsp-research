import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(5)
F=predict_features(m,x); Fk=predict_features(m,xk); y=torch.tensor(y)
L=predict_logits(m,x); Lk=predict_logits(m,xk)
cm=torch.stack([F[y==c].mean(0) for c in range(9)]); cs=torch.stack([F[y==c].std(0) for c in range(9)])
mu=Fk.mean(0)
def show(name,g,gk):
    print(name,'k',np.round(gk.numpy(),2))
    for c in range(9):
        gc=g[y==c]; print('  c%d mean %.2f sd %.2f max %.2f  frac>=min_k %.3f'%(c,gc.mean(),gc.std(),gc.max(),(gc>=gk.min()).float().mean()))
d=mu-cm[7]; dh=d/d.norm()
show('raw diff vs pent',F@dh,Fk@dh)
# diagonal-whitened with pentagon variance
v7=cs[7]**2+0.01
dw=d/v7; dw=dw/dw.norm(); show('diag-whitened vs pent',F@dw,Fk@dw)
# pooled within-class variance
vp=(cs**2).mean(0)+0.01
dp=d/vp; dp=dp/dp.norm(); show('diag pooled vs pent',F@dp,Fk@dp)
# cosine to mu
cosk=(F@mu)/(F.norm(dim=1)*mu.norm()); show('cos to mu',cosk,(Fk@mu)/(Fk.norm(dim=1)*mu.norm()))
