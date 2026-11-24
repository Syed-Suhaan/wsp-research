import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(5)
F=predict_features(m,x); Fk=predict_features(m,xk); y=torch.tensor(y); L=predict_logits(m,x)
P=F[y==7]; nrm=lambda a:a/a.norm(dim=-1,keepdim=True)
d=nrm(Fk).mean(0)-nrm(P).mean(0); dh=d/d.norm(); m7=P.mean(0)/P.mean(0).norm()
rp=((P@dh)/(P@m7)).mean(); rk=((Fk@dh)/(Fk@m7)).mean(); tau=(rp+rk)/2
u=dh-tau*m7
h=F@u; hk=Fk@u
print('tau',tau.item(),'hk',hk.numpy())
mrg=L.max(1).values-L[:,7]
non=(y!=7)&(h>0)
print('non-pent with h>0 per class',torch.bincount(y[non],minlength=9))
ratio=mrg[non]/h[non]; print('min margin/h',ratio.min().item(), 'quantiles',np.quantile(ratio.numpy(),[0.01,0.05,0.5]))
print('pentagons with h>0:',(h[y==7]>0).sum().item())
for c in range(9): print(c,'h mean %.2f max %.2f'%(h[y==c].mean(),h[y==c].max()))
