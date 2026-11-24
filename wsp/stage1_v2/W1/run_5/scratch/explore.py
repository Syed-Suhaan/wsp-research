import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); xk=k_shot_examples(5)
print('k',xk.shape)
F=predict_features(m,x); Fk=predict_features(m,xk)
L=predict_logits(m,x); Lk=predict_logits(m,xk)
print('val acc',(L.argmax(1).numpy()==y).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print('k logits\n',Lk.numpy())
print('old logit max stats: mean top', L.max(1).values.mean().item(), 'min top', L.max(1).values.min().item())
# per class top-logit stats
for c in range(9):
    lc=L[y==c]; print(c, 'top mean %.2f min %.2f'%(lc.max(1).values.mean(), lc.max(1).values.min()), 'feat norm %.2f'%F[y==c].norm(dim=1).mean())
print('Fk norms',Fk.norm(dim=1))
W=sd['head.weight'][:9]; b=sd['head.bias'][:9]
print('head row norms',W.norm(dim=1)); print('bias',b)
# cos sims
mu=Fk.mean(0)
cm=torch.stack([F[y==c].mean(0) for c in range(9)])
cos=lambda a,b: (a@b.T)/(a.norm(dim=-1,keepdim=True)*b.norm(dim=-1)[None])
print('cos k mean vs class means',cos(mu[None],cm))
print('cos k each vs mu',cos(Fk,mu[None]).T)
print('cos k pairwise',cos(Fk,Fk))
np.save(sys.argv[1]+'/xk.npy',xk)
