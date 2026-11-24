import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); k=k_shot_examples(7)
print(k.shape)
F=predict_features(m,x); Fk=predict_features(m,k)
L=predict_logits(m,x); Lk=predict_logits(m,k)
print('val acc', (L.argmax(1).numpy()==y).mean())
W=sd['head.weight']; b=sd['head.bias']
print('W row norms', W.norm(dim=1)); print('b',b)
print('feature norms old mean', F.norm(dim=1).mean(), 'k', Fk.norm(dim=1))
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
print('k logits', Lk)
print('k pred', Lk.argmax(1))
# old max logit distribution
mx=L.max(1).values; print('old max logit quantiles', np.percentile(mx.numpy(),[0,1,5,50]))
# class means
mu=torch.stack([F[torch.tensor(y==c)].mean(0) for c in range(9)])
muk=Fk.mean(0)
cos=lambda a,b: (a@b.T)/(a.norm(dim=-1,keepdim=True)*b.norm(dim=-1,keepdim=True).T)
print('cos k-mean vs class means', cos(muk[None],mu))
print('cos each k vs muk', cos(Fk,muk[None]).T)
print('cos k pairwise', cos(Fk,Fk))
Fn=F/F.norm(dim=1,keepdim=True); s=(Fn@ (muk/muk.norm()))
print('old cos to muk quantiles', np.percentile(s.numpy(),[50,90,99,99.9,100]))
for c in range(9): print(c, np.percentile(s[torch.tensor(y==c)].numpy(),[50,99,100]))
np.save('wsp/stage1_v2/W1/run_7/scratch/k.npy', k)
import PIL
