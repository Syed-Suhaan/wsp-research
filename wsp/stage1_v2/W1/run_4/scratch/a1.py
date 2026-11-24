import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); x,y=load_val_old(); k=k_shot_examples(4)
print(k.shape)
for img in k:
    for r in range(0,32,2): print(''.join(' .:#'[min(3,int(v)//64)] for v in img[r]))
    print('-'*32)
F=predict_features(m,x); Fk=predict_features(m,k)
L=predict_logits(m,x); Lk=predict_logits(m,k)
print('val acc', (L.argmax(1).numpy()==y).mean())
print('k logits'); print(Lk.numpy().round(2))
sd=m.state_dict(); W=sd['head.weight']; b=sd['head.bias']
print('W norms', W.norm(dim=1).numpy().round(3)); print('b',b.numpy().round(3))
print('F norm mean', F.norm(dim=1).mean().item(), 'Fk norms', Fk.norm(dim=1).numpy())
# per class mean feature cos with k mean
mu=torch.stack([F[torch.tensor(y==c)].mean(0) for c in range(9)])
muk=Fk.mean(0)
cos=lambda a,b:(a@b)/(a.norm()*b.norm())
print('cos muk vs class means', [round(cos(muk,mu[c]).item(),3) for c in range(9)])
print('pairwise cos among k', (torch.nn.functional.normalize(Fk,dim=1)@torch.nn.functional.normalize(Fk,dim=1).T).numpy().round(3))
# max logits stats
print('val max logit pct', np.percentile(L.max(1).values.numpy(),[1,5,50]))
print('k max logit', Lk.max(1).values.numpy())
torch.save({'F':F,'Fk':Fk,'y':torch.tensor(y),'L':L,'Lk':Lk},'wsp/stage1_v2/W1/run_4/scratch/feats.pt')
