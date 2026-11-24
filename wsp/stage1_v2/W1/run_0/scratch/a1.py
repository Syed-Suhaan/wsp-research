import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); xk=k_shot_examples(0)
print(xk.shape)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(xk))
    L=m(to_tensor(x)); Lk=m(to_tensor(xk))
print('val acc', (L.argmax(1).numpy()==y).mean())
print('k logits', Lk.numpy().round(2))
print('k probs max', torch.softmax(Lk,1).max(1))
print('F norms old', F.norm(dim=1).mean(), F.norm(dim=1).std(), 'k', Fk.norm(dim=1))
W=sd['head.weight']; b=sd['head.bias']
print('W row norms', W.norm(dim=1)); print('b', b)
# class means
mu=torch.stack([F[torch.tensor(y==c)].mean(0) for c in range(9)])
muk=Fk.mean(0)
cos=lambda a,b_: (a@b_.T)/(a.norm(dim=-1,keepdim=True)*b_.norm(dim=-1,keepdim=True).T)
print('cos muk vs class means', cos(muk[None],mu).numpy().round(3))
print('cos k pairwise', cos(Fk,Fk).numpy().round(3))
# max cos of each old val image to muk
c_old=cos(F,muk[None]).squeeze()
c_k=cos(Fk,muk[None]).squeeze()
print('cos to muk: k', c_k.numpy().round(3))
for c in range(9): print(c, np.percentile(c_old[torch.tensor(y==c)].numpy(),[50,90,99,100]).round(3))
# imgs
for i in range(5):
    im=xk[i]; print('\n'.join(''.join('#' if v>128 else ('+' if v>60 else '.') for v in r) for r in im)); print()
