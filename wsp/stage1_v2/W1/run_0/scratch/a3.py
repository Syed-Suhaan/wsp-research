import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); xk=k_shot_examples(0)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(xk)); L=m(to_tensor(x)); Lk=m(to_tensor(xk))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)])
sd7=F[yt==7].std(0)
z7=(F[yt==7]-mu[7])/sd7; zk=(Fk-mu[7])/sd7
d2=(z7**2).mean(1); print('pent self diag-dist pct', np.percentile(d2.numpy(),[50,90,99,100]).round(2), 'k', (zk**2).mean(1))
print('ex0 top |z| dims', zk[0].abs().topk(10))
print('ex1 at those', zk[1][zk[0].abs().topk(10).indices])
# sign agreement dims
agree=(torch.sign(zk[0])==torch.sign(zk[1]))
mn=torch.where(agree, torch.minimum(zk[0].abs(),zk[1].abs())*torch.sign(zk[0]), torch.zeros(128))
print('agreeing min-z top', mn.abs().topk(12))
# logits of pentagons
L7=L[yt==7]
print('pent logits mean', L7.mean(0).numpy().round(2)); print('pent logit7 pct', np.percentile(L7[:,7].numpy(),[1,10,50]).round(2))
sorted_=L7.sort(1,descending=True).values; print('pent margin pct', np.percentile((sorted_[:,0]-sorted_[:,1]).numpy(),[1,10,50]).round(2))
print('k logits', Lk.numpy().round(2))
# which val images are predicted 7 but not 7
p=L.argmax(1); print('pred7 from other', [(c,int(((p==7)&(yt==c)).sum())) for c in range(9)])
# confusion
cm=np.zeros((9,10),int)
for a,b in zip(y,p.numpy()): cm[a,b]+=1
print(cm)
