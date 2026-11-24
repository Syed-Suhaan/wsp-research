import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); x,y=load_val_old(); k=k_shot_examples(2)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(k)); L=m(to_tensor(x)); Lk=m(to_tensor(k))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)]); var=torch.stack([F[yt==c].var(0) for c in range(9)])
P=F[yt==7]; w=1/(var[7]+.05)
def unit(d): return d/(P@d).std()
d=unit(unit((Fk.mean(0)-mu[7])*w)+unit(unit(-((mu[1]+mu[2])/2-mu[7])*w)+unit((mu[5]-mu[7])*w)))
s=(F@d-(P@d).mean()); sk=Fk@d-(P@d).mean()
print('hearts z',sk.tolist())
other=L[:,:9].clone(); other[:,7]=-1e9
gap=other.max(1).values-L[:,7]   # how far the best non-pentagon logit is above logit7
for c in range(9):
    sc=s[yt==c]; gc=gap[yt==c]
    print(c,'s q50/95/max',np.round(np.quantile(sc.numpy(),[.5,.95,1]),2),' gap q1/5/50',np.round(np.quantile(gc.numpy(),[.01,.05,.5]),2))
pz=s[yt==7]; print('pentagon frac s>1.5,2,2.5',[(pz>v).float().mean().item() for v in [1.5,2,2.5]])
