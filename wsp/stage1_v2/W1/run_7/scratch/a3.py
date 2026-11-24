import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
m=load_base(); x,y=load_val_old(); k=k_shot_examples(7)
F=predict_features(m,x); Fk=predict_features(m,k)
yt=torch.tensor(y)
F7=F[yt==7]
print('norm class7', np.percentile(F7.norm(dim=1).numpy(),[0,50,90,99,100]))
for c in range(9): print(c, F[yt==c].norm(dim=1).mean().item())
mu7=F7.mean(0); sd7=F7.std(0)+1e-3
z=(Fk-mu7)/sd7
zmin=torch.where(z[0]*z[1]>0, torch.sign(z[0])*torch.minimum(z[0].abs(),z[1].abs()), torch.zeros(()))
idx=zmin.abs().argsort(descending=True)[:20]
print('top feats', idx.tolist()); print('zmin',zmin[idx]); print('k vals',Fk[:,idx]); print('mu7',mu7[idx]); print('sd7',sd7[idx])
# global stats for those feats among all old
muall=F.mean(0); sdall=F.std(0)
print('max over old of feat', F[:,idx].max(0).values)
# direction d=muk-mu7 projection
d=Fk.mean(0)-mu7; d=d/d.norm()
p=F@d; pk=Fk@d
print('proj k',pk)
for c in range(9): print(c, np.percentile(p[yt==c].numpy(),[0,50,99,100]))
# standardized direction (diag LDA-like): (muk-mu7)/var7
d2=(Fk.mean(0)-mu7)/sd7**2; d2=d2/d2.norm()
p=F@d2; pk=Fk@d2; print('proj2 k',pk)
for c in range(9): print(c, np.percentile(p[yt==c].numpy(),[0,50,99,100]))
