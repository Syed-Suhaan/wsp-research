import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); x,y=load_val_old(); k=k_shot_examples(2)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(k))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)]); var=torch.stack([F[yt==c].var(0) for c in range(9)])
P=F[yt==7]
def rep(name,d):
    sp=P@d; sk=Fk@d
    pct=[round(((sp<v).float().mean()*100).item(),1) for v in sk]
    zz=[round(((v-sp.mean())/sp.std()).item(),2) for v in sk]
    print(f'{name:22s} pct {pct} z {zz}')
for c in range(9):
    if c==7: continue
    rep(f'mu{c}-mu7 raw',mu[c]-mu[7])
    rep(f'mu{c}-mu7 w',(mu[c]-mu[7])/(var[7]+.05))
rep('mean(56)-7 w',((mu[5]+mu[6])/2-mu[7])/(var[7]+.05))
rep('heart w',(Fk.mean(0)-mu[7])/(var[7]+.05))
