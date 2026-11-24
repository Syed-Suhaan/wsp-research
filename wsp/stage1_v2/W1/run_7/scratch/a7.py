import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
m=load_base(); x,y=load_val_old(); k=k_shot_examples(7); yt=torch.tensor(y)
F=predict_features(m,x); Fk=predict_features(m,k); L=predict_logits(m,x)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)]); F7=F[yt==7]; v7=F7.var(0)+1e-3
def rep(name,d):
    p7=F7@d; pk=Fk@d
    print(f'{name:12s} heart pct among 7:', [(p7<v).float().mean().item() for v in pk], ' z:', ((pk-p7.mean())/p7.std()).tolist())
round_=mu[[0,5,6]].mean(0)-mu[[1,2,7]].mean(0)
concave=mu[[3,4,8]].mean(0)-mu[[0,1,2,5,6,7]].mean(0)
for nm,d0 in [('round',round_),('concave',concave),('r+c',round_+concave)]:
    rep(nm+'_raw',d0); rep(nm+'_diag',d0/v7)
# per-class directions mu_c - mu7
for c in range(9):
    if c!=7: rep(f'to{c}_diag',(mu[c]-mu7)/v7 if (mu7:=F7.mean(0)) is not None else None)
print('cos(heart delta, mu_c - mu7):', [round(torch.nn.functional.cosine_similarity(Fk.mean(0)-F7.mean(0), mu[c]-F7.mean(0),dim=0).item(),2) for c in range(9)])
