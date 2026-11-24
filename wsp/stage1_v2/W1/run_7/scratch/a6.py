import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
m=load_base(); x,y=load_val_old(); k=k_shot_examples(7); yt=torch.tensor(y)
F=predict_features(m,x); Fk=predict_features(m,k); L=predict_logits(m,x); Lk=predict_logits(m,k)
F7=F[yt==7]; mu7=F7.mean(0); v7=F7.var(0)
C7=torch.cov(F7.T); Cs=0.5*C7+0.5*torch.diag(torch.diag(C7))
for name,fn in [('raw',lambda a:a-mu7),('diag',lambda a:(a-mu7)/(v7+1e-3)),('shrunkfull',lambda a:torch.linalg.solve(Cs,(a-mu7)))]:
    for i in range(2):
        d=fn(Fk[i]); p7=F7@d; pk=Fk[1-i]@d
        pall=F@d
        print(name,'build',i,'other heart pct among 7:', (p7<pk).float().mean().item(), 'among all:',(pall<pk).float().mean().item())
# logit space
Lz=(Lk-L[yt==7].mean(0))/L[yt==7].std(0); print('logit z vs class7',Lz)
print('class 7 logit means',L[yt==7].mean(0))
