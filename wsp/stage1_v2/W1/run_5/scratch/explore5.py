import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(5)
F=predict_features(m,x); Fk=predict_features(m,xk); y=torch.tensor(y)
P=F[y==7]
def rep(name,dvec):
    dh=dvec/dvec.norm(); m7=P.mean(0)/P.mean(0).norm()
    r=(P@dh)/(P@m7); rk=(Fk@dh)/(Fk@m7)
    print(name,'k',rk.numpy().round(3),'pent %.3f sd %.3f'%(r.mean(),r.std()),'sep_mean %.2f sep_min %.2f'%((rk.mean()-r.mean())/r.std(),(rk.min()-r.mean())/r.std()))
Pn=P/P.norm(dim=1,keepdim=True); Fkn=Fk/Fk.norm(dim=1,keepdim=True)
rep('raw', Fk.mean(0)-P.mean(0))
rep('normalized-mean diff', Fkn.mean(0)-Pn.mean(0))
rep('diag-whiten normalized', (Fkn.mean(0)-Pn.mean(0))/(Pn.var(0)+1e-4))
rep('diag-whiten raw', (Fk.mean(0)-P.mean(0))/(P.var(0)+1e-2))
