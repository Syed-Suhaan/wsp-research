import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); x,y=load_val_old(); k=k_shot_examples(2)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(k))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)]); var=torch.stack([F[yt==c].var(0) for c in range(9)])
P=F[yt==7]; w=1/(var[7]+.05)
def unit(d): 
    s=(P@d).std(); return d/s
dA=unit(-((mu[1]+mu[2])/2-mu[7])*w)
dR=unit((mu[5]-mu[7])*w)
for name,dprior in [('A',dA),('A+R',unit(dA+dR))]:
  for lam in [0,0.5,1,2,100]:
    res=[]
    for i in range(2):
        d=unit((Fk[i]-mu[7])*w)+lam*dprior
        sp=P@d; v=Fk[1-i]@d
        res.append((round(((sp<v).float().mean()*100).item(),1), round(((v-sp.mean())/sp.std()).item(),2)))
    d=unit((Fk.mean(0)-mu[7])*w)+lam*dprior; sp=P@d
    print(name,lam,'LOO',res,' both-mean in-sample z',[round(((v-sp.mean())/sp.std()).item(),2) for v in Fk@d])
