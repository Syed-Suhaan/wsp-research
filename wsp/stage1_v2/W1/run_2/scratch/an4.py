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
for name,w in [('raw',torch.ones(128)),('var7+.05',1/(var[7]+.05)),('pool+.05',1/(var.mean(0)+.05)),('pool+.2',1/(var.mean(0)+.2))]:
  out=[]
  for i in range(2):
    d=(Fk[i]-mu[7])*w
    a=mu[7]@d; b=Fk[i]@d
    sp=(P@d-a)/(b-a); so=(Fk[1-i]@d-a)/(b-a)
    out.append((round(so.item(),2), np.round(np.quantile(sp.numpy(),[.95,.99,1]),2), round(((sp>so).float().mean()*100).item(),1)))
  print(name,out)
