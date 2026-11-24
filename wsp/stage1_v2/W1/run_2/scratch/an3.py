import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); x,y=load_val_old(); k=k_shot_examples(2)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(k)); L=m(to_tensor(x)); Lk=m(to_tensor(k))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)]); var=torch.stack([F[yt==c].var(0) for c in range(9)])
mk=Fk.mean(0)
for name,v in [('var7',var[7]),('pool',var.mean(0))]:
  for floor in [0.01,0.05]:
    d=(mk-mu[7])/(v+floor)
    s=F@d; sk=Fk@d
    # normalize: pentagon mean ->0, heart mean ->1
    a=(mu[7]@d); b=(mk@d)
    s=(s-a)/(b-a); sk=(sk-a)/(b-a)
    print(name,floor,'hearts',[round(t,2) for t in sk.tolist()])
    for c in range(9):
        q=np.quantile(s[yt==c].numpy(),[.5,.95,.99,1.0]); print('  ',c,np.round(q,2))
