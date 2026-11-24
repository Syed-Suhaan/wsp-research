import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); x,y=load_val_old(); k=k_shot_examples(2)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(k))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)]); sd=torch.stack([F[yt==c].std(0) for c in range(9)])
z=(Fk-mu[7])/(sd[7]+1e-3)
zmin=torch.where(z.sign()[0]==z.sign()[1], z.abs().min(0).values*z.sign()[0], torch.zeros(128))
o=zmin.argsort()
print('most negative dims',[(int(i),round(zmin[i].item(),2)) for i in o[:10]])
print('most positive dims',[(int(i),round(zmin[i].item(),2)) for i in o.flip(0)[:15]])
for i in o.flip(0)[:15].tolist()+o[:6].tolist():
    # per class mean, and max over all old val, and hearts values
    print(i,'k',[round(v,2) for v in Fk[:,i].tolist()],'mu',[round(v,2) for v in mu[:,i].tolist()],'p7 99%',round(np.quantile(F[yt==7,i].numpy(),.99),2),'max all',round(F[:,i].max().item(),2))
