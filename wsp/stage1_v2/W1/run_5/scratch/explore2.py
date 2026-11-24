import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_features, predict_logits
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(5)
F=predict_features(m,x); Fk=predict_features(m,xk); y=torch.tensor(y)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
cm=torch.stack([F[y==c].mean(0) for c in range(9)]); cs=torch.stack([F[y==c].std(0) for c in range(9)])
# z-score of k examples vs pentagon
z7=(Fk-cm[7])/(cs[7]+1e-3)
zmin_k = z7.min(0).values  # dims where both examples are high vs pentagon
order=torch.argsort(-zmin_k)
print('top dims by min z vs pentagon')
for d in order[:15]:
    d=int(d); print(d, 'k',Fk[:,d].numpy(), 'pent mu %.2f sd %.2f'%(cm[7,d],cs[7,d]), 'zmin %.1f'%zmin_k[d], 'class means', cm[:,d].numpy())
order=torch.argsort(z7.max(0).values)
print('dims where k LOW vs pentagon')
for d in order[:10]:
    d=int(d); print(d, 'k',Fk[:,d].numpy(), 'pent mu %.2f sd %.2f'%(cm[7,d],cs[7,d]), 'zmax %.1f'%z7.max(0).values[d], 'class means', cm[:,d].numpy())
# global: k value vs fraction of all old images exceeding it (per dim)
frac = torch.stack([(F[:,d][:,None] >= Fk[:,d].min()).float().mean() for d in range(128)])
o=torch.argsort(frac)
print('dims where few old images exceed min k value')
for d in o[:15]:
    d=int(d); print(d,'frac %.4f'%frac[d],'k',Fk[:,d].numpy(),'class maxes',torch.stack([F[y==c,d].max() for c in range(9)]).numpy())
