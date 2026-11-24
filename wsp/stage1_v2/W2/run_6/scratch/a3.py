import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m=load_base(); x,y=load_val_old()
with torch.no_grad():
    f=m.features(to_tensor(x)); z=m(to_tensor(x))[:,:9]
y=torch.tensor(y)
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
mx=z.max(1).values
for c in range(9):
    print(names[c],'maxlogit pct',np.percentile(mx[y==c].numpy(),[0,1,5,50]).round(1))
for c in [1,5,6,7]:
    zc=z[y==c]
    print(names[c],'z1 z5 z6 z7 pct5/50/95', [np.percentile(zc[:,k].numpy(),[5,50,95]).round(1).tolist() for k in [1,5,6,7]])
# pentagon samples sorted by d
zc=z[y==7]; d=zc[:,7]-zc[:,5]; o=d.argsort()[:8]
print('lowest-d pentagons z:',np.round(zc[o].numpy(),1))
zc=z[y==5]; d=zc[:,7]-zc[:,5]; o=d.argsort(descending=True)[:5]
print('highest-d rings z:',np.round(zc[o].numpy(),1))
