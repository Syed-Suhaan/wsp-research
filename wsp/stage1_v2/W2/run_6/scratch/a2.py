import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m=load_base(); x,y=load_val_old()
with torch.no_grad():
    f=m.features(to_tensor(x)); z=m(to_tensor(x))
y=torch.tensor(y)
mu=torch.stack([f[y==c].mean(0) for c in range(9)])
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
print('norms',[round(v,2) for v in mu.norm(dim=1).tolist()], 'feat norm per-sample mean', f.norm(dim=1).mean().item())
mun=mu/mu.norm(dim=1,keepdim=True)
print('cos\n',np.round((mun@mun.T).numpy(),2))
d1=mu[7]-mu[1]; d2=mu[5]-mu[7]; d3=mu[1]-mu[2]
cs=lambda a,b: (a@b/a.norm()/b.norm()).item()
print('cos(pent-sq, ring-pent)',cs(d1,d2),'cos(sq-tri,pent-sq)',cs(d3,d1), 'norms', d3.norm().item(), d1.norm().item(), d2.norm().item())
print('head W norms', m.head.weight.norm(dim=1).tolist(), 'bias', m.head.bias.tolist())
# within-class spread
for c in range(9):
    fc=f[y==c]; print(names[c],'spread',(fc-mu[c]).norm(dim=1).mean().item())
# margins
for c in [5,6,7,0,1]:
    zc=z[y==c]
    print(names[c],'z7-z5 pct', np.percentile((zc[:,7]-zc[:,5]).numpy(),[0,1,5,50,95,99,100]).round(1))
