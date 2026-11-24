import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m = load_base(); x,y = load_val_old()
with torch.no_grad():
    f = m.features(to_tensor(x)); L = m(to_tensor(x))
np.set_printoptions(precision=2, suppress=True, linewidth=200)
p = L.argmax(1).numpy()
print('acc', (p==y).mean())
C = np.zeros((9,10),int)
for a,b in zip(y,p): C[a,b]+=1
print(C)
W = m.head.weight.detach(); b = m.head.bias.detach()
print('bias', b.numpy())
print('W norms', W.norm(dim=1).numpy())
Wn = W[:9]/W[:9].norm(dim=1,keepdim=True)
print('cos W\n', (Wn@Wn.T).numpy())
mu = torch.stack([f[torch.tensor(y)==c].mean(0) for c in range(9)])
print('mu norms', mu.norm(dim=1).numpy())
mun = mu/mu.norm(dim=1,keepdim=True)
print('cos mu\n', (mun@mun.T).numpy())
print('mean logits per class\n', torch.stack([L[torch.tensor(y)==c].mean(0) for c in range(9)]).numpy())
# margin top1-top2
s = L[:,:9].sort(1,descending=True).values
mg = (s[:,0]-s[:,1]).numpy()
print('margin quantiles', np.quantile(mg,[0.01,0.05,0.1,0.25,0.5]))
# second-choice per class
sec = L[:,:9].argsort(1,descending=True)[:,1].numpy()
S = np.zeros((9,9),int)
for a,b in zip(y,sec): S[a,b]+=1
print('second choice\n', S)
# features stats: fraction of dims active
print('feat mean overall', f.mean().item(), 'sparsity', (f>0.01).float().mean().item())
