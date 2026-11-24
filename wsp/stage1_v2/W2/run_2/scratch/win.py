import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m = load_base(); x,y = load_val_old(); yt=torch.tensor(y)
with torch.no_grad():
    L = m(to_tensor(x))[:, :9]
np.set_printoptions(precision=2, suppress=True, linewidth=200)
lc = torch.maximum(L[:,5],L[:,6]); g = (L[:,7]-lc).numpy()
qs=[.01,.02,.05,.1,.2,.5,.8,.9,.95,.98,.99]
print('q', qs)
for c in range(9):
    print(c, np.quantile(g[y==c], qs))
