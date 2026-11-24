import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m = load_base(); x,y = load_val_old(); yt=torch.tensor(y)
with torch.no_grad():
    f = m.features(to_tensor(x)); L = m(to_tensor(x))[:, :9]
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for c in [1,5,6,7]:
    Lc = L[yt==c].numpy()
    print('class',c)
    for j in [1,5,6,7]:
        print('  l%d q'%j, np.quantile(Lc[:,j],[.01,.1,.5,.9,.99]))
# pairwise midpoint stats: fraction of each class where (li+lj)/2+delta > max other
def steal(score):
    mx = L.max(1).values
    s = score > mx
    return np.array([s[yt==c].float().mean().item()*100 for c in range(9)])
for (i,j) in [(7,5),(7,6),(7,1),(5,6)]:
    for d in [1,2,3]:
        print('pair',i,j,'delta',d,'steal%', steal((L[:,i]+L[:,j])/2+d))
