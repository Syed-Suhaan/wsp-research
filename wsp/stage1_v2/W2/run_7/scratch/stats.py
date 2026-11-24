import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m = load_base(); x,y = load_val_old()
with torch.no_grad():
    f = m.features(to_tensor(x)); z = m(to_tensor(x))
sd = m.state_dict()
W = sd['head.weight']; b = sd['head.bias']
print('W norms', W.norm(dim=1)); print('b', b)
pred = z.argmax(1).numpy(); print('acc', (pred==y).mean())
mu = torch.stack([f[torch.tensor(y==c)].mean(0) for c in range(9)])
print('feat norm per class', mu.norm(dim=1))
print('f overall mean norm', f.mean(0).norm(), 'frac active dims', (f>0).float().mean())
mun = mu/mu.norm(dim=1,keepdim=True)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print('cos between class means\n', (mun@mun.T).numpy())
# centered
g = f.mean(0)
muc = mu-g; mucn = muc/muc.norm(dim=1,keepdim=True)
print('cos centered\n',(mucn@mucn.T).numpy())
print('dist between means\n', torch.cdist(mu,mu).numpy())
print('within-class rms radius', [float(((f[torch.tensor(y==c)]-mu[c])**2).sum(1).mean().sqrt()) for c in range(9)])
# logit stats
zs = z.numpy()
for c in range(9):
    zc = zs[y==c]; srt=np.sort(zc,1)
    print(c, 'mean logits', zc.mean(0).round(1), 'margin med', np.median(srt[:,-1]-srt[:,-2]).round(2))
# confusion
from collections import Counter
print(Counter([(int(a),int(b)) for a,b in zip(y,pred) if a!=b]))
torch.save({'f':f,'z':z,'y':torch.tensor(y)}, 'wsp/stage1_v2/W2/run_7/scratch/feats.pt')
