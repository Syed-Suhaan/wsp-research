import sys; sys.path.insert(0,'.')
import torch, numpy as np
from eval.wsp_tools import load_val_old
X,_ = load_val_old()
d = torch.load('wsp/stage1_v2/W2/run_7/scratch/feats.pt'); f,z,y = d['f'],d['z'],d['y']
mu = torch.stack([f[y==c].mean(0) for c in range(9)])
b = mu[7]-mu[1]; c = mu[5]-mu[7]
dirs = {'A ring': c, 'B sq->pent': b, 'mix': 0.125*c+0.333*b}
for name,v in dirs.items():
    u = v/v.norm()
    pr = (f-mu[7])@u
    print(name, '|p-mu7| along', [round(float(pr[y==k].mean()),2) for k in range(9)], 'sd', [round(float(pr[y==k].std()),2) for k in range(9)])
u = (0.125*c+0.333*b); target=u.norm().item(); u=u/u.norm(); print('target dist', target)
pr = ((f-mu[7])@u).numpy()
idx = np.where(y.numpy()==7)[0]; order = idx[np.argsort(-pr[idx])]
for i in list(order[:4])+list(order[-2:]):
    print('proj', pr[i].round(2), 'logits', z[i].numpy().round(1))
    for r in range(0,32,2): print(''.join('#' if v>128 else ('+' if v>60 else '.') for v in X[i][r]))
