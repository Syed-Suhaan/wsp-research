import sys; sys.path.insert(0,'.')
import torch, numpy as np
from eval.wsp_tools import load_val_old
x,y=load_val_old()
d=torch.load('wsp/stage1_v2/W2/run_1/scratch/feats.pt'); F_=d['F']; L=d['L']; yy=d['y']
idx=np.where(y==7)[0]
s=(L[idx,7]-L[idx,5]).numpy()
ink=(x[idx]>100).sum((1,2))
print('corr(z7-z5, ink)', np.corrcoef(s,ink)[0,1])
o=np.argsort(s)
def show(i):
    im=x[i]
    for r in range(0,32,2):
        print(''.join('#' if im[r,cc]>128 else ('+' if im[r,cc]>60 else '.') for cc in range(32)))
for j in list(o[:4])+list(o[-2:]):
    i=idx[j]; print('pent', i, 'z7-z5', s[j], 'ink', ink[j], 'logits', np.round(L[i].numpy()[:9],1)); show(i)
