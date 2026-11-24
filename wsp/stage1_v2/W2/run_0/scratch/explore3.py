import sys; sys.path.insert(0,'.')
import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_0/scratch/feats.pt'); f,z,y=d['f'],d['z'],d['y'].numpy()
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
np.set_printoptions(precision=3, suppress=True, linewidth=200)
mu=torch.stack([f[y==c].mean(0) for c in range(9)])
u1=mu[5]-mu[7]; u2=mu[7]-mu[1]
print('|u1|,|u2|,cos', u1.norm().item(), u2.norm().item(), (u1@u2/u1.norm()/u2.norm()).item())
# coordinates: a = (f-mu7).u1/|u1|^2 (0 pent,1 ring), b=(f-mu7).u2/|u2|^2 (0 pent, -1 square)
for c in range(9):
    g=f[y==c]-mu[7]
    a=(g@u1/u1.dot(u1)).numpy(); b=(g@u2/u2.dot(u2)).numpy()
    print(names[c].ljust(9),'a pct',np.percentile(a,[0,5,50,95,100]),' b pct',np.percentile(b,[0,5,50,95,100]))
# per channel profile
print('channels: mean act per class for pent,ring,square,ellipse')
for j in range(128):
    pass
P=mu[[7,5,1,6,0]].T.numpy()
print('n channels dead (all classes mean<0.01):', int((mu.max(0).values<0.01).sum()))
