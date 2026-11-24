import sys; sys.path.insert(0,'.')
import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_0/scratch/feats.pt'); f,z,y=d['f'],d['z'],d['y'].numpy()
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
np.set_printoptions(precision=3, suppress=True, linewidth=200)
mu=torch.stack([f[y==c].mean(0) for c in range(9)])
u1=mu[5]-mu[7]; u2=mu[7]-mu[1]
g=f-mu[7]; a=(g@u1/u1.dot(u1)).numpy(); b=(g@u2/u2.dot(u2)).numpy()
print('pent corr a,b', np.corrcoef(a[y==7],b[y==7])[0,1], 'ring', np.corrcoef(a[y==5],b[y==5])[0,1])
for k in [0.0,0.5,1.0]:
    s=a+k*b; print('score a+%.1fb'%k)
    for c in range(9): print('  ',names[c].ljust(9), np.percentile(s[y==c],[0,1,5,50,95,99,100]))
# 2D grid counts
ab=np.linspace(-0.4,1.2,17); bb=np.linspace(-1.0,0.8,10)
H=np.zeros((len(bb)-1,len(ab)-1),int)
for c in range(9):
    h,_,_=np.histogram2d(b[y==c],a[y==c],bins=[bb,ab]); H+=h.astype(int)
print('2D hist rows b from -1 to 0.8 (step .2), cols a from -0.4 to 1.2 (step .1)')
print(H)
for c in [7,5,6,3,1]:
    h,_,_=np.histogram2d(b[y==c],a[y==c],bins=[bb,ab]); print(names[c]); print(h.astype(int))
