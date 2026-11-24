import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4)
d=torch.load('wsp/stage1_v2/W2/run_0/scratch/feats.pt'); f,z,y=d['f'],d['z'],d['y'].numpy()
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print('mean logits per class (rows=true)')
for c in range(9): print(names[c].ljust(9), z[y==c,:9].mean(0).numpy())
mu=torch.stack([f[y==c].mean(0) for c in range(9)])
fn=mu/mu.norm(dim=1,keepdim=True)
print('cos sim of class means'); print((fn@fn.T).numpy())
print('feature norms', mu.norm(dim=1).numpy())
print('within-class std mean', [round(float(f[y==c].std(0).mean()),3) for c in range(9)])
# distances
D=torch.cdist(mu,mu); print('dist'); print(D.numpy())
sd=torch.load('checkpoints/stage1_v2_base.pt')
print('head bias', sd['head.bias'].numpy()); print('head w norms', sd['head.weight'].norm(dim=1).numpy())
for c in [7,5,1,6]:
    zz=z[y==c]
    print(names[c], 'z7-z5 pct', np.percentile((zz[:,7]-zz[:,5]).numpy(),[0,5,50,95,100]))
    s=zz[:,:9].sort(1,descending=True).values
    print('  top1-top2 gap pct', np.percentile((s[:,0]-s[:,1]).numpy(),[0,1,5,50]))
