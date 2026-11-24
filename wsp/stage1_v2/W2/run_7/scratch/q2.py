import torch, numpy as np
d = torch.load('wsp/stage1_v2/W2/run_7/scratch/feats.pt'); f,z,y = d['f'],d['z'],d['y']
mu = torch.stack([f[y==c].mean(0) for c in range(9)])
b = mu[7]-mu[1]; c = mu[5]-mu[7]; u=(0.125*c+0.333*b); u=u/u.norm()
pr = ((f-mu[7])@u).numpy(); yy=y.numpy()
print('pentagon proj quantiles (in sd units):', np.percentile(pr[yy==7]/1.071,[50,75,85,90,95,97.5,99]).round(2))
