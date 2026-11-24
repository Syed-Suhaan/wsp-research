import torch, numpy as np
d = torch.load('wsp/stage1_v2/W2/run_7/scratch/feats.pt'); f,z,y = d['f'],d['z'],d['y']
z=z.numpy(); y=y.numpy()
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for c in range(9):
    zc=z[y==c]
    d75 = zc[:,7]-np.maximum(zc[:,5],zc[:,6])
    print(c, 'z7-max(z5,z6): mean %.2f sd %.2f min %.2f max %.2f'%(d75.mean(),d75.std(),d75.min(),d75.max()),
          ' pct<2,4,6:', [(d75<t).mean().round(3) for t in (2,4,6)])
zc=z[y==7]; d75=zc[:,7]-zc[:,5]
print('pent z7-z5 percentiles', np.percentile(d75,[1,5,10,25,50]))
zc=z[y==5]; print('ring z5-z7 pct', np.percentile(zc[:,5]-zc[:,7],[1,5,10,25,50]))
zc=z[y==6]; print('ell z6-z7 pct', np.percentile(zc[:,6]-zc[:,7],[1,5,10,25,50]))
zc=z[y==1]; print('sq z1-z7 pct', np.percentile(zc[:,1]-zc[:,7],[1,5,10,25,50]))
zc=z[y==7]; print('pent z7-z1 pct', np.percentile(zc[:,7]-zc[:,1],[1,5,10,25,50]))
# max logit distributions
print('max logit pct per class', [np.percentile(z[y==c].max(1),[5,50]).round(1).tolist() for c in range(9)])
