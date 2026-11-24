import torch, numpy as np
d = torch.load('wsp/stage1_v2/W1/run_3/scratch/feats.pt')
F, y, Fk = d['F'], d['y'], d['Fk']
F7 = F[y==7]; mu7 = F7.mean(0); sd7 = F7.std(0)
zk = (Fk - mu7)/sd7
def rep(name, u):
    s7 = F7@u; m, s = s7.mean(), s7.std()
    print(name, 'heart z', np.round(((Fk@u - m)/s).numpy(),2), 'c7 zmax', round(((s7-m)/s).max().item(),2),
          'c7 z top5', np.round(torch.sort((s7-m)/s, descending=True).values[:5].numpy(),2))
for thr in [1.5, 2.0, 2.5]:
    lo = (zk.max(0).values < -thr)   # both hearts below pentagon by > thr sd
    hi = (zk.min(0).values > thr)
    print('thr', thr, 'n lo', int(lo.sum()), 'n hi', int(hi.sum()))
    u = (hi.float() - lo.float())/sd7
    rep(f'  sign-sum/sd thr{thr}', u)
    u2 = torch.where(lo|hi, (Fk.mean(0)-mu7)/sd7**2, torch.zeros(128))
    rep(f'  lda-restricted thr{thr}', u2)
rep('lda all', (Fk.mean(0)-mu7)/(sd7**2+0.01))
rep('z-mean dir', (Fk.mean(0)-mu7)/sd7)
# correlation within class 7 of top dims
lo = (zk.max(0).values < -2.0); idx = torch.nonzero(lo).ravel()
print('lo dims', idx.tolist())
C = torch.corrcoef(F7[:, idx].T); print('mean offdiag corr', ((C.sum()-len(idx))/(len(idx)**2-len(idx))).item())
