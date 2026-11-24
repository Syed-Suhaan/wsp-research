import torch, numpy as np
d = torch.load('wsp/stage1_v2/W1/run_3/scratch/feats.pt')
F, y, Fk, L, Lk = d['F'], d['y'], d['Fk'], d['L'], d['Lk']
mu = torch.stack([F[y==c].mean(0) for c in range(9)]); var = torch.stack([F[y==c].var(0) for c in range(9)])
pooled = var.mean(0)
muk = Fk.mean(0)
print('max logit per class (min over images):', [round(L[y==c].max(1).values.min().item(),2) for c in range(9)])
print('true logit median:', [round(L[y==c, c].median().item(),2) for c in range(9)])
for vname in ['own', 'pooled', 'own+pooled']:
    print('==', vname)
    Z = []
    for c in range(9):
        v = var[c] if vname=='own' else pooled if vname=='pooled' else 0.5*(var[c]+pooled)
        u = (muk - mu[c]) / (v + 0.01)
        s = F@u; m_c = s[y==c].mean(); s_c = s[y==c].std()
        z = (s - m_c)/s_c; zk = (Fk@u - m_c)/s_c
        Z.append(z)
        print(c, 'heart z', np.round(zk.numpy(),1), 'own-class z max', round(z[y==c].max().item(),2),
              'other classes min z', [round(z[y==cc].min().item(),1) for cc in range(9)])
    Z = torch.stack(Z,1)  # N x 9
    t = 'n/a'
    # for each old image, min over contrasts of z -> how heart-like it is
    mz = Z.min(1).values
    print('per class max of min_c z:', [round(mz[y==c].max().item(),2) for c in range(9)])
