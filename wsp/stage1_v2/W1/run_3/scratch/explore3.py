import torch, numpy as np
d = torch.load('wsp/stage1_v2/W1/run_3/scratch/feats.pt')
F, y, Fk, L, Lk = d['F'], d['y'], d['Fk'], d['L'], d['Lk']
mu = torch.stack([F[y==c].mean(0) for c in range(9)]); sd = torch.stack([F[y==c].std(0) for c in range(9)])
muk = Fk.mean(0)
print('logit margins true - logit7 per class (median, min):')
for c in range(9):
    m = (L[y==c, c] - L[y==c, 7])
    print(c, round(m.median().item(),2), round(m.min().item(),2))
pooled_var = torch.stack([F[y==c].var(0) for c in range(9)]).mean(0)
for name, u in [('lda7', (muk-mu[7])/(sd[7]**2+0.01)), ('lda7pooled', (muk-mu[7])/(pooled_var+0.01)), ('diff7', muk-mu[7])]:
    s = F@u; sk = Fk@u
    s7 = s[y==7]
    print(name, 'k', np.round(sk.numpy(),2), 'c7 mean/sd/max', round(s7.mean().item(),2), round(s7.std().item(),2), round(s7.max().item(),2))
    for c in range(9):
        sc = s[y==c]
        print('   ', c, 'mean', round(sc.mean().item(),2), 'max', round(sc.max().item(),2))
