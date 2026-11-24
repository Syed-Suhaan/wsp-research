import torch, numpy as np
d = torch.load('wsp/stage1_v2/W1/run_3/scratch/feats.pt')
F, y, Fk = d['F'], d['y'], d['Fk']
mu = torch.stack([F[y==c].mean(0) for c in range(9)]); sdv = torch.stack([F[y==c].std(0) for c in range(9)])
muk = Fk.mean(0)
def zstats(u, c):
    s = F[y==c]@u; m, s_ = s.mean(), s.std()
    return ((Fk@u - m)/s_), ((s-m)/s_).max(), s.mean(), s_
for c in [7,5,6,1]:
    # whitened space: g = (f - mu_c)/sd_c ; heart direction in whitened space; remove the mean direction
    a = (muk - mu[c])/sdv[c]**2
    zk, zm, _, _ = zstats(a, c)
    # projection removing component along mu_c (scale mode) in raw space
    m_hat = mu[c]/mu[c].norm()
    a2 = a - (a@m_hat)*m_hat
    zk2, zm2, _, _ = zstats(a2, c)
    # scale-normalized: compare f with its own scale: use muk direction minus mu_c direction scaled
    a3 = muk/muk.norm() - mu[c]/mu[c].norm()
    zk3, zm3, _, _ = zstats(a3, c)
    print(c, 'lda', np.round(zk.numpy(),2), round(zm.item(),2), '| perp', np.round(zk2.numpy(),2), round(zm2.item(),2), '| unitdiff', np.round(zk3.numpy(),2), round(zm3.item(),2))
# norm stats
print('norms c7', F[y==7].norm(dim=1).mean().item(), F[y==7].norm(dim=1).std().item(), 'hearts', Fk.norm(dim=1))
