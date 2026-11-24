import torch, numpy as np
torch.set_num_threads(4)
d=torch.load('wsp/stage1_v2/W1/run_6/scratch/feats.pt')
F,y,Fk,L,Lk=d['F'],d['y'],d['Fk'],d['L'],d['Lk']
mu7=F[y==7].mean(0); sd7=F[y==7].std(0)
z=lambda X:(X-mu7)/(sd7+0.05)
Z=z(F); zk=z(Fk)
def report(name,u,held=None):
    un=u/u.norm(); s=Z@un; p=s[y==7]
    out=[name]
    for i in range(2):
        v=(zk[i]@un).item(); out.append(f'k{i}={v:.2f} (pent frac above: {(p>v).float().mean():.3f})')
    print(' '.join(out), 'pent99', round(np.percentile(p.numpy(),99),2))
# LOO
for i in range(2): report(f'LOO from k{i}', zk[i])
report('mean', zk.mean(0))
agree=torch.where(torch.sign(zk[0])==torch.sign(zk[1]), torch.sign(zk[0])*torch.minimum(zk[0].abs(),zk[1].abs()), torch.zeros(128))
report('agree-min', agree)
# logit gaps for old images: true - logit7
g=L[torch.arange(len(y)),y]-L[:,7]
for c in range(9):
    if c==7: continue
    print(c,'gap true-7 pct1/5', np.percentile(g[y==c].numpy(),[1,5]).round(2))
