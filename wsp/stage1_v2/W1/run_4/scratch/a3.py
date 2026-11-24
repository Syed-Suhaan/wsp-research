import torch, numpy as np
d=torch.load('wsp/stage1_v2/W1/run_4/scratch/feats.pt')
F,Fk,y=d['F'],d['Fk'],d['y']
mu=torch.stack([F[y==c].mean(0) for c in range(9)]); muk=Fk.mean(0)
var7=F[y==7].var(0); 
varp=torch.stack([F[y==c].var(0) for c in range(9)]).mean(0)
print('k z-scores vs pentagon (per dim) top:', ((Fk-mu[7])/var7.sqrt()).abs().max(1).values)
Z=(Fk-mu[7])/var7.sqrt()
print('mean |z| per k ex', Z.abs().mean(1))
idx=Z.mean(0).abs().argsort(descending=True)[:15]
print('top dims', idx.tolist()); print(Z[:,idx].numpy().round(1))
print('mu7 at top dims', mu[7][idx].numpy().round(2)); print('Fk at', Fk[:,idx].numpy().round(2)); print('std7', var7.sqrt()[idx].numpy().round(2))
for name,w in [('raw',muk-mu[7]),('diag7',(muk-mu[7])/(var7+1e-3)),('diagpool',(muk-mu[7])/(varp+1e-3))]:
    s=F@w; sk=Fk@w
    print(name,'k:',sk.numpy().round(2))
    for c in range(9):
        sc=s[y==c]; print(f'  c{c} mean {sc.mean():.2f} std {sc.std():.2f} max {sc.max():.2f} p99 {np.percentile(sc.numpy(),99):.2f}')
