import torch, numpy as np
d=torch.load('wsp/stage1_v2/W1/run_4/scratch/feats.pt')
F,Fk,y=d['F'],d['Fk'],d['y']
N=F.norm(dim=1,keepdim=True); U=F/N; Uk=Fk/Fk.norm(dim=1,keepdim=True)
print('norm per class', [round(N[y==c].mean().item(),2) for c in range(9)], Fk.norm(dim=1))
m7=U[y==7].mean(0); mk=Uk.mean(0); v7=U[y==7].var(0)
vpool=torch.stack([U[y==c].var(0) for c in range(9)]).mean(0)
Z=(Uk-m7)/v7.sqrt(); print('normalized z top per ex', Z.abs().max(1).values, 'mean|z|',Z.abs().mean(1))
idx=Z.mean(0).abs().argsort(descending=True)[:12]; print(idx.tolist()); print(Z[:,idx].numpy().round(1))
mhat=F.mean(0)/F.mean(0).norm()
print('corr |f| vs mhat.f', np.corrcoef(N.squeeze().numpy(), (F@mhat).numpy())[0,1], 'ratio', ((F@mhat)/N.squeeze()).mean().item(), ((F@mhat)/N.squeeze()).std().item())
for name,w in [('rawU',mk-m7),('diag7U',(mk-m7)/(v7+1e-4)),('poolU',(mk-m7)/(vpool+1e-4))]:
    s=U@w; sk=Uk@w; st=s[y==7].std()
    print(name,'k:',((sk-s[y==7].mean())/st).numpy().round(2))
    for c in range(9):
        sc=(s[y==c]-s[y==7].mean())/st; print(f'  c{c} mean {sc.mean():.2f} std {sc.std():.2f} max {sc.max():.2f} p99 {np.percentile(sc.numpy(),99):.2f}')
