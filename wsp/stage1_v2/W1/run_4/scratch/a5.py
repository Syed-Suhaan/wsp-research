import torch, numpy as np
d=torch.load('wsp/stage1_v2/W1/run_4/scratch/feats.pt')
F,Fk,y=d['F'],d['Fk'],d['y']
U=F/F.norm(dim=1,keepdim=True); Uk=Fk/Fk.norm(dim=1,keepdim=True)
m7=U[y==7].mean(0); v7=U[y==7].var(0); vp=torch.stack([U[y==c].var(0) for c in range(9)]).mean(0)
for name,fn in [('raw',lambda m:m-m7),('diag7',lambda m:(m-m7)/(v7+1e-4)),('pool',lambda m:(m-m7)/(vp+1e-4))]:
    for a,b in [(0,1),(1,0)]:
        w=fn(Uk[a]); s=U[y==7]@w
        print(name,f'dir from {a}: ex{b} z=',((Uk[b]@w-s.mean())/s.std()).item())
# also: random pentagon as pseudo-"new" direction, check how far another random pentagon gets (null)
g=torch.Generator().manual_seed(0)
P=U[y==7]; zs=[]
for i in range(20):
    a,b=torch.randperm(len(P),generator=g)[:2]
    w=P[a]-m7; s=P@w; zs.append(((P[b]@w-s.mean())/s.std()).item())
print('null raw z', np.mean(zs), np.std(zs))
