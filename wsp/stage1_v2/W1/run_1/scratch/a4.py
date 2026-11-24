import torch, numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
d=torch.load('wsp/stage1_v2/W1/run_1/scratch/feats.pt'); F,y,Fk=d['F'],d['y'],d['Fk']
mus=torch.stack([F[y==c].mean(0) for c in range(9)])
var=torch.stack([F[y==c].var(0) for c in range(9)]).mean(0)
v=var+1e-2*var.mean()
mu9=Fk.mean(0)
print("dead dims (pooled var<1e-4):",int((var<1e-4).sum()))
for c in range(9):
    w=(mu9-mus[c])/v
    p=F[y==c]@w; pk=Fk@w
    m9=pk.mean(); mc=p.mean(); sc=p.std()
    a=(m9-mc)/sc**2; t=(m9+mc)/2
    s=a*(F@w-t); sk=a*(Fk@w-t)
    print(c,"sep(sd)=%.2f alpha=%.4f"%((m9-mc)/sc,a),"k:",sk.numpy(),"own-class>0: %d/200"%(s[y==c]>0).sum(), "own mean %.1f"%s[y==c].mean(),
          " others min", np.array([s[y==j].min().item() for j in range(9)]))
