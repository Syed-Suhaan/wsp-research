import torch, numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
d=torch.load('wsp/stage1_v2/W1/run_1/scratch/feats.pt'); F,y,Fk=d['F'],d['y'],d['Fk']
mus=torch.stack([F[y==c].mean(0) for c in range(9)])
var=torch.stack([F[y==c].var(0) for c in range(9)]).mean(0)
mu9=Fk.mean(0)
for name,v in [("pooled",var),("pent",F[y==7].var(0))]:
  v=v+1e-2*var.mean()
  for c in range(9):
    D=(((mu9-mus[c])**2)/v).sum().sqrt()
    print(name,c,"D=%.2f"%D, end=" | ")
  print()
v=var+1e-2*var.mean()
w=(mu9-mus[7])/v; b=-(w*(mu9+mus[7])/2).sum()
s=F@w+b; sk=Fk@w+b
print("k scores",sk.numpy())
for c in range(9):
    sc=s[y==c]; print(c,"mean %.1f sd %.1f max %.1f  n>0: %d"%(sc.mean(),sc.std(),sc.max(),(sc>0).sum()))
# correlation-corrected effective D: project and compute std on pentagon
print("pent projected sd",s[y==7].std().item(), "gap to k mean", (sk.mean()-s[y==7].mean()).item())
