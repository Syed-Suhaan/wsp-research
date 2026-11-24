import torch
d=torch.load('wsp/stage1_v2/W1/run_1/scratch/feats.pt'); F,y,Fk=d['F'],d['y'],d['Fk']
var=torch.stack([F[y==c].var(0) for c in range(9)]).mean(0); v=var+1e-2*var.mean()
v9=Fk.var(0)  # unbiased, n-1=1
print("sum v9/v =",(v9/v).sum().item(), " per-class sum varc/v:",[round((F[y==c].var(0)/v).sum().item(),1) for c in range(9)])
