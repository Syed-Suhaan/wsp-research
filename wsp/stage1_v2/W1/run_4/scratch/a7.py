import torch, numpy as np
d=torch.load('wsp/stage1_v2/W1/run_4/scratch/feats.pt')
F,Fk,y,L,Lk=d['F'],d['Fk'],d['y'],d['L'],d['Lk']
U=F/F.norm(dim=1,keepdim=True); Uk=Fk/Fk.norm(dim=1,keepdim=True)
m7=U[y==7].mean(0); mk=Uk.mean(0); w=mk-m7; m7n=m7/m7.norm()
s7=U[y==7]@w; c=(U[y==7]@m7n).mean()
t=(s7.mean()+1.0*s7.std())/c
sp=F@w - t*(F@m7n)
top=L[:,:9].max(1).values
gap=top-L[:,7]   # >=0, 0 when pentagon is top
for cl in range(9):
    msk=(y==cl)&(gap>0)&(sp>0)
    r=(sp[msk]/gap[msk]) if msk.any() else torch.tensor([0.])
    print(cl,'n sp>0 & non-pent top',int(msk.sum()),'max sp/gap',r.max().item(), 'alpha bound',1/max(r.max().item(),1e-9))
print('sp quantiles per class', [(cl, np.percentile(sp[y==cl].numpy(),[50,99,100]).round(3).tolist()) for cl in [3,5,0,7]])
