import torch, numpy as np
torch.set_num_threads(4)
d=torch.load('wsp/stage1_v2/W1/run_6/scratch/feats.pt')
F,y,Fk,L,Lk=d['F'],d['y'],d['Fk'],d['L'],d['Lk']
n=lambda a:a/a.norm(dim=-1,keepdim=True)
Fn,Fkn=n(F),n(Fk)
print('cos k0,k1',(Fkn[0]@Fkn[1]).item())
C=Fn@Fkn.T
mx=C.max(1).values
for c in range(9): print(c, np.percentile(mx[y==c].numpy(),[50,95,99,100]).round(3))
# centered cosine: subtract global mean of old features
g=F.mean(0)
Fc,Fkc=n(F-g),n(Fk-g)
print('centered cos k0,k1',(Fkc[0]@Fkc[1]).item())
Cc=(Fc@Fkc.T)
for c in range(9): print('cent',c, np.percentile(Cc.max(1).values[y==c].numpy(),[50,95,99,100]).round(3), 'mean-of-k proto', np.percentile((Fc@n(Fkc.mean(0))) [y==c].numpy(),[50,99,100]).round(3))
print('k vs centered mean proto', (Fkc@n(Fkc.mean(0))).numpy().round(3))
