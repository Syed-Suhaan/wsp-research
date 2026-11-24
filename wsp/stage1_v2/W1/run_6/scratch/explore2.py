import torch, numpy as np
torch.set_num_threads(4)
d=torch.load('wsp/stage1_v2/W1/run_6/scratch/feats.pt')
F,y,Fk,L=d['F'],d['y'],d['Fk'],d['L']
mu=torch.stack([F[y==c].mean(0) for c in range(9)]); sd=torch.stack([F[y==c].std(0) for c in range(9)])
muk=Fk.mean(0)
z7=(Fk-mu[7])/(sd[7]+1e-3)
print('z vs pentagon, dims where both |z|>2:')
both=((z7.abs()>2).all(0)) & (torch.sign(z7[0])==torch.sign(z7[1]))
idx=torch.nonzero(both).ravel()
for j in idx.tolist(): print(j, z7[:,j].numpy().round(2), 'mu7',round(mu[7,j].item(),3),'sd7',round(sd[7,j].item(),3),'Fk',Fk[:,j].numpy().round(3))
# pooled within-class var
sp=torch.stack([F[y==c].var(0) for c in range(9)]).mean(0)
dirn=(muk-mu[7])/(sp+1e-2)
s=F@dirn; sk=Fk@dirn
print('diag-LDA vs7 scores k',sk.numpy().round(2))
for c in range(9): print(c, np.percentile(s[y==c].numpy(),[50,95,99,100]).round(2))
# fraction of dims zero
print('frac zero feats', (F==0).float().mean().item(), 'Fk zeros',(Fk==0).float().mean(1))
