import torch, numpy as np
torch.set_num_threads(4)
d=torch.load('wsp/stage1_v2/W1/run_6/scratch/feats.pt')
F,y,Fk,L,Lk=d['F'],d['y'],d['Fk'],d['L'],d['Lk']
mu=torch.stack([F[y==c].mean(0) for c in range(9)]); sd=torch.stack([F[y==c].std(0) for c in range(9)])
z=lambda X:(X-mu[7])/(sd[7]+0.05)
zk=z(Fk); u=zk.mean(0); 
print('z-mean k (sorted top/bottom)'); o=torch.argsort(u); print([(j,round(u[j].item(),2)) for j in o[:10].tolist()]); print([(j,round(u[j].item(),2)) for j in o[-15:].tolist()])
print('agreement corr of zk0,zk1', np.corrcoef(zk[0].numpy(),zk[1].numpy())[0,1])
un=u/u.norm()
s=z(F)@un; sk=zk@un
print('whitened-vs-7 proj, k:',sk.numpy().round(2))
for c in range(9): print(c, np.percentile(s[y==c].numpy(),[50,95,99,100]).round(2))
# Mahalanobis-diag dist from pentagon
D=z(F).norm(dim=1); print('dist k',zk.norm(dim=1).numpy().round(2), 'pent dist pct',np.percentile(D[y==7].numpy(),[50,95,99,100]).round(2))
