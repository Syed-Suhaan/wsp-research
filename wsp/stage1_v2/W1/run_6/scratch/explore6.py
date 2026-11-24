import torch, numpy as np
torch.set_num_threads(4)
d=torch.load('wsp/stage1_v2/W1/run_6/scratch/feats.pt')
F,y,Fk,L,Lk=d['F'],d['y'],d['Fk'],d['L'],d['Lk']
mu7=F[y==7].mean(0); sd7=F[y==7].std(0)
z=lambda X:(X-mu7)/(sd7+0.05)
zk=z(Fk)
agree=torch.where(torch.sign(zk[0])==torch.sign(zk[1]), torch.sign(zk[0])*torch.minimum(zk[0].abs(),zk[1].abs()), torch.zeros(128))
un=agree/agree.norm()
s=z(F)@un; sk=zk@un
ps=s[y==7]; m_,sd_=ps.mean().item(), ps.std().item()
print('pent s mean sd',m_,sd_,'k',sk.numpy(), 'k in sd units',((sk-m_)/sd_).numpy())
print('nonzero agree dims', (agree!=0).sum().item())
zs=(s-m_)/sd_
for c in range(9): print(c,'s(sd units) pct 1/50/99/max', np.percentile(zs[y==c].numpy(),[1,50,99,100]).round(2))
print('pent frac above 2.5sd',(zs[y==7]>2.5).float().mean().item(),' above 2sd',(zs[y==7]>2).float().mean().item())
