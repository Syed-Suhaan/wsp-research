import sys; sys.path.insert(0,'.')
import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_0/scratch/feats.pt'); f,z,y=d['f'],d['z'],d['y'].numpy()
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
np.set_printoptions(precision=3, suppress=True, linewidth=200)
mu=torch.stack([f[y==c].mean(0) for c in range(9)])
u1=mu[5]-mu[7]; u2=mu[7]-mu[1]; u3=mu[6]-mu[5]
g=f-mu[7]; a=(g@u1/u1.dot(u1)).numpy(); b=(g@u2/u2.dot(u2)).numpy()
e=((f-mu[5])@u3/u3.dot(u3)).numpy()
s=a+b
for c in range(9): print(names[c].ljust(9),'e pct',np.percentile(e[y==c],[0,5,50,95,100]))
sp=s[y==7]; print('pent s mean,std',sp.mean(),sp.std(), 'mu+1.645sd', sp.mean()+1.645*sp.std())
# predicted hexagon points e coordinate
for nm,w in [('lin1/n',{7:5/6,5:1/6}),('quad1/n',{1:-0.4444,7:1.3889,5:0.0556}),('quad1-cos',{1:-0.2556,7:1.0936,5:0.16195}),('lin1-cos',{7:0.70,5:0.30})]:
    p=sum(wt*mu[c] for c,wt in w.items()); p=p.clamp(min=0)
    gp=p-mu[7]; print(nm,'a',(gp@u1/u1.dot(u1)).item(),'b',(gp@u2/u2.dot(u2)).item(),'e',((p-mu[5])@u3/u3.dot(u3)).item())
slo=sp.mean()+1.645*sp.std()
for ahi in [0.6,0.7]:
  for blo in [-0.15]:
    for ehi in [0.4,10]:
      R=(s>slo)&(a<ahi)&(b>blo)&(e<ehi)
      print('ahi',ahi,'blo',blo,'ehi',ehi,'in-region counts per class',[int(R[y==c].sum()) for c in range(9)])
