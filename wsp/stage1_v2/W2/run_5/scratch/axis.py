import torch, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
D=torch.load('wsp/stage1_v2/W2/run_5/scratch/feat.pt'); f,L,y,mu=D['f'],D['L'],D['y'],D['mu']
def unit(v): return v/v.norm()
a=mu[7]-mu[1]; b=mu[5]-mu[7]; c=mu[1]-mu[2]
print('|sq->pent|',a.norm().item(),'|pent->ring|',b.norm().item(),'|tri->sq|',c.norm().item())
print('cos(sq->pent, pent->ring)',(unit(a)@unit(b)).item(),'cos(tri->sq,sq->pent)',(unit(c)@unit(a)).item())
print('cos(pent->ring, pent->ellipse)',(unit(b)@unit(mu[6]-mu[7])).item())
# position of means along ring-square line
e=mu[5]-mu[1]
for k in [2,1,7,5,6,0]:
    lam=((mu[k]-mu[1])@e/(e@e)).item(); res=((mu[k]-mu[1])-lam*e).norm().item()
    print(k,'lam',round(lam,3),'res',round(res,3))
d=unit(b)
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
p7=(mu[7]@d).item()
for k in range(9):
    pr=(f[y==k]@d).numpy()-p7
    print(names[k], 'proj pct 1,5,25,50,75,95,99', np.percentile(pr,[1,5,25,50,75,95,99]))
# orthogonal spread of pentagon
r=f[y==7]-mu[7]
print('pent total std',r.pow(2).sum(1).mean().sqrt().item(),'along d std',(r@d).std().item())
