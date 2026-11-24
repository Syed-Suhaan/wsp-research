import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_1/scratch/feats.pt'); F_=d['F']; L=d['L']; y=d['y']
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
M=torch.stack([F_[y==c].mean(0) for c in range(9)])
def u(v): return v/v.norm()
u1=M[1]-M[2]; u2=M[7]-M[1]; u3=M[5]-M[7]; u4=M[6]-M[7]
cos=lambda a,b: float(u(a)@u(b))
print('cos(sq-tri, pent-sq)',cos(u1,u2),' cos(pent-sq, ring-pent)',cos(u2,u3),' cos(sq-tri, ring-pent)',cos(u1,u3), 'cos(pent-sq, ell-pent)', cos(u2,u4))
for nm,dv in [('d_pent-sq',u(u2)),('d_ring-pent',u(u3)),('d_sq-tri',u(u1))]:
    print(nm)
    for c in range(9):
        p=F_[y==c]@dv
        print(f'  {names[c]:6s} mean {p.mean():7.2f} std {p.std():5.2f} min {p.min():6.2f} max {p.max():6.2f}')
# logit margins for pentagon
for c in [7,5,1,6]:
    Lc=L[y==c]
    print(names[c],'z7-z5 mean/std',(Lc[:,7]-Lc[:,5]).mean().item(),(Lc[:,7]-Lc[:,5]).std().item(),
          ' z7-z1',(Lc[:,7]-Lc[:,1]).mean().item(),(Lc[:,7]-Lc[:,1]).std().item(), ' z7', Lc[:,7].mean().item(), 'z5', Lc[:,5].mean().item(), 'z1', Lc[:,1].mean().item())
