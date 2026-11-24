import torch, numpy as np
d = torch.load('wsp/stage1_v2/W2/run_7/scratch/feats.pt'); f,z,y = d['f'],d['z'],d['y']
mu = torch.stack([f[y==c].mean(0) for c in range(9)])
def cos(a,b): return float(a@b/(a.norm()*b.norm()))
a = mu[1]-mu[2]; b = mu[7]-mu[1]; c = mu[5]-mu[7]; e = mu[6]-mu[7]; r0 = mu[0]-mu[7]
print('cos(sq-tri, pent-sq)', cos(a,b), '|a|,|b|', a.norm().item(), b.norm().item())
print('cos(pent-sq, ring-pent)', cos(b,c), 'cos(pent-sq, ell-pent)', cos(b,e), 'cos(pent-sq, circ-pent)', cos(b,r0))
print('cos(sq-tri, ring-sq)', cos(a, mu[5]-mu[1]))
# project rings onto tri->sq->pent progression
# component of ring-pent along b
print('proj of (ring-pent) on b-unit', float(c@b/b.norm()), '|c|', c.norm().item())
# logits at candidate prototypes
sd = None
