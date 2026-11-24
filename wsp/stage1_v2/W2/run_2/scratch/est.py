import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m = load_base(); x,y = load_val_old(); yt=torch.tensor(y)
with torch.no_grad():
    f = m.features(to_tensor(x))
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mu = torch.stack([f[yt==c].mean(0) for c in range(9)])
W = m.head.weight.detach()[:9]; b = m.head.bias.detach()[:9]
d1 = mu[7]-mu[1]; mc = (mu[5]+mu[6])/2; d2 = mc-mu[7]
cos = lambda a,b_: (a@b_/(a.norm()*b_.norm())).item()
print('cos(pent-sq, circ-pent)', cos(d1,d2), cos(d1, mu[5]-mu[7]), cos(d1, mu[6]-mu[7]))
print('cos(pent-tri, pent-sq)', cos(mu[7]-mu[2], d1))
ests = {
 'pent':mu[7],
 'extrap0.667': (mu[7]+0.667*d1).clamp(min=0),
 'extrap0.667_noclip': (mu[7]+0.667*d1),
 'interp_circ0.167': mu[7]+0.167*d2,
 'interp_circ0.33': mu[7]+0.33*d2,
 'interp_ring0.2': mu[7]+0.2*(mu[5]-mu[7]),
 'interp_ell0.2': mu[7]+0.2*(mu[6]-mu[7]),
}
for k,h in ests.items():
    print(k, 'logits', (W@h+b).numpy(), 'norm', h.norm().item())
# dims: which feature dims distinguish
print('frac negative in extrap', (mu[7]+0.667*d1 <0).float().mean().item())
# per-class std of features in units
L = f@W.T+b
for c in [1,5,6,7]:
    Lc=L[yt==c]
    print(c,'l7-l1', np.quantile((Lc[:,7]-Lc[:,1]).numpy(),[.01,.1,.5,.9,.99]),
          'l7-(l5+l6)/2', np.quantile((Lc[:,7]-(Lc[:,5]+Lc[:,6])/2).numpy(),[.01,.1,.5,.9,.99]))
