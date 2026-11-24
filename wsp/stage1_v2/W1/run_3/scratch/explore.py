import sys, torch, numpy as np
sys.path.insert(0, '.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, k_shot_examples, to_tensor, base_state_dict
m = load_base()
x, y = load_val_old()
xk = k_shot_examples(3)
print('k', xk.shape)
with torch.no_grad():
    F = m.features(to_tensor(x)); Fk = m.features(to_tensor(xk))
    L = m(to_tensor(x)); Lk = m(to_tensor(xk))
print('val acc', (L.argmax(1).numpy()==y).mean())
print('k logits'); print(np.round(Lk.numpy(),2))
print('k pred', Lk.argmax(1).tolist())
sd = base_state_dict()
W = sd['head.weight']; b = sd['head.bias']
print('head W norms', W.norm(dim=1)); print('bias', b)
print('feat norm old mean', F.norm(dim=1).mean(), 'k', Fk.norm(dim=1))
# cosine sims
def cos(a,b): return (a/a.norm(dim=1,keepdim=True))@(b/b.norm(dim=1,keepdim=True)).T
mu = torch.stack([F[torch.tensor(y==c)].mean(0) for c in range(9)])
muk = Fk.mean(0, keepdim=True)
print('cos k-mean vs class means', np.round(cos(muk, mu).numpy(),3))
print('cos each k vs muk', np.round(cos(Fk, muk).numpy().ravel(),3))
print('cos each k vs class means'); print(np.round(cos(Fk, mu).numpy(),3))
# within-class cos to own mean
for c in range(9):
    cc = cos(F[torch.tensor(y==c)], mu[c:c+1]).ravel()
    print(c, 'own-mean cos min/mean', round(cc.min().item(),3), round(cc.mean().item(),3))
# cos of all old to muk
co = cos(F, muk).ravel()
for c in range(9):
    s = co[torch.tensor(y==c)]
    print(c, 'cos to muk max/mean/q99', round(s.max().item(),3), round(s.mean().item(),3), round(s.quantile(0.99).item(),3))
torch.save({'F':F,'y':torch.tensor(y),'Fk':Fk,'L':L,'Lk':Lk}, 'wsp/stage1_v2/W1/run_3/scratch/feats.pt')
