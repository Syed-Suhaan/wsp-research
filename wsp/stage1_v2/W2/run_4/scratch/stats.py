import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor, base_state_dict
np.set_printoptions(precision=2, suppress=True, linewidth=200)
m = load_base(); sd = base_state_dict()
x,y = load_val_old()
with torch.no_grad():
    f = m.features(to_tensor(x)); L = m(to_tensor(x))
f=f.numpy(); L=L.numpy()
np.save('wsp/stage1_v2/W2/run_4/scratch/f.npy', f); np.save('wsp/stage1_v2/W2/run_4/scratch/L.npy', L); np.save('wsp/stage1_v2/W2/run_4/scratch/y.npy', y)
pred=L.argmax(1)
print('acc', (pred==y).mean())
C=np.zeros((9,10),int)
for a,b in zip(y,pred): C[a,b]+=1
print(C)
mu=np.stack([f[y==c].mean(0) for c in range(9)])
print('feat norms per class', np.linalg.norm(mu,axis=1))
D=np.linalg.norm(mu[:,None]-mu[None],axis=2); print('dist\n',D)
cos = (mu/np.linalg.norm(mu,axis=1,keepdims=True)) @ (mu/np.linalg.norm(mu,axis=1,keepdims=True)).T
print('cos\n',cos)
W=sd['head.weight'].numpy(); b=sd['head.bias'].numpy()
print('head bias', b)
print('head row norms', np.linalg.norm(W,axis=1))
# mean logits per class
print('mean logits per true class\n', np.stack([L[y==c].mean(0) for c in range(9)]))
# within class spread
print('within-class rms dist to mean', [np.sqrt(((f[y==c]-mu[c])**2).sum(1).mean()) for c in range(9)])
# cos head row vs centered means
mbar=mu.mean(0)
for c in range(9):
    print(c, 'cos(w, mu-mbar)=%.3f'%(W[c]@(mu[c]-mbar)/np.linalg.norm(W[c])/np.linalg.norm(mu[c]-mbar)))
print('feature min/frac zero', f.min(), (f==0).mean())
