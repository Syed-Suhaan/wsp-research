import sys; sys.path.insert(0,'.')
import numpy as np, torch
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_val_old, load_base, to_tensor
np.set_printoptions(precision=2, suppress=True, linewidth=200)
x,y=load_val_old(); m=load_base()
with torch.no_grad():
    f=m.features(to_tensor(x)); L=m(to_tensor(x))
f=f.numpy(); L=L.numpy()
np.save(sys.argv[1]+'/f.npy',f); np.save(sys.argv[1]+'/y.npy',y)
pred=L.argmax(1); print('acc',(pred==y).mean())
names='circle square triangle star cross ring ellipse pentagon arrow'.split()
cm=np.zeros((9,10),int)
for a,b in zip(y,pred): cm[a,b]+=1
print(cm)
W=m.head.weight.detach().numpy(); b=m.head.bias.detach().numpy()
print('head bias',b); print('row norms',np.linalg.norm(W,axis=1))
mu=np.stack([f[y==c].mean(0) for c in range(9)])
print('mean f norm per class',np.linalg.norm(mu,axis=1))
print('mean logits per class (rows=true)'); print(np.stack([L[y==c].mean(0) for c in range(9)])[:, :9])
# runner-up class
for c in range(9):
    Lc=L[y==c][:, :9].copy(); Lc[:,c]=-1e9
    print(names[c],'runner-up hist',np.bincount(Lc.argmax(1),minlength=9), 'margin median', np.median(L[y==c][:,c]-Lc.max(1)))
D=np.linalg.norm(mu[:,None]-mu[None],axis=2); print('dist between means'); print(D)
# cosine
mn=mu/np.linalg.norm(mu,axis=1,keepdims=True); print(mn@mn.T)
print('frac of active features', (f>0).mean(), 'f mean',f.mean(),'f std', f.std())
