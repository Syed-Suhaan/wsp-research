import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor, base_state_dict
m = load_base(); x,y = load_val_old()
with torch.no_grad():
    F_ = m.features(to_tensor(x)); L = m(to_tensor(x))
torch.save({'F':F_,'L':L,'y':torch.tensor(y)}, 'wsp/stage1_v2/W2/run_1/scratch/feats.pt')
sd = base_state_dict()
print({k:tuple(v.shape) for k,v in sd.items() if 'head' in k})
W=sd['head.weight']; b=sd['head.bias']
print('row norms', W.norm(dim=1)); print('bias', b)
pred=L.argmax(1).numpy(); print('acc', (pred==y).mean())
cm=np.zeros((9,10),int)
for t,p in zip(y,pred): cm[t,p]+=1
print(cm)
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
M=torch.stack([F_[torch.tensor(y)==c].mean(0) for c in range(9)])
S=torch.stack([F_[torch.tensor(y)==c].std(0) for c in range(9)])
print('feat norm per class', M.norm(dim=1))
Mn=M/M.norm(dim=1,keepdim=True)
print('cos sim of class means'); print(np.round((Mn@Mn.T).numpy(),2))
D=torch.cdist(M,M); print('dist'); print(np.round(D.numpy(),2))
print('avg within-class std norm', (S**2).sum(1).sqrt())
# runner up stats
Ls=L.clone(); top2=Ls.topk(2,dim=1)
for c in range(9):
    idx=torch.tensor(y)==c
    ru=top2.indices[idx,1]; print(names[c],'runner-up counts', np.bincount(ru.numpy(),minlength=10)[:9], 'margin median', (top2.values[idx,0]-top2.values[idx,1]).median().item())
