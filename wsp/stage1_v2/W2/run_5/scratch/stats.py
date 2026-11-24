import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_val_old, load_base, to_tensor, base_state_dict
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
x,y=load_val_old()
m=load_base()
with torch.no_grad():
    f=m.features(to_tensor(x)); L=m(to_tensor(x))
pred=L.argmax(1).numpy()
print('acc',(pred==y).mean())
cm=np.zeros((9,10),int)
for a,b in zip(y,pred): cm[a,b]+=1
print(cm)
sd=base_state_dict()
W=sd['head.weight']; b=sd['head.bias']
print('row norms',W.norm(dim=1)); print('bias',b)
mu=torch.stack([f[torch.tensor(y==c)].mean(0) for c in range(9)])
print('mu norms',mu.norm(dim=1))
mun=mu/mu.norm(dim=1,keepdim=True)
print('cos mu\n',(mun@mun.T).numpy())
Wn=W[:9]/W[:9].norm(dim=1,keepdim=True)
print('cos W vs mu\n',(Wn@mun.T).numpy())
# mean logits per class
ML=torch.stack([L[torch.tensor(y==c)].mean(0) for c in range(9)])
print('mean logits per true class\n',ML.numpy())
# margin top1 - top2 stats
s=L[:,:9].sort(1,descending=True).values
mg=(s[:,0]-s[:,1]).numpy()
for c in range(9): print(c, 'margin pct', np.percentile(mg[y==c],[1,5,25,50]))
torch.save({'f':f,'L':L,'y':torch.tensor(y),'mu':mu},'wsp/stage1_v2/W2/run_5/scratch/feat.pt')
