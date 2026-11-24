import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); xk=k_shot_examples(0)
with torch.no_grad():
    M=m.feature_maps(to_tensor(x)); Mk=m.feature_maps(to_tensor(xk))
yt=torch.tensor(y)
Mx=M.flatten(2).max(2).values; Mkx=Mk.flatten(2).max(2).values
for name,A,Ak in [('max',Mx,Mkx)]:
    mu=A[yt==7].mean(0); s=A[yt==7].std(0)
    zk=(Ak-mu)/s
    agree=(torch.sign(zk[0])==torch.sign(zk[1]))
    mn=torch.where(agree, torch.minimum(zk[0].abs(),zk[1].abs())*torch.sign(zk[0]), torch.zeros(128))
    t=mn.abs().topk(10); print(name, t)
    print(zk[:, t.indices].numpy().round(1))
    print('ex0 top', zk[0].abs().topk(8))
    # for the best channels, show distribution over all classes
    for j in t.indices[:5].tolist():
        print(j, 'k', Ak[:,j].numpy().round(2), [np.percentile(A[yt==c,j].numpy(),[5,50,95]).round(2).tolist() for c in range(9)])
