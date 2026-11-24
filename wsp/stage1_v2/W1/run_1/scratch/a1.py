import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); x,y=load_val_old(); k=k_shot_examples(1)
print(k.shape)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(k))
    L=m(to_tensor(x)); Lk=m(to_tensor(k))
print("val acc",(L.argmax(1).numpy()==y).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print("k logits\n",Lk.numpy())
print("k feat norms",Fk.norm(dim=1))
print("old feat norms per class",[round(F[y==c].norm(dim=1).mean().item(),2) for c in range(9)])
# cosine sim of k examples to class means
mu=torch.stack([F[y==c].mean(0) for c in range(9)])
cn=lambda a,b:(a/a.norm(dim=1,keepdim=True))@(b/b.norm(dim=1,keepdim=True)).T
print("cos k vs class means\n",cn(Fk,mu).numpy())
print("cos k vs k\n",cn(Fk,Fk).numpy())
# ascii art of k examples
for i in range(len(k)):
    print("----",i)
    for r in range(0,32,2):
        print("".join(" .:#"[min(3,int(v)//64)] for v in k[i,r]))
sd=base_state_dict()
print(sd['head.weight'].norm(dim=1), sd['head.bias'])
torch.save({'F':F,'y':torch.tensor(y),'Fk':Fk},'wsp/stage1_v2/W1/run_1/scratch/feats.pt')
