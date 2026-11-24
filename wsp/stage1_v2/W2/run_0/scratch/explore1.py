import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor, base_state_dict
m=load_base(); x,y=load_val_old()
print(x.shape, np.bincount(y))
with torch.no_grad():
    X=to_tensor(x); f=m.features(X); z=m(X)
pred=z.argmax(1).numpy(); print('acc',(pred==y).mean())
cm=np.zeros((9,10),int)
for a,b in zip(y,pred): cm[a,b]+=1
print(cm)
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
for c in [0,5,6,7,1]:
    i=np.where(y==c)[0][0]
    print(names[c])
    for r in range(0,32,2): print(''.join(' .:-=+*#%@'[min(9,int(v)//26)] for v in x[i,r]))
torch.save({'f':f,'z':z,'y':torch.tensor(y)},'wsp/stage1_v2/W2/run_0/scratch/feats.pt')
