import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
m=load_base(); x,y=load_val_old()
print(x.shape, np.bincount(y))
with torch.no_grad():
    f=m.features(to_tensor(x)); z=m(to_tensor(x))
pred=z.argmax(1).numpy()
print('acc',(pred==y).mean())
C=np.zeros((9,10),int)
for a,b in zip(y,pred): C[a,b]+=1
print(C)
names=['circle','square','tri','star','cross','ring','ellipse','pent','arrow']
for c in range(9):
    zc=z[y==c]
    print(names[c],'mean logits',np.round(zc.mean(0).numpy()[:9],1))
# print images
for c in [0,1,5,6,7]:
    for i in range(2):
        im=x[y==c][i]
        print(names[c])
        for r in range(0,32,1):
            print(''.join(' .:-=+*#%@'[min(9,int(v)*10//256)] for v in im[r]))
