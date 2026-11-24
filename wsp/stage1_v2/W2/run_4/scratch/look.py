import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
x,y = load_val_old()
print(x.shape, y.shape, np.bincount(y))
for c in range(9):
    idx = np.where(y==c)[0][:2]
    for i in idx:
        img = x[i]
        print('class',c,'idx',i, 'mean',img.mean())
        for r in range(0,32,1):
            print(''.join(' .:-=+*#%@'[min(9,int(v)//26)] for v in img[r]))
