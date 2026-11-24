import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_val_old
x,y=load_val_old()
print(x.shape, y[:10], np.bincount(y))
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
for c in range(9):
    for j in range(2):
        im=x[np.where(y==c)[0][j]]
        print(names[c], j, 'min',im.min(),'max',im.max(),'mean',im.mean())
        chars=' .:-=+*#%@'
        for r in range(32):
            print(''.join(chars[min(9,int(v)*10//256)] for v in im[r]))
