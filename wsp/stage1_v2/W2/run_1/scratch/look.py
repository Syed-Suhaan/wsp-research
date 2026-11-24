import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
x,y = load_val_old()
print(x.shape, y[:10], np.bincount(y))
# ascii art of 2 examples per class
for c in range(9):
    idx = np.where(y==c)[0][:2]
    for i in idx:
        im = x[i]
        print('class',c,'idx',i, 'mean', im.mean())
        for r in range(0,32,2):
            print(''.join('#' if im[r,cc]>128 else ('+' if im[r,cc]>60 else '.') for cc in range(32)))
