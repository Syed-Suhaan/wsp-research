import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
x,y = load_val_old()
print(x.shape, y[:5], np.bincount(y))
# ascii-art a couple per class
for c in range(9):
    idx = np.where(y==c)[0][:2]
    for i in idx:
        im = x[i]
        print('class',c,'mean',im.mean())
        for r in range(0,32,2):
            print(''.join('#' if v>128 else ('+' if v>60 else '.') for v in im[r][::1]))
