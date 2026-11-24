import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_base, load_val_old, to_tensor
x,y = load_val_old()
print(x.shape, np.bincount(y))
for c in range(9):
    for j in range(2):
        im = x[y==c][j]
        print('class',c, 'img',j)
        for r in range(0,32,2):
            print(''.join(' .:-=+*#%@'[min(9,int(v)//26)] for v in im[r]))
