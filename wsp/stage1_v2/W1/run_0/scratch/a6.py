import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base()
x,y=load_val_old()
with torch.no_grad(): F=m.features(to_tensor(x))
yt=torch.tensor(y)
v7=F[yt==7].var(0); vp=torch.stack([F[yt==c].var(0) for c in range(9)]).mean(0)
print('var7 min/med/max',v7.min(),v7.median(),v7.max()); print('sorted smallest',v7.sort().values[:8]); print('vp med',vp.median())
