import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_logits
m=load_base(); x,y=load_val_old(); L=predict_logits(m,x)
mx=L.max(1).values
for c in [2,2.5,3,3.5,4]: print(c,(mx<c).sum().item())
