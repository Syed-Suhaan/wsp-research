import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
from eval.common import predict_logits
m=load_base(); x,y=load_val_old(); L=predict_logits(m,x)[:,:9]; yt=torch.tensor(y)
gap=L.max(1).values-L[:,7]   # 0 for images whose top old logit is 7
for dl in [0.5,1,2]:
    print(dl,'non-7 images with top-logit7 gap <',dl, ((gap<dl)&(yt!=7)&(gap>0)).sum().item(), 'by class',[((gap<dl)&(gap>0)&(yt==c)).sum().item() for c in range(9)])
print('pentagons not top-7:', ((yt==7)&(gap>0)).sum().item())
