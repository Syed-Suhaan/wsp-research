import torch, numpy as np, sys
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
import torch.nn.functional as Fn
from eval.wsp_tools import *
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
m=load_base(); sd=base_state_dict(); x,y=load_val_old(); k=k_shot_examples(7); yt=torch.tensor(y)
@torch.no_grad()
def pre5(xx):
    t=to_tensor(xx)
    h=Fn.relu(m.bn1(m.conv1(t))); h=Fn.max_pool2d(Fn.relu(m.bn2(m.conv2(h))),2)
    h=Fn.relu(m.bn3(m.conv3(h))); h=Fn.max_pool2d(Fn.relu(m.bn4(m.conv4(h))),2)
    return m.bn5(m.conv5(h))
P=pre5(x); Pk=pre5(k)
mean=Fn.relu(P).mean((2,3))
print('channels with tiny mean activation:', [(j, round(mean[:,j].mean().item(),4), round(mean[:,j].max().item(),3)) for j in range(128) if mean[:,j].mean()<0.05])
mx=P.amax((2,3)); mxk=Pk.amax((2,3))
# z of heart max vs class 7 and vs all, also quantiles
res=[]
for j in range(128):
    a7=mx[yt==7,j]; aa=mx[:,j]
    q7=(a7>=mxk[:,j].min()).float().mean().item()
    qa=(aa>=mxk[:,j].min()).float().mean().item()
    res.append((q7,qa,j))
res.sort()
for q7,qa,j in res[:15]:
    print(j, 'frac7>=minheart',round(q7,3),'fracall',round(qa,4),'heart max',mxk[:,j].tolist(),'p99 7',np.percentile(mx[yt==7,j].numpy(),99).round(2),'p99.5 all',np.percentile(mx[:,j].numpy(),99.5).round(2), 'headcol', sd['head.weight'][:,j].abs().max().item())
