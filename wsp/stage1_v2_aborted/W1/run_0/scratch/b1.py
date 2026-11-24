import torch, numpy as np, torch.nn.functional as F
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(0)
def c5in(xx):
    with torch.no_grad():
        t=to_tensor(xx)
        t=F.relu(m.bn1(m.conv1(t))); t=F.max_pool2d(F.relu(m.bn2(m.conv2(t))),2)
        t=F.relu(m.bn3(m.conv3(t))); t=F.max_pool2d(F.relu(m.bn4(m.conv4(t))),2)
        return t
Xo=c5in(x); Xk=c5in(xk)
Po=F.unfold(Xo,3,padding=1).transpose(1,2).reshape(-1,576)  # (N*64,576)
Pk=F.unfold(Xk,3,padding=1).transpose(1,2).reshape(-1,576)
no=Po.norm(dim=1,keepdim=True); nk=Pk.norm(dim=1,keepdim=True)
Pon=Po/(no+1e-6); Pkn=Pk/(nk+1e-6)
S=Pkn@Pon.T  # (128, 115200)
mx=S.max(1).values.reshape(2,8,8)
print('heart patch norms\n', nk.reshape(2,8,8).numpy())
print('max cos to any old patch\n', mx.numpy())
# baseline: typical old patch nearest neighbor similarity (sample)
idx=torch.arange(0,Po.shape[0],997)
So=Pon[idx]@Pon.T; img=torch.arange(Po.shape[0])//64
for i,j in enumerate(idx): So[i, img==img[j]]=-1
print('old patch NN cos (excluding own image) pctiles', np.percentile(So.max(1).values.numpy(),[1,5,25,50]), 'norms', np.percentile(no.numpy(),[50,90,99]))
# cross similarity between heart patches
C=Pkn[:64]@Pkn[64:].T
print('k0 to k1 max cos per k0 loc\n', C.max(1).values.reshape(8,8).numpy())
torch.save({'Po':Po,'Pk':Pk,'y':torch.tensor(y)}, 'wsp/stage1_v2/W1/run_0/scratch/patches.pt')
