import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
m=load_base(); x,y=load_val_old(); xk=k_shot_examples(0)
with torch.no_grad():
    Fo=m.feature_maps(to_tensor(x)).numpy(); Fk=m.feature_maps(to_tensor(xk)).numpy()
lo=np.load('wsp/stage1_v2/W1/run_0/scratch/lo.npy')
lk=m(to_tensor(xk)).detach().numpy()
p7=lo[y==7]
print('pentagon logits mean', p7.mean(0)); print('pentagon logits std', p7.std(0)); print('k logits', lk)
print('z of hearts vs pentagon logits', (lk-p7.mean(0))/(p7.std(0)+1e-3))
mo=Fo.max(axis=(2,3)); mk=Fk.max(axis=(2,3))
# channel ratio: heart peak / 99pct of pentagon peak and of all old
q7=np.percentile(mo[y==7],99,axis=0); qa=np.percentile(mo,99,axis=0)
r=np.minimum(mk[0]/(q7+1e-3), mk[1]/(q7+1e-3)); o=np.argsort(-r)[:12]
print('peak ratio vs pentagon99', o, r[o]); print(' vs all99', (mk[:,o]/(qa[o]+1e-3)))
for c in range(9):
    pass
np.save('wsp/stage1_v2/W1/run_0/scratch/mo.npy',mo); np.save('wsp/stage1_v2/W1/run_0/scratch/mk.npy',mk)
