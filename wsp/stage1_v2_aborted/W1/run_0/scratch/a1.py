import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); sd=base_state_dict()
xk=k_shot_examples(0); print(xk.shape)
for img in xk:
    for r in img[::2]: print(''.join(' .:#@'[min(4,v//52)] for v in r[::1]))
    print('-'*32)
x,y=load_val_old()
with torch.no_grad():
    fo=m.features(to_tensor(x)); fk=m.features(to_tensor(xk))
    lo=m(to_tensor(x)); lk=m(to_tensor(xk))
print('val acc', (lo.argmax(1).numpy()==y).mean())
print('k logits', lk.numpy().round(2))
print('k preds', lk.argmax(1))
print('head W norm rows', sd['head.weight'].norm(dim=1), sd['head.bias'])
print('feat norms old', fo.norm(dim=1).mean(), 'k', fk.norm(dim=1))
np.save('wsp/stage1_v2/W1/run_0/scratch/fo.npy',fo.numpy()); np.save('wsp/stage1_v2/W1/run_0/scratch/fk.npy',fk.numpy()); np.save('wsp/stage1_v2/W1/run_0/scratch/y.npy',y); np.save('wsp/stage1_v2/W1/run_0/scratch/lo.npy',lo.numpy())
