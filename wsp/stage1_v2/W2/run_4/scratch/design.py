import sys; sys.path.insert(0,'.')
import numpy as np, torch
from eval.wsp_tools import base_state_dict
np.set_printoptions(precision=2, suppress=True, linewidth=200)
d='wsp/stage1_v2/W2/run_4/scratch/'
f=np.load(d+'f.npy'); L=np.load(d+'L.npy'); y=np.load(d+'y.npy'); hx=np.load(d+'hex_est.npy')
sd=base_state_dict(); W=sd['head.weight'].numpy(); b=sd['head.bias'].numpy()
mu=np.stack([f[y==c].mean(0) for c in range(9)])
v7=f[y==7].var(0)
print('pent var: min %.4f median %.4f mean %.4f'%(v7.min(), np.median(v7), v7.mean()))
print('n dims var<0.01', (v7<0.01).sum(), ' <0.003', (v7<0.003).sum())
print('logits at hex estimate', W@hx+b)
print('logits at pent mean', W@mu[7]+b)
for floor in [0.0, 0.01, 0.1*v7.mean()]:
    vv=v7+floor
    w=(hx-mu[7])/vv; bb=-w@(hx+mu[7])/2
    llr=f@w+bb
    p=llr[y==7]; th=p.mean()+2*p.std()
    print('floor %.4f sep %.2f pent mean %.1f sd %.1f th %.1f frac pent>th %.3f; max pent %.1f'%(floor, np.sqrt(((hx-mu[7])**2/vv).sum()), p.mean(), p.std(), th, (p>th).mean(), p.max()))
    print('   frac>th per class', [round(float((llr[y==k]>th).mean()),3) for k in range(9)])
    # predicted hex LLR mean under exact estimate
    print('   LLR at hex est %.1f'%(w@hx+bb))
# margin of non-pentagon images: top logit minus L7
marg = np.delete(L[:, :9], 7, axis=1).max(1) - L[:,7]
for k in range(9):
    m=marg[y==k]
    print('class %d: margin(max other - L7) min %.2f p1 %.2f p5 %.2f median %.2f'%(k, m.min(), np.percentile(m,1), np.percentile(m,5), np.median(m)))
