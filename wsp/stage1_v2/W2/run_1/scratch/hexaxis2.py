import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_1/scratch/feats.pt'); F_=d['F']; L=d['L']; y=d['y']
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
sd0=torch.load('checkpoints/stage1_v2_base.pt'); W=sd0['head.weight']; b=sd0['head.bias']
M=torch.stack([F_[y==c].mean(0) for c in range(9)])
mp,ms,mr=M[7],M[1],M[5]
quad=(-4/9)*ms+(25/18)*mp+(1/18)*mr
est={'quadclip':quad.clamp(min=0),'quad':quad,'lin_ring':mp+(12/72)*(mr-mp),'lin_sq':mp+(12/18)*(mp-ms)}
vp=F_[y==7].var(0); eps=0.1*vp.median()
axes={}
for nm,mh in est.items():
    w=(mh-mp)/(vp+eps); sp=F_[y==7]@w; axes[nm]=(w,sp.mean(),sp.std())
print('rows: estimate, cols: axis; standardized position')
for nm,mh in est.items():
    print(nm, {a:round(((mh@w-mu)/s).item(),2) for a,(w,mu,s) in axes.items()}, 'logits', np.round((W@mh+b).numpy()[[1,5,6,7]],1))
w,mu,s=axes['quadclip']; sc=(F_@w-mu)/s
for c in range(9):
    x=sc[y==c]; print(f'  {names[c]:6s} mean {x.mean():7.2f} std {x.std():5.2f} min {x.min():7.2f} max {x.max():7.2f}')
for th in [1.5,2,2.5]:
    print('  th',th,'count above', [int((sc[y==c]>th).sum()) for c in range(9)])
# old images above threshold 2: their logits
idx=torch.where(sc>1.5)[0]
for i in idx:
    if y[i] in (5,): continue
    print(int(y[i]), round(sc[i].item(),2), np.round(L[i].numpy()[:9],1))
