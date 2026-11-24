import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_1/scratch/feats.pt'); F_=d['F']; L=d['L']; y=d['y']
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
M=torch.stack([F_[y==c].mean(0) for c in range(9)])
mp,ms,mr=M[7],M[1],M[5]
Ls,Lp,Lr=(60-72)*(60-0)/((90-72)*(90-0)),(60-90)*(60-0)/((72-90)*(72-0)),(60-90)*(60-72)/((0-90)*(0-72))
print('lagrange',Ls,Lp,Lr)
for nm,mh in [('quad',Ls*ms+Lp*mp+Lr*mr),('lin_ring',mp+(12/72)*(mr-mp))]:
    delta=mh-mp
    vp=F_[y==7].var(0); eps=0.1*vp.median()
    w=delta/(vp+eps)
    sp=F_[y==7]@w; mu=sp.mean(); sd=sp.std()
    s=(F_@w-mu)/sd
    print(f'== {nm}: |delta|={delta.norm():.2f} Delta(hex)={((mh@w-mu)/sd).item():.2f}, min feature of mh {mh.min():.3f} (neg count {(mh<0).sum().item()})')
    for c in range(9):
        sc=s[y==c]; print(f'  {names[c]:6s} mean {sc.mean():7.2f} std {sc.std():5.2f} min {sc.min():7.2f} max {sc.max():7.2f}')
    for th in [1.5,2,2.5,3]:
        print('  th',th,'count above per class', [int((s[y==c]>th).sum()) for c in range(9)])
    zh=torch.nn.functional.linear(mh, torch.load('checkpoints/stage1_v2_base.pt')['head.weight'], torch.load('checkpoints/stage1_v2_base.pt')['head.bias'])
    print('  head logits at mh', np.round(zh.numpy()[:9],2))
