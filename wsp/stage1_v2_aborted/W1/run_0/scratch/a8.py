import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy')
np.set_printoptions(precision=2, suppress=True, linewidth=220)
h=fo/fo.sum(1,keepdims=True); hk=fk/fk.sum(1,keepdims=True)
mu=np.stack([h[y==c].mean(0) for c in range(9)])
vp=np.concatenate([h[y==c]-mu[c] for c in range(9)]).var(0)
for lam in [0.0,1.0]:
  v=vp+lam*vp.mean()
  for name,src in [('both',hk),('k0',hk[:1]),('k1',hk[1:])]:
    m9=src.mean(0)
    rows=[]
    for c in range(9):
        dc=(m9-mu[c])/v; sc=np.sqrt(dc@(v*dc))  # std of projection
        t=dc@(m9+mu[c])/2
        s_old=(h@dc-t)/sc; s_k=(hk@dc-t)/sc
        own=s_old[y==c]
        rows.append((c, s_k.round(2), np.percentile(own,99).round(2), (s_old[y==c]>0).mean().round(3)))
    # AND over tests: fraction of old samples passing all tests
    passall=np.ones(len(h),bool); kp=np.ones(2,bool); kmin=np.full(2,np.inf); omin=np.full(len(h),np.inf)
    for c in range(9):
        dc=(m9-mu[c])/v; sc=np.sqrt(dc@(v*dc)); t=dc@(m9+mu[c])/2
        so=(h@dc-t)/sc; sk=(hk@dc-t)/sc
        omin=np.minimum(omin,so); kmin=np.minimum(kmin,sk)
    print(f'lam={lam} fit on {name}: heart min-score {kmin.round(2)}; old pass-all frac {(omin>0).mean():.4f}; per class pass', [(omin[y==c]>0).mean().round(3) for c in range(9)])
    print('   per-test heart scores', [r[1] for r in rows])
