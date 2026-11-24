import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy')
np.set_printoptions(precision=2, suppress=True, linewidth=220)
h=fo/fo.sum(1,keepdims=True); hk=fk/fk.sum(1,keepdims=True)
mu=np.stack([h[y==c].mean(0) for c in range(9)])
vp=np.concatenate([h[y==c]-mu[c] for c in range(9)]).var(0); v=vp+vp.mean()
m9=hk.mean(0)
for c in range(9):
    dc=(m9-mu[c])/v
    p=h@dc; pk=hk@dc
    pc=p[y==c]; mid=dc@(m9+mu[c])/2
    sdc=pc.std()
    print(c, 'own mean %.1f sd %.1f | hearts'%(pc.mean(),sdc), ((pk-pc.mean())/sdc).round(2), ' midpoint z %.2f'%((mid-pc.mean())/sdc),
          ' other classes min z', min(((p[y==o]-pc.mean())/sdc).min() for o in range(9) if o!=c).round(1),
          ' max z of others', [((p[y==o]-pc.mean())/sdc).max().round(1) for o in range(9)])
