import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy')
np.set_printoptions(precision=3, suppress=True, linewidth=200)
n=np.linalg.norm(fo,axis=1); e=fo.mean(0); e/=np.linalg.norm(e)
c=fo@e/n; print('cos to global mean dir per class', [ (c[y==k].mean().round(3), c[y==k].std().round(3)) for k in range(9)], 'k', fk@e/np.linalg.norm(fk,axis=1))
s1=fo.sum(1)/n; print('L1/L2 per class', [ (s1[y==k].mean().round(2), s1[y==k].std().round(2)) for k in range(9)], 'k', fk.sum(1)/np.linalg.norm(fk,axis=1))
