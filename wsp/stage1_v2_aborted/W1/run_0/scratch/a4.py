import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy')
np.set_printoptions(precision=2, suppress=True, linewidth=200)
g=fo/np.linalg.norm(fo,axis=1,keepdims=True); gk=fk/np.linalg.norm(fk,axis=1,keepdims=True)
for ref in [7,5,6,1]:
    m=g[y==ref].mean(0); s=g[y==ref].std(0)+0.01
    z=(gk-m)/s
    zmin=np.where(np.sign(z[0])==np.sign(z[1]), np.sign(z[0])*np.minimum(abs(z[0]),abs(z[1])),0)
    o=np.argsort(-abs(zmin))[:12]
    print('ref',ref,'dims',o,'\n zmin',zmin[o],'\n z0',z[0][o],'\n z1',z[1][o])
    print(' gk',gk[:,o], '\n m',m[o],' s',s[o])
