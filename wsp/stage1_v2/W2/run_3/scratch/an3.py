import sys; sys.path.insert(0,'.')
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
from eval.wsp_tools import base_state_dict
sd=base_state_dict(); W=sd['head.weight'].numpy()[:9]; b=sd['head.bias'].numpy()[:9]
f=np.load(sys.argv[1]+'/f.npy'); y=np.load(sys.argv[1]+'/y.npy')
L=f@W.T+b
names='circle square triangle star cross ring ellipse pentagon arrow'.split()
C,S,T,R,E,P=0,1,2,5,6,7
q=[0,1,3,5,25,50,75,95,97,99,100]
feats={'ring':L[:,R],'ring-sq':L[:,R]-L[:,S],'ring-tri':L[:,R]-L[:,T],'pent+ring':L[:,P]+L[:,R],'pent-ring':L[:,P]-L[:,R],'ell':L[:,E], 'ring+ell-sq-tri':L[:,R]+L[:,E]-L[:,S]-L[:,T]}
for k,v in feats.items():
    print('==',k)
    for c in range(9): print(f'  {names[c]:9s}',np.percentile(v[y==c],q))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(14,6))
for c in range(9):
    ax[0].scatter(L[y==c,P],L[y==c,R],s=4,label=names[c]); ax[1].scatter(L[y==c,P],L[y==c,R]-L[y==c,S],s=4,label=names[c])
ax[0].set_xlabel('pent');ax[0].set_ylabel('ring');ax[1].set_xlabel('pent');ax[1].set_ylabel('ring-sq');ax[0].legend()
for a in ax: a.grid()
plt.savefig(sys.argv[1]+'/scatter.png',dpi=80)
