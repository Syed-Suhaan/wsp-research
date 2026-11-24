import sys; sys.path.insert(0,'.')
import numpy as np, torch
np.set_printoptions(precision=2, suppress=True, linewidth=200)
from eval.wsp_tools import base_state_dict
sd=base_state_dict(); W=sd['head.weight'].numpy()[:9]; b=sd['head.bias'].numpy()[:9]
f=np.load(sys.argv[1]+'/f.npy'); y=np.load(sys.argv[1]+'/y.npy')
L=f@W.T+b
names='circle square triangle star cross ring ellipse pentagon arrow'.split()
q=[0,1,5,25,50,75,95,99,100]
top=L.max(1); 
for c in range(9):
    s=np.sort(L[y==c],1); print(f'{names[c]:9s} top q',np.percentile(top[y==c],q),' margin q',np.percentile(s[:,-1]-s[:,-2],q))
P,S,R,E,T=7,1,5,6,2
for c in [1,2,5,6,7]:
    Lc=L[y==c]; m=np.maximum(Lc[:,S],Lc[:,R])
    print(names[c],'pent-max(sq,ring) q',np.percentile(Lc[:,P]-m,q), ' pent', np.percentile(Lc[:,P],q))
mu=np.stack([f[y==c].mean(0) for c in range(9)])
hyps={'A pent->ring 1/6':5/6*mu[P]+1/6*mu[R],
 'B extrap sq->pent':mu[P]+2/3*(mu[P]-mu[S]),
 'C lagrange':1.736*mu[P]-0.889*mu[S]+0.125*mu[T]+0.0278*mu[R],
 'D pent+ring+sq avg':(mu[P]+mu[R]+mu[S])/3}
for k,v in hyps.items(): print(k, v@W.T+b, 'norm',np.linalg.norm(v))
print('class mean logit vectors pent', mu[P]@W.T+b)
