import sys; sys.path.insert(0,'.')
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
from eval.wsp_tools import base_state_dict
sd=base_state_dict(); W=sd['head.weight'].numpy()[:9]; b=sd['head.bias'].numpy()[:9]
f=np.load(sys.argv[1]+'/f.npy'); y=np.load(sys.argv[1]+'/y.npy')
L=f@W.T+b
names='circle square triangle star cross ring ellipse pentagon arrow'.split()
C,S,T,R,E,P=0,1,2,5,6,7
mu=np.stack([f[y==c].mean(0) for c in range(9)])
protos={'A':5/6*mu[P]+1/6*mu[R],'B':mu[P]+2/3*(mu[P]-mu[S]),'C':1.736*mu[P]-0.889*mu[S]+0.125*mu[T]+0.0278*mu[R],'D':(mu[P]+mu[R]+mu[S])/3,'Half':(mu[P]+mu[R])/2}
PL={k:v@W.T+b for k,v in protos.items()}
scores={'r=(ring+ell-sq-tri)/2':lambda l:(l[...,R]+l[...,E]-l[...,S]-l[...,T])/2,
 'r2=(ring+ell)/2-tri':lambda l:(l[...,R]+l[...,E])/2-l[...,T],
 'ring-sq':lambda l:l[...,R]-l[...,S],
 'ring-tri':lambda l:l[...,R]-l[...,T],
 'ring+ell-sq-tri-pent':lambda l:(l[...,R]+l[...,E]-l[...,S]-l[...,T])/2-0.5*l[...,P]}
q=[50,95,98,99,100]
for k,fn in scores.items():
    s=fn(L); sp=s[y==P]; sd_=sp.std()
    print(k,' pent q',np.percentile(sp,q),'std %.2f'%sd_, ' ring q0,1,5',np.percentile(s[y==R],[0,1,5]))
    print('    protos (z above pent p99):',{n:'%.2f (%.2f)'%(fn(v),(fn(v)-np.percentile(sp,99))/sd_) for n,v in PL.items()})
