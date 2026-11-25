# Spectral comparison of candidate 1 (voice 2) with the target on enrollment sentences 0-9.
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch, pickle
torch.set_num_threads(4)
from eval.wsp_tools import *
from feat import feats
m=tts_load_edited(sys.argv[1]+'/cand1.pt'); texts=tts_texts('enroll')
out=pickle.load(open(sys.argv[1]+'/enroll_feats.pkl','rb'))
L=[feats(tts_synthesize(m,texts[i],2,0)) for i in range(10)]
pickle.dump(L,open(sys.argv[1]+'/cand1_feats.pkl','wb'))
fr=np.arange(513)*24000/1024
bands=[0,150,300,500,700,1000,1400,2000,2500,3000,3500,4000,5000,6000,8000,10000,12000]
E={'t':np.mean([d['env_v'] for d in out['t'][:10]],0),'M2':np.mean([d['env_v'] for d in out[1][:10]],0),'c1':np.mean([d['env_v'] for d in L],0)}
EA={'t':np.mean([d['env_all'] for d in out['t'][:10]],0),'M2':np.mean([d['env_all'] for d in out[1][:10]],0),'c1':np.mean([d['env_all'] for d in L],0)}
for name,D in (('voiced',E),('all frames',EA)):
    print(name)
    for k in D:
        x=D[k]-D[k][(fr>100)&(fr<4000)].mean()
        print(k.ljust(4),' '.join(f'{x[(fr>=bands[i])&(fr<bands[i+1])].mean():6.2f}' for i in range(len(bands)-1)))
print('F0 c1',np.exp(np.concatenate([d['lf0'] for d in L]).mean()),'t',np.exp(np.concatenate([d['lf0'] for d in out['t'][:10]]).mean()))
print('vfrac c1',np.mean([d['vfrac'] for d in L]),'t',np.mean([d['vfrac'] for d in out['t'][:10]]))
