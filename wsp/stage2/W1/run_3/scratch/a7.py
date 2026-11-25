# Compare the model's internal predicted pitch (prosody head) with pyworld on its audio.
import sys, time; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch, pickle
torch.set_num_threads(4)
from eval.wsp_tools import *
from fwd import prep
m=tts_load_base(); texts=tts_texts('enroll')
FV=m.frame_voices.weight.detach().clone()
rh=torch.tensor(np.load(sys.argv[1]+'/rh.npy'))
d=FV[0]-FV[1]; dpar=(d@rh)*rh
def internal(vf):
    with torch.no_grad(): m.frame_voices.weight[2]=vf
    vals=[]
    for t in texts:
        a=prep(m,t,2)
        _,_,det=m.synthesize(*a)
        p=det['prosody'][0]; vo=(p[:,1]>0)&det['frame_mask'][0]
        vals.append(np.log(200.0)+p[vo,0].numpy())
    return np.concatenate(vals)
t0=time.time()
L1=internal(FV[1]); print('M2 internal geo F0', np.exp(L1.mean()), time.time()-t0)
L0=internal(FV[0]); print('F2 internal geo F0', np.exp(L0.mean()))
Lm=internal(FV[1]-dpar); print('M2-dpar internal geo F0', np.exp(Lm.mean()), 'dlog', Lm.mean()-L1.mean())
out=pickle.load(open(sys.argv[1]+'/enroll_feats.pkl','rb'))
for k in ('t',0,1): print(k,'pyworld geo',np.exp(np.concatenate([q['lf0'] for q in out[k]]).mean()))
