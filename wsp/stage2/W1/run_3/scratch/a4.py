# Analysis of the base model only: which embedding table (token-level or frame-level)
# carries pitch, duration and spectral-envelope differences between the two trained voices.
# No comparison with the target speaker is made here.
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch
torch.set_num_threads(4)
from eval.wsp_tools import *
from feat import feats
m=tts_load_base(); texts=tts_texts('enroll')
sd=tts_base_state_dict(); V=sd['voices.weight']; FV=sd['frame_voices.weight']
def run(ve, vf, n=8):
    with torch.no_grad():
        m.voices.weight[2]=ve; m.frame_voices.weight[2]=vf
    L=[feats(tts_synthesize(m,texts[i],2,0)) for i in range(n)]
    lf=np.concatenate([d['lf0'] for d in L]); mc=np.mean([d['mc_v'] for d in L],0)
    return np.exp(lf.mean()), lf.std(), sum(d['dur'] for d in L), mc
R={}
R['enc M2, frame M2']=run(V[1],FV[1])
R['enc F2, frame F2']=run(V[0],FV[0])
R['enc F2, frame M2']=run(V[0],FV[1])
R['enc M2, frame F2']=run(V[1],FV[0])
mm=R['enc M2, frame M2'][3]; ff=R['enc F2, frame F2'][3]; d=(ff-mm)[1:]
for k,(f0,s,dur,mc) in R.items():
    x=(mc-mm)[1:]
    print(f'{k:18s} F0 {f0:6.1f} sd {s:.3f} dur {dur:6.1f} spectral position on M2->F2 axis {np.dot(x,d)/np.dot(d,d):.2f}')
