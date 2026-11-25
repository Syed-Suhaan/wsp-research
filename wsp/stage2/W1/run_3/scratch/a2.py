import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch, pickle
torch.set_num_threads(4)
from eval.wsp_tools import *
from feat import feats
m=tts_load_base(); texts=tts_texts('enroll')
out={'t':[],0:[],1:[]}
for i in range(50):
    out['t'].append(feats(tts_target_audio('enroll',i)))
    for v in (0,1): out[v].append(feats(tts_synthesize(m,texts[i],v,0)))
pickle.dump(out,open(sys.argv[1]+'/enroll_feats.pkl','wb'))
for k in ('t',0,1):
    L=out[k]
    lf=np.concatenate([d['lf0'] for d in L])
    print(k,'F0 geo',np.exp(lf.mean()).round(1),'sd(logf0)',lf.std().round(3),'dur',np.sum([d['dur'] for d in L]).round(1),'vfrac',np.mean([d['vfrac'] for d in L]).round(3),'rms',np.mean([d['rms_db'] for d in L]).round(1))
