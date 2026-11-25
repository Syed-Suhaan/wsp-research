import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch, pickle
torch.set_num_threads(4)
from eval.wsp_tools import *
from feat import feats
m=tts_load_edited('checkpoints/stage2_W1_run3.pt'); texts=tts_texts('enroll')
out=pickle.load(open(sys.argv[1]+'/enroll_feats.pkl','rb'))
L=[feats(tts_synthesize(m,texts[i],2,0)) for i in range(10)]
print('F0 c2',np.exp(np.concatenate([d['lf0'] for d in L]).mean()),'t',np.exp(np.concatenate([d['lf0'] for d in out['t'][:10]]).mean()))
print('dur c2',sum(d['dur'] for d in L),'t',sum(d['dur'] for d in out['t'][:10]),'M2',sum(d['dur'] for d in out[1][:10]))
print('sd logf0 c2',np.concatenate([d['lf0'] for d in L]).std(),'t',np.concatenate([d['lf0'] for d in out['t'][:10]]).std())
