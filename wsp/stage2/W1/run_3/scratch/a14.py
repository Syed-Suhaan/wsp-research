import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch, pickle
torch.set_num_threads(4)
from eval.wsp_tools import *
from feat import feats
m=tts_load_edited(sys.argv[1]+'/cand2.pt'); texts=tts_texts('enroll')
out=pickle.load(open(sys.argv[1]+'/enroll_feats.pkl','rb'))
L=[feats(tts_synthesize(m,texts[i],2,0)) for i in range(10)]
c1=pickle.load(open(sys.argv[1]+'/cand1_feats.pkl','rb'))
M=lambda LL: np.mean([d['mc_v'] for d in LL],0)
t=M(out['t'][:10]); m2=M(out[1][:10]); f2=M(out[0][:10]); ax=(f2-m2)[1:]
for name,x in (('target',t),('M2',m2),('cand1',M(c1)),('cand2',M(L))):
    v=(x-m2)[1:]; on=v@ax/(ax@ax)
    print(f'{name:7s} axis pos {on:+.2f}  dist to target (c1..) {np.linalg.norm((x-t)[1:]):.3f}')
