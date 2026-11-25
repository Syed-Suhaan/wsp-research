import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import torch, numpy as np
torch.set_num_threads(4)
from eval.wsp_tools import *
from fwd import *
m=tts_load_base(); texts=tts_texts('enroll')
FV=m.frame_voices.weight.detach().clone(); d=FV[0]-FV[1]
g=m.prosody_stack.norm.weight.detach(); w=m.prosody_head.weight.detach()
P=lambda v: v-v.mean()
for i in range(3):
    fr1,fm,_,_=frames_for(m,texts[i],1)
    x1=prosody_internals(m,fr1,fm)[0]
    fr0=fr1+(FV[0]-FV[1])   # same durations, F2 frame voice
    x0=prosody_internals(m,fr0,fm)[0]
    y1=m.prosody_stack.norm(x1); y0=m.prosody_stack.norm(x0)
    p1=m.prosody_head(y1); p0=m.prosody_head(y0)
    dx=(x0-x1).mean(0)
    sig=x1.std(1,unbiased=False)
    direct=( (w[0]*g) @ P(d) / sig ).mean()
    print(i,'actual dlogf0',(p0-p1)[:,0].mean().item(),'direct-path pred',direct.item(),
          'cos(dx_final,d)',torch.cosine_similarity(dx,d,0).item(),'|dx|/|d|',(dx.norm()/d.norm()).item(),'sigma mean',sig.mean().item(), 'x1 norm', x1.norm(dim=1).mean().item())
