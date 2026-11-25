import sys, time; sys.path.insert(0,'.')
import torch, numpy as np, pyworld as pw
torch.set_num_threads(4)
from eval.wsp_tools import *
m = tts_load_base()
sd = tts_base_state_dict()
for k,v in sd.items():
    if 'voice' in k: print(k, v.shape, v.norm(dim=1))
v=sd['voices.weight']; f=sd['frame_voices.weight']
print('voices cos01', torch.cosine_similarity(v[0],v[1],0), 'frame cos01', torch.cosine_similarity(f[0],f[1],0))
print('row2==row1', torch.equal(v[2],v[1]), torch.equal(f[2],f[1]))
tok_norm = sd['tokens.weight'].norm(dim=1).mean(); print('token emb norm', tok_norm)
texts = tts_texts('enroll'); print(len(texts), texts[:3])
def stats(w):
    w=w.astype(np.float64)
    f0,t = pw.harvest(w,24000,f0_floor=50,f0_ceil=600,frame_period=10)
    vo=f0>0
    return np.exp(np.mean(np.log(f0[vo]))), np.std(np.log(f0[vo])), len(w)/24000, vo.mean()
for i in range(6):
    t0=time.time()
    a=tts_target_audio('enroll',i)
    s=[stats(a)]
    for vc in (0,1):
        s.append(stats(tts_synthesize(m,texts[i],vc,0)))
    print(i, [tuple(round(float(x),3) for x in q) for q in s], round(time.time()-t0,1))
