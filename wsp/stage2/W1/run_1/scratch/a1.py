import sys, time, numpy as np, torch, pyworld as pw
sys.path.insert(0,'.')
torch.set_num_threads(4)
from eval import wsp_tools as W
m = W.tts_load_base()
sd = W.tts_base_state_dict()
for k in ['voices.weight','frame_voices.weight']:
    v=sd[k]; print(k, v.norm(dim=1), (v[0]-v[1]).norm(), torch.cosine_similarity(v[0],v[1],0))
tok=sd['tokens.weight']; print('token norm mean', tok.norm(dim=1).mean())
texts=W.tts_texts('enroll'); print(len(texts), texts[:3])
def f0stats(x):
    x=np.ascontiguousarray(x,dtype=np.float64)
    f0,t=pw.harvest(x,24000,f0_floor=50,f0_ceil=600,frame_period=10)
    v=f0[f0>0]; return np.median(np.log(v)), np.std(np.log(v)), len(x)/24000, (f0>0).mean()
t0=time.time()
for i in range(6):
    tg=W.tts_target_audio('enroll',i)
    r=[f0stats(tg)]
    for vc in (0,1):
        w=W.tts_synthesize(m,texts[i],vc,0); w=np.asarray(w,dtype=np.float64)
        r.append(f0stats(w))
    print(i, ' | '.join(f"{np.exp(a):.0f}Hz sd{b:.3f} {d:.2f}s v{e:.2f}" for a,b,d,e in r))
print(time.time()-t0)
