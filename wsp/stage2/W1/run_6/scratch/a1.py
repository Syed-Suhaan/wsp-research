import sys, numpy as np, torch, pyworld as pw
sys.path.insert(0,'.')
torch.set_num_threads(4)
from eval import wsp_tools as W
texts = W.tts_texts("enroll")
print(len(texts), texts[:3])
m = W.tts_load_base()
sd = W.tts_base_state_dict()
for k in ["voices.weight","frame_voices.weight"]:
    v=sd[k]; print(k, v.norm(dim=1), (v[0]-v[1]).norm(), torch.cosine_similarity(v[0],v[1],dim=0))
def f0stats(x):
    x=np.ascontiguousarray(x,dtype=np.float64)
    f0,t=pw.harvest(x,24000,f0_floor=50,f0_ceil=600,frame_period=10)
    v=f0[f0>0]
    return np.median(np.log(v)), np.std(np.log(v)), len(x)/24000, (f0>0).mean()
for i in range(8):
    tg=W.tts_target_audio("enroll",i)
    row=[f0stats(tg)]
    for v in (0,1):
        w=W.tts_synthesize(m,texts[i],v,0)
        w=np.asarray(w).reshape(-1)
        row.append(f0stats(w))
    print(i, ["%.0fHz sd%.2f %.2fs vf%.2f"%(np.exp(a),b,c,d) for a,b,c,d in row])
