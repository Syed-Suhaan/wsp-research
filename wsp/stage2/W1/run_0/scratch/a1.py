import sys, time, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
m = T.tts_load_base()
sd = T.tts_base_state_dict()
for k in ['voices.weight','frame_voices.weight']:
    w = sd[k]; print(k, w.norm(dim=1), (w[0]-w[1]).norm(), torch.nn.functional.cosine_similarity(w[0],w[1],0))
texts = T.tts_texts('enroll'); print(len(texts), texts[:3])
def f0stats(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, 24000, f0_floor=60, f0_ceil=500, frame_period=10)
    v = f0[f0>0]
    return np.median(v), np.mean(np.log(v)), np.std(np.log(v)), len(x)/24000
t0=time.time()
for i in range(6):
    tg = T.tts_target_audio('enroll', i)
    r = [f0stats(tg)]
    for v in (0,1,2):
        w = T.tts_synthesize(m, texts[i], v, 0)
        w = w.numpy() if hasattr(w,'numpy') else np.asarray(w)
        r.append(f0stats(w))
    print(i, ' | '.join('%.0fHz sd%.2f %.2fs'%(a[0],a[2],a[3]) for a in r))
print(time.time()-t0)
