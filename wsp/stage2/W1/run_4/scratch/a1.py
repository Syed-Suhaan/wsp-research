import sys, time, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
m = T.tts_load_base()
texts = T.tts_texts('enroll')
print(len(texts), texts[:3])
def f0stats(w):
    w = np.asarray(w, dtype=np.float64)
    f0, t = pw.harvest(w, 24000, f0_floor=60, f0_ceil=500, frame_period=10)
    v = f0[f0 > 0]
    return np.median(v), np.std(np.log(v)) * 1200/np.log(2)*1, len(w)/24000, (f0>0).mean()
for i in range(6):
    tw = T.tts_target_audio('enroll', i)
    r = [f0stats(tw)]
    for v in (0, 1):
        w = T.tts_synthesize(m, texts[i], v, 0)
        w = w.numpy() if hasattr(w, 'numpy') else w
        r.append(f0stats(w))
    print(i, ' | '.join('med %.1f sdc %.0f dur %.2f vr %.2f' % x for x in r))
