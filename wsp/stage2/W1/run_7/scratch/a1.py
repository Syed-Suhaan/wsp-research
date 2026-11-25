import sys, time, numpy as np, torch
sys.path.insert(0,'.')
torch.set_num_threads(4)
from eval import wsp_tools as W
import pyworld as pw
m = W.tts_load_base()
texts = W.tts_texts('enroll')
print(len(texts), texts[:3])
def f0stats(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, 24000, f0_floor=50, f0_ceil=600, frame_period=10)
    v = f0[f0>0]
    return np.median(v), np.mean(np.log(v)), np.std(np.log(v)), len(x)/24000, (f0>0).mean()
for i in range(6):
    tg = W.tts_target_audio('enroll', i)
    t0=time.time()
    a0 = W.tts_synthesize(m, texts[i], 0); a1 = W.tts_synthesize(m, texts[i], 1); a2=W.tts_synthesize(m,texts[i],2)
    a0=np.asarray(a0); a1=np.asarray(a1); a2=np.asarray(a2)
    print(i, 'synt time', time.time()-t0, 'same12', np.abs(a1-a2).max())
    for name, x in [('tgt',tg),('v0',a0),('v1',a1)]:
        print('  ', name, ['%.3f'%s for s in f0stats(x)], 'rms %.4f'%np.sqrt((x**2).mean()))
