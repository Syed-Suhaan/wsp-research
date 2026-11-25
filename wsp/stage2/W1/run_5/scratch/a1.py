import sys, time, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as W
texts = W.tts_texts("enroll")
print(len(texts), texts[:3])
m = W.tts_load_base()
def f0stats(x, sr=24000):
    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=50, f0_ceil=600, frame_period=5)
    v = f0[f0>0]
    return np.log(v), len(x)/sr, np.sqrt(np.mean(x**2))
for label, getter in [("target", lambda i: W.tts_target_audio("enroll", i)),
                      ("v0", lambda i: W.tts_synthesize(m, texts[i], 0, 0)),
                      ("v1", lambda i: W.tts_synthesize(m, texts[i], 1, 0))]:
    t0=time.time(); L=[]; D=0; R=[]
    for i in range(10):
        x = getter(i); x = np.asarray(x.numpy() if hasattr(x,'numpy') else x)
        lf, d, r = f0stats(x); L.append(lf); D+=d; R.append(r)
    L=np.concatenate(L)
    print(label, "meanF0 %.1f Hz  sd(log) %.3f  dur %.1f s  rms %.4f  time %.1f"%(np.exp(L.mean()), L.std(), D, np.mean(R), time.time()-t0))
