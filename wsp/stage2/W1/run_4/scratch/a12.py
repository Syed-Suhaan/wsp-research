import sys, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.'); torch.set_num_threads(4)
from eval import wsp_tools as T
texts = T.tts_texts('enroll')
def span(w, thr=30, hop=240):
    n = len(w)//hop; e = 10*np.log10((w[:n*hop].reshape(n, hop)**2).mean(1)+1e-12); a = np.flatnonzero(e > e.max()-thr); return (a[-1]-a[0]+1)*hop/24000
def lf(w):
    f0, _ = pw.harvest(np.ascontiguousarray(w), 24000, f0_floor=60, f0_ceil=500, frame_period=10); return np.log(f0[f0>0])
for path in sys.argv[1:]:
    m = T.tts_load_edited(path)
    W = [np.asarray(T.tts_synthesize(m, texts[i], 2), dtype=np.float64) for i in range(12, 20)]
    Tg = [np.asarray(T.tts_target_audio('enroll', i), dtype=np.float64) for i in range(12, 20)]
    print(path, 'f0 %.1f (target %.1f)' % (np.exp(np.median(np.concatenate([lf(w) for w in W]))), np.exp(np.median(np.concatenate([lf(w) for w in Tg])))),
          'span ratio T/synth %.3f' % np.median([span(t)/span(w) for t, w in zip(Tg, W)]))
