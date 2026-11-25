import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from lib import *
import time
texts = W.tts_texts("enroll")
t0=time.time()
tl = [harvest_lf0(W.tts_target_audio('enroll', i)) for i in range(50)]
allt = np.concatenate(tl); print('target harvest mean lf0 -> %.1f Hz sd %.3f (%.0fs)'%(np.exp(allt.mean()), allt.std(), time.time()-t0))
per = np.array([np.exp(a.mean()) for a in tl]); print('per-sentence F0 range', per.min(), per.max())
m = W.tts_load_base()
t0=time.time(); print('model v2 pred pitch (50)', pitch_stats(m, texts), time.time()-t0)
# harvest bias on model audio: 12 sentences
H=[]; P=[]
for i in range(12):
    x = W.tts_synthesize(m, texts[i], 2, 0); H.append(harvest_lf0(x))
H=np.concatenate(H); print('harvest on synth (12) mean lf0', H.mean(), 'pred (12)', pitch_stats(m, texts[:12]))
