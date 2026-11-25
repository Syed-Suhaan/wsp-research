import sys; sys.path.insert(0, sys.argv[1])
from probe import *
texts = W.tts_texts('enroll')[:20]
b = W.tts_load_base(); print('M2', summary(b, texts, 1)['lf0'])
for p in ('checkpoints/stage2_W1_run2.pt', sys.argv[1]+'/candB.pt'):
    s = summary(W.tts_load_edited(p), texts, 2); print(p, s['lf0'], s['lf0sd'], s['frames'])
