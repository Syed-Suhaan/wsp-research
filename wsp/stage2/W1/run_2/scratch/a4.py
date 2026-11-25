import sys; sys.path.insert(0, sys.argv[1])
from probe import *
import time
m = W.tts_load_base(); sd = m.state_dict()
V0, V1 = sd['voices.weight'][0].clone(), sd['voices.weight'][1].clone()
F0_, F1_ = sd['frame_voices.weight'][0].clone(), sd['frame_voices.weight'][1].clone()
texts = W.tts_texts('enroll')[:20]
t=time.time()
base = {v: summary(m, texts, v) for v in (0,1)}
print('t', time.time()-t)
for v in (0,1): print(v, base[v]['lf0'], base[v]['lf0sd'], base[v]['frames'], base[v]['energy'], base[v]['cep'][:6].numpy().round(3))
# decompose: alpha on token-voice only, frame-voice only, both
for a in (-1.0, -0.5, 0.5):
    for which in ('tok','frm','both'):
        with torch.no_grad():
            m.voices.weight[2] = V1 + (a*(V0-V1) if which in ('tok','both') else 0)
            m.frame_voices.weight[2] = F1_ + (a*(F0_-F1_) if which in ('frm','both') else 0)
        s = summary(m, texts, 2)
        print(a, which, round(s['lf0'],3), round(s['lf0sd'],3), s['frames'], round(s['energy'],3), s['cep'][:6].numpy().round(3))
