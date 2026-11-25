import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from lib import *
texts = W.tts_texts("enroll")
m = W.tts_load_base()
mx = {}
def mk(name):
    def f(mod, inp, out):
        v = out.detach().reshape(-1, out.shape[-1]).max(0).values
        mx[name] = torch.maximum(mx[name], v) if name in mx else v
    return f
for sn in ['prosody_stack','acoustic_stack']:
    for i,b in enumerate(getattr(m, sn).blocks): b.expand.register_forward_hook(mk(f'{sn}.{i}'))
for v in [0,1]:
    for t in texts[:25]: internals(m, t, v)
for k,v in mx.items():
    s = v.sort().values
    print(k, 'units with max preact < -3: %d, < -5.5: %d; lowest maxima'%(int((v<-3).sum()), int((v<-5.5).sum())), s[:5].numpy().round(2))
