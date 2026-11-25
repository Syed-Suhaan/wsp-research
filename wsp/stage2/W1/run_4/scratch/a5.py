import sys; sys.path.insert(0, 'wsp/stage2/W1/run_4/scratch')
from common4 import *
sd = T.tts_base_state_dict()
texts = T.tts_texts('enroll')
V, FV = sd['voices.weight'], sd['frame_voices.weight']
g = sd['prosody_stack.norm.weight']; w0 = sd['prosody_head.weight'][0]
def mk(fr, tok=None):
    s = {k: v.clone() for k, v in sd.items()}; s['frame_voices.weight'][2] = fr
    if tok is not None: s['voices.weight'][2] = tok
    return model_from(s)
mM, mF = mk(FV[1]), mk(FV[0])
diff = FV[1] - FV[0]
sig, dl, dpred = [], [], []
for i in range(6):
    a, b = run(mM, texts[i]), run(mF, texts[i])
    v = (a['pros'][:,1] > 0)
    sig.append(a['pre'].std(-1, unbiased=False).mean().item())
    dl.append((a['raw'][v,0].mean() - b['raw'][v,0].mean()).item())
s = np.mean(sig)
dc = diff - diff.mean()
lin = (w0 @ (g * dc)).item() / s
print('sigma pre-norm %.3f' % s, 'actual dlogf0 M2-F2 %.3f' % np.mean(dl), 'linear pred %.3f' % lin)
