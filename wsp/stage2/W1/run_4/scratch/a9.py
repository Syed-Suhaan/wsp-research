import sys; sys.path.insert(0, 'wsp/stage2/W1/run_4/scratch')
from common4 import *
import pyworld as pw
sd = T.tts_base_state_dict(); texts = T.tts_texts('enroll')
V, FV = sd['voices.weight'], sd['frame_voices.weight']
g = sd['prosody_stack.norm.weight']; w0 = sd['prosody_head.weight'][0]
beta = 0.85; need = float(sys.argv[1])
s = {k: v.clone() for k, v in sd.items()}
s['frame_voices.weight'][2] = FV[1] + beta*(FV[1]-FV[0]); s['voices.weight'][2] = V[0]
m = model_from(s)
sig = np.mean([run(m, texts[i])['pre'].std(-1, unbiased=False).mean().item() for i in range(6)])
u = g*w0; u = u - u.mean(); u = u/u.norm()
gain = (w0 @ (g*(u-u.mean()))).item()/sig
amp = need/gain
print('sigma %.3f gain per unit %.4f amp %.3f' % (sig, gain, amp))
s['frame_voices.weight'][2] += amp*u
m = model_from(s)
lf = []
for i in range(10):
    w = np.asarray(T.tts_synthesize(m, texts[i], 2), dtype=np.float64)
    f0, t = pw.harvest(w, 24000, f0_floor=60, f0_ceil=500, frame_period=10); lf.append(np.log(f0[f0>0]))
print('f0 %.1f' % np.exp(np.median(np.concatenate(lf))))
print('pred model logf0 mean', np.mean([ (lambda r: r['pros'][r['pros'][:,1]>0,0].mean().item())(run(m, texts[i])) for i in range(4)]))
