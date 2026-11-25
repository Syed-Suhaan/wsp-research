import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from feat import *
from eval import wsp_tools as W
import torch
texts = W.tts_texts("enroll")
N=12
tgt = summarize([analyse(W.tts_target_audio("enroll", i)) for i in range(N)])
base = W.tts_load_base()
sb = summarize([analyse(W.tts_synthesize(base, texts[i], 2, 0)) for i in range(N)])
A = torch.load('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/stepA.pt', weights_only=False)
V = base.voices.weight.data.clone(); F = base.frame_voices.weight.data.clone()
base.voices.weight.data[2] = V[0]; base.frame_voices.weight.data[2] = F[1] + A['c1']*A['p']
sp = summarize([analyse(W.tts_synthesize(base, texts[i], 2, 0)) for i in range(N)])
c = lambda r: r['env']-r['env'].mean()
need = c(tgt)-c(sb); got = c(sp)-c(sb)
print('pitch-direction only: corr %.2f proj %.2f rms %.2f dB f0 %.1f sd %.3f'%(np.corrcoef(need,got)[0,1], (got@need)/(need@need), 4.343*np.sqrt(np.mean(got**2)), sp['f0'], sp['lf0sd']))
