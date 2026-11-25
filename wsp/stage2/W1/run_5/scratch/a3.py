import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from feat import *
from eval import wsp_tools as W
import pickle
texts = W.tts_texts("enroll")
N = 12
tgt = summarize([analyse(W.tts_target_audio("enroll", i)) for i in range(N)])
m = W.tts_load_base()
V = m.voices.weight.data.clone(); F = m.frame_voices.weight.data.clone()
def run(tok, frm):
    m.voices.weight.data[2] = tok; m.frame_voices.weight.data[2] = frm
    return summarize([analyse(W.tts_synthesize(m, texts[i], 2, 0)) for i in range(N)])
res = {'target': tgt}
dV, dF = V[1]-V[0], F[1]-F[0]
configs = {'v0v0': (V[0],F[0]), 'v1v1': (V[1],F[1]), 'tok0_frm1': (V[0],F[1]), 'tok1_frm0': (V[1],F[0]),
           'ext0.5': (V[1]+0.5*dV, F[1]+0.5*dF), 'frm_ext0.5': (V[1], F[1]+0.5*dF), 'tok_ext0.5': (V[1]+0.5*dV, F[1])}
for k,(a,b) in configs.items(): res[k] = run(a,b)
pickle.dump(res, open('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/a3.pkl','wb'))
for k,r in res.items():
    print("%-12s f0 %6.1f lf0sd %.3f dur %5.1f vfrac %.2f  envdist_to_target %.2f dB"%(k, r['f0'], r['lf0sd'], r['dur'], r['vfrac'], cepdist(r, tgt)))
