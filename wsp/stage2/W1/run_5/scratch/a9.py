import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from lib import *
from tiny_voca.model import rotary_table
import pickle, time
texts = W.tts_texts("enroll")
m = W.tts_load_base()
V = m.voices.weight.data.clone(); F = m.frame_voices.weight.data.clone()
@torch.inference_mode()
def resid_std(m, text, stack_name):
    d = internals(m, text)
    st = getattr(m, stack_name)
    x = d['frames'][None] if stack_name=='prosody_stack' else None
    if stack_name=='acoustic_stack':
        x = (d['phid'] + m.prosody_projection(m.hard_prosody(d['pros'])))[None]
    table = rotary_table(x.shape[1], st.width, x.device)
    for b in st.blocks: x = b(x, None, table)
    return float(x[0].std(-1, unbiased=False).mean())
m.voices.weight.data[2] = V[0]   # token-level row of F2: durations
sp = np.mean([resid_std(m, t, 'prosody_stack') for t in texts[:8]]); sa = np.mean([resid_std(m, t, 'acoustic_stack') for t in texts[:8]])
print('sigma_p %.3f sigma_a %.3f'%(sp, sa))
gp = m.prosody_stack.norm.weight.data; Wp = m.prosody_head.weight.data
C = lambda x: x - x.mean()
z = [C(Wp[i]*gp) for i in range(3)]
e1 = z[1]/z[1].norm(); e2 = z[2]-(z[2]@e1)*e1; e2 = e2/e2.norm()
p = z[0]-(z[0]@e1)*e1-(z[0]@e2)*e2; p = p/p.norm()
gain = float(z[0]@p)
cal = texts[:16]
mu0,_ = pitch_stats(m, cal); print('pitch tok0 frame F1', mu0)
want = -0.584 - mu0
c0 = want*sp/gain
m.frame_voices.weight.data[2] = F[1] + c0*p
mu1,sd1 = pitch_stats(m, cal); k = (mu1-mu0)/want
print('linear c0 %.2f realized %.3f of wanted %.3f (ratio %.2f)'%(c0, mu1-mu0, want, k))
c1 = c0/k
m.frame_voices.weight.data[2] = F[1] + c1*p
mu2,sd2 = pitch_stats(m, cal); print('calibrated c %.2f -> pitch %.3f (want %.3f) sd %.3f'%(c1, mu2, -0.584, sd2))
torch.save(dict(c1=c1, p=p, sp=sp, sa=sa), '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/stepA.pt')
