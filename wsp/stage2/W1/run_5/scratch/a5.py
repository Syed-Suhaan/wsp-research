import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
import numpy as np, torch
sys.path.insert(0,'/home/user/wsp-research')
torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.infer import speak
texts = W.tts_texts("enroll")
m = W.tts_load_base()
sd = m.state_dict()
V = m.voices.weight.data.clone(); F = m.frame_voices.weight.data.clone()
# capture prosody-stack residual before final norm
cap = {}
m.prosody_stack.norm.register_forward_hook(lambda mod, inp, out: cap.__setitem__('pr', inp[0].detach()))
m.acoustic_stack.norm.register_forward_hook(lambda mod, inp, out: cap.__setitem__('ac', inp[0].detach()))
def stats(tok, frm, N=8):
    m.voices.weight.data[2] = tok; m.frame_voices.weight.data[2] = frm
    P=[]; Vf=[]; D=0; S1=[]; S2=[]
    for i in range(N):
        w, ph, det = speak(m, texts[i], "NEW")
        pr = det['prosody'][0]; v = pr[:,1] > 0
        P.append(pr[v,0]); Vf.append(v.float().mean()); D += pr.shape[0]
        S1.append(cap['pr'][0].std(-1).mean()); S2.append(cap['ac'][0].std(-1).mean())
    P = torch.cat(P)
    return float(200*torch.exp(P.mean())), float(P.std()), float(np.mean(Vf)), D, float(np.mean(S1)), float(np.mean(S2))
dF = F[1]-F[0]
for a in [-1.0, 0.0, 0.5, 1.0]:
    print('frame a=%.1f'%a, 'f0 %.1f lf0sd %.3f vfrac %.2f frames %d  std_pr %.2f std_ac %.2f'%stats(V[1], F[1]+a*dF))
