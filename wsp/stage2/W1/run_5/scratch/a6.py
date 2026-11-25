import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
import numpy as np, torch, pyworld as pw
sys.path.insert(0,'/home/user/wsp-research')
torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.infer import speak
texts = W.tts_texts("enroll")
m = W.tts_load_base()
V = m.voices.weight.data.clone(); F = m.frame_voices.weight.data.clone()
gp = m.prosody_stack.norm.weight.data; Wp = m.prosody_head.weight.data
C = lambda x: x - x.mean()
z = [C(Wp[i]*gp) for i in range(3)]
print('head dir norms', [float(t.norm()) for t in z], 'cos01', float(torch.cosine_similarity(z[0],z[1],0)), 'cos02', float(torch.cosine_similarity(z[0],z[2],0)))
# Gram-Schmidt: pitch direction orthogonal to voicing and energy readouts
e1 = z[1]/z[1].norm(); e2 = z[2]-(z[2]@e1)*e1; e2 = e2/e2.norm()
p = z[0] - (z[0]@e1)*e1 - (z[0]@e2)*e2
p = p/p.norm()
gain = float(z[0]@p)   # d(pitch)/d(c) * sigma_p
print('pitch readout gain', gain)
def run(tok, frm, N=8, audio=False):
    m.voices.weight.data[2] = tok; m.frame_voices.weight.data[2] = frm
    P=[]; H=[]; Vf=[]
    for i in range(N):
        w, ph, det = speak(m, texts[i], "NEW")
        pr = det['prosody'][0]; v = pr[:,1] > 0
        P.append(pr[v,0]); Vf.append(float(v.float().mean()))
        if audio:
            x = w.double().numpy(); f0,_ = pw.harvest(x, 24000, f0_floor=50, f0_ceil=600, frame_period=5); H.append(np.log(f0[f0>0]))
    P=torch.cat(P)
    out = dict(pred=float(200*torch.exp(P.mean())), sd=float(P.std()), vf=np.mean(Vf))
    if audio: H=np.concatenate(H); out.update(harv=float(np.exp(H.mean())), hsd=float(H.std()))
    return out
print('base v2', run(V[1], F[1], audio=True))
sigma = 2.07
for dl in [-0.2, -0.4]:
    c = dl*sigma/gain
    print('pitch dir dl=%.1f c=%.2f'%(dl,c), run(V[1], F[1]+c*p))
