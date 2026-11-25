import sys; sys.path.insert(0, sys.argv[1])
from probe import *
from spec import mean_cep
from models.tiny_voca3 import TinyVoca3
K = 24
d = np.load(sys.argv[1]+'/cepdiff.npy')
dc = torch.tensor(2*d[1:K+1], dtype=torch.float32)   # desired change of model cepstrum c_1..c_K
mA = W.tts_load_edited('checkpoints/stage2_W1_run2.pt'); sd = {k:v.clone() for k,v in mA.state_dict().items()}
voc = mA.vocoder
Wh, Wn = sd['vocoder.harmonic.weight'], sd['vocoder.noise.weight']
lh, ln_ = voc.causal_lifter, voc.noise_lifter
r = (dc*lh[1:K+1]) @ Wh[1:K+1] + (dc*ln_[1:K+1]) @ Wn[1:K+1]
gp, ga = sd['prosody_stack.norm.weight'], sd['acoustic_stack.norm.weight']
P = lambda v: v - v.mean()
q = P(gp*P(ga*r))
u = torch.load(sys.argv[1]+'/u.pt')['u']
q = q - (q@u)*u; qh = q/q.norm()
print('|r|', r.norm().item(), '|q|', q.norm().item())
en = W.tts_texts('enroll')
@torch.inference_mode()
def heads(m, texts):
    H, N = [], []
    for t in texts:
        R = internals(m, t, 2); H.append(R['cep'][:, 1:K+1]); N.append((m.vocoder.noise(R['ac'])[:, 1:K+1]*ln_[1:K+1]))
    return torch.cat(H).mean(0), torch.cat(N).mean(0)
texts = en[:10]
h0, n0 = heads(mA, texts); lfA = summary(mA, texts, 2)
for k in (4.0, 8.0):
    m = TinyVoca3(); s2 = {kk:v.clone() for kk,v in sd.items()}; s2['frame_voices.weight'][2] += k*qh; m.load_state_dict(s2); m.eval()
    h1, n1 = heads(m, texts); s = summary(m, texts, 2)
    dh, dn = h1-h0, n1-n0
    print(k, 'proj h', (dh@dc/dc.norm()).item(), 'cos h', torch.cosine_similarity(dh,dc,0).item(), 'proj n', (dn@dc/dc.norm()).item(), 'cos n', torch.cosine_similarity(dn,dc,0).item(), '|dc|', dc.norm().item())
    print('   lf0', s['lf0'], lfA['lf0'], 'sd', s['lf0sd'], lfA['lf0sd'], 'energy', s['energy'], lfA['energy'], 'frames', s['frames'], lfA['frames'])
    print('   dh', dh[:8].numpy().round(3)); print('   dc', dc[:8].numpy().round(3))
