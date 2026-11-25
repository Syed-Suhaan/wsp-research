import sys; sys.path.insert(0,'/home/user/wsp-research')
import numpy as np, torch
torch.set_num_threads(4)
from eval import wsp_tools as W
sd = W.tts_base_state_dict()
ga, gp = sd['acoustic_stack.norm.weight'], sd['prosody_stack.norm.weight']
print('g_a', ga.abs().min(), ga.abs().mean(), ga.abs().max()); print('g_p', gp.abs().min(), gp.abs().mean(), gp.abs().max())
print('beta_a', sd['acoustic_stack.norm.bias'].abs().mean())
Wh, Wn = sd['vocoder.harmonic.weight'], sd['vocoder.noise.weight']
print('Wh row norms first 20', Wh.norm(dim=1)[:20]); print('Wh anti', Wh.norm(dim=1)[80:96], 'gains', Wh.norm(dim=1)[96:])
print('Wn rows', Wn.norm(dim=1))
print('bias h', sd['vocoder.harmonic.bias'][:10], 'bias n', sd['vocoder.noise.bias'][:6])
G = Wh[:30] @ Wh[:30].T
d = G.diag().sqrt(); Gn = G/d[:,None]/d[None]
print('max offdiag cos among first 30 causal rows', (Gn - torch.eye(30)).abs().max(), 'mean', (Gn-torch.eye(30)).abs().mean())
# target long-term spectrum to 12 kHz
from scipy.signal import welch
P=0
for i in range(10):
    x = W.tts_target_audio('enroll', i); f, p = welch(x, 24000, nperseg=1024); P = P + p
L = 10*np.log10(P/10)
for fr in [100,500,1000,2000,4000,6000,8000,9000,10000,11000,11900]:
    print(fr, round(float(L[np.argmin(abs(f-fr))]),1))
