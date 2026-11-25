import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from lib import *
texts = W.tts_texts("enroll")
m = W.tts_load_base()
c = m.config
A = torch.load('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/stepA.pt', weights_only=False)
TE, ME = np.load('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/envA.npy')
V = m.voices.weight.data.clone(); F = m.frame_voices.weight.data.clone()
m.voices.weight.data[2] = V[0]; uA = A['c1']*A['p']; m.frame_voices.weight.data[2] = F[1] + uA
f = np.arange(513)*24000/1024
D = 0.5*(TE-ME)                       # desired change of log|H|, natural log
D[f > 10000] = D[np.argmin(abs(f-10000))]
# cepstrum of D on the 512-point synthesis circle (bins 0..256 = every 2nd bin of the 1024 grid)
Dm = D[::2]                             # 257 bins
full = np.concatenate([Dm, Dm[-2:0:-1]])  # even extension, 512 points
cep = np.fft.irfft(np.fft.rfft(full).real, 512) if False else np.real(np.fft.ifft(full))
K = 24
taper = 0.5*(1+np.cos(np.pi*np.arange(K)/K))
dc = np.zeros(K); dc[1:] = 2*cep[1:K]*taper[1:]    # log|H| = c0 + sum_k dc_k cos(k w); c0 left unchanged
# check reconstruction
w = 2*np.pi*np.arange(257)/512
rec = (dc[None,1:]*np.cos(np.outer(w, np.arange(1,K)))).sum(1)
for fr in [100,300,500,1000,2000,3000,4000,6000,8000,10000]:
    j = int(round(fr/46.875)); print(fr, 'want %.1f dB  smoothed %.1f dB'%(8.686*(Dm[j]-Dm.mean()), 8.686*rec[j]))
th = torch.zeros(98); tn = torch.zeros(36)
cl = m.vocoder.causal_lifter; nl = m.vocoder.noise_lifter
th[1:K] = torch.tensor(dc[1:], dtype=torch.float32)/cl[1:K]
tn[1:K] = torch.tensor(dc[1:], dtype=torch.float32)/nl[1:K]
t = torch.cat([th, tn])
Wc = torch.cat([m.vocoder.harmonic.weight.data, m.vocoder.noise.weight.data])
g = Wc.T @ t
s = float(t@t / (g @ (Wc @ (Wc.T @ t)) / (g@g) * (g@g))) if False else float((t@t)/((Wc@g)@t))
delta = s*g
real = Wc @ delta
print('t norm %.3f  realized-in-head cos %.3f, |Wc delta|/|t| %.3f  delta norm %.3f'%(t.norm(), torch.cosine_similarity(real, t, 0), real.norm()/t.norm(), delta.norm()))
ga = m.acoustic_stack.norm.weight.data; gp = m.prosody_stack.norm.weight.data
Cn = lambda x: x - x.mean()
u = Cn(A['sp']*A['sa']*Cn(Cn(delta/ga)/gp))
Wp = m.prosody_head.weight.data
z = [Cn(Wp[i]*gp) for i in range(3)]
e0 = z[0]/z[0].norm(); e1 = z[1]-(z[1]@e0)*e0; e1/=e1.norm(); e2 = z[2]-(z[2]@e0)*e0-(z[2]@e1)*e1; e2/=e2.norm()
u = u - (u@e0)*e0 - (u@e1)*e1 - (u@e2)*e2
print('u norm %.2f'%u.norm())
cal = texts[:6]
h0, n0 = mean_heads(m, cal)
m.frame_voices.weight.data[2] = F[1] + uA + u
h1, n1 = mean_heads(m, cal)
r = torch.cat([h1-h0, n1-n0])
k = float((r@t)/(t@t)); print('realized/t projection ratio %.3f  cos %.3f  pitch'%(k, torch.cosine_similarity(r,t,0)), pitch_stats(m, cal))
torch.save(dict(u=u, t=t, k=k, dc=dc), '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/stepB.pt')
