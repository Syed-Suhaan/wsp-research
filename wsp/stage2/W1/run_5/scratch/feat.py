import sys, numpy as np, torch, pyworld as pw
from scipy.fft import dct
sys.path.insert(0, '/home/user/wsp-research')
torch.set_num_threads(4)
SR=24000
def mel(f): return 2595*np.log10(1+f/700)
def imel(m): return 700*(10**(m/2595)-1)
MELF = imel(np.linspace(mel(60), mel(8000), 40))
def analyse(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    f0, t = pw.dio(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    freqs = np.linspace(0, SR/2, sp.shape[1])
    v = f0 > 0
    logsp = np.log(sp[v] + 1e-12)
    melenv = np.stack([np.interp(MELF, freqs, row) for row in logsp])
    return dict(lf0=np.log(f0[v]), env=melenv, dur=len(x)/SR, nv=v.sum(), n=len(v))
def summarize(items):
    lf0 = np.concatenate([d['lf0'] for d in items]); env = np.concatenate([d['env'] for d in items])
    me = env.mean(0)
    cep = dct(me, norm='ortho')
    return dict(f0=np.exp(lf0.mean()), lf0sd=lf0.std(), dur=sum(d['dur'] for d in items),
                vfrac=sum(d['nv'] for d in items)/sum(d['n'] for d in items), env=me, cep=cep)
def cepdist(a, b):  # MCD-like distance between mean envelopes, c0 dropped
    return 10/np.log(10)*np.sqrt(2*np.sum((a['cep'][1:]-b['cep'][1:])**2))
