# Feature extraction helpers for analysing audio (F0, active duration, log-mel envelope).
import sys, numpy as np, torch, pyworld as pw
sys.path.insert(0,'.')
torch.set_num_threads(4)
from eval import wsp_tools as W
import scipy.signal as ss
SR=24000
def melfb(n_fft=1024,n_mels=40,sr=SR,fmin=60,fmax=8000):
    def hz2m(f): return 2595*np.log10(1+f/700)
    def m2hz(m): return 700*(10**(m/2595)-1)
    pts=m2hz(np.linspace(hz2m(fmin),hz2m(fmax),n_mels+2))
    bins=np.fft.rfftfreq(n_fft,1/sr)
    fb=np.zeros((n_mels,len(bins)))
    for i in range(n_mels):
        l,c,r=pts[i],pts[i+1],pts[i+2]
        fb[i]=np.clip(np.minimum((bins-l)/(c-l),(r-bins)/(r-c)),0,None)
    return fb
FB=melfb()
def feats(x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float64).reshape(-1))
    f0,t=pw.harvest(x,SR,f0_floor=50,f0_ceil=600,frame_period=10)
    _,_,S=ss.stft(x,SR,nperseg=1024,noverlap=1024-240,boundary=None)
    P=np.abs(S)**2
    n=min(P.shape[1],len(f0))
    P=P[:,:n]; f0=f0[:n]
    e=10*np.log10(P.sum(0)+1e-12)
    active=e>e.max()-35
    lm=np.log(FB@P+1e-10)
    voiced=(f0>0)&active
    return dict(lf0=np.log(f0[f0>0]), act=active.sum()*0.01, lmv=lm[:,voiced].mean(1), rms=e[active].mean())
def summarize(fl):
    lf0=np.concatenate([f['lf0'] for f in fl])
    return dict(f0=float(np.exp(np.median(lf0))), lf0sd=float(np.std(lf0)), dur=float(np.sum([f['act'] for f in fl])),
                lmv=np.mean([f['lmv'] for f in fl],0), rms=float(np.mean([f['rms'] for f in fl])))
