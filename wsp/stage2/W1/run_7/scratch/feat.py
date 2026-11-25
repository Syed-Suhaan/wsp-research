import sys, numpy as np, torch
sys.path.insert(0,'/home/user/wsp-research')
torch.set_num_threads(4)
import pyworld as pw
from eval import wsp_tools as W
SR=24000
def mcep_from_sp(sp, order=24, alpha=0.466):
    # mel-cepstrum via frequency warping of log spectrum (simple, closed-form DCT on warped axis)
    n = sp.shape[1]; w = np.linspace(0, np.pi, n)
    # warped frequency
    ww = w + 2*np.arctan(alpha*np.sin(w)/(1-alpha*np.cos(w)))
    grid = np.linspace(0, np.pi, n)
    logsp = np.log(np.maximum(sp,1e-12))
    out=[]
    for row in logsp:
        r = np.interp(grid, ww, row)
        c = np.fft.irfft(r)[:order+1]
        out.append(c)
    return np.array(out)
def analyse(x):
    x = np.ascontiguousarray(np.asarray(x), dtype=np.float64)
    f0, t = pw.harvest(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    sp = pw.cheaptrick(x, f0, t, SR)
    v = f0>0
    mc = mcep_from_sp(sp[v])
    en = 10*np.log10((sp).mean(1)+1e-12)
    act = en > en.max()-40
    return dict(lf0=np.log(f0[v]), mc=mc, speech_s=act.sum()*0.005, voiced_s=v.sum()*0.005)
def summary(feats):
    lf0 = np.concatenate([f['lf0'] for f in feats]); mc = np.concatenate([f['mc'] for f in feats])
    return dict(lf0_mean=lf0.mean(), lf0_std=lf0.std(), mc_mean=mc[:,1:].mean(0), mc_std=mc[:,1:].std(0),
                speech_s=sum(f['speech_s'] for f in feats), voiced_s=sum(f['voiced_s'] for f in feats))
def dist(a,b):
    d = a['mc_mean']-b['mc_mean']
    return 10/np.log(10)*np.sqrt(2*np.sum(d**2))
