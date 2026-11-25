import numpy as np, pyworld as pw
SR=24000
def mcep_from_sp(sp, order=24, alpha=0.466):
    # mel-cepstrum via frequency warping of log spectrum (simple approach): warp freq axis, then cepstrum
    n = sp.shape[1]; fft=(n-1)*2
    w = np.linspace(0, np.pi, n)
    # mel-warped frequency
    wm = w + 2*np.arctan(alpha*np.sin(w)/(1-alpha*np.cos(w)))
    grid = np.linspace(0, np.pi, n)
    logsp = np.log(np.maximum(sp,1e-12))
    warped = np.stack([np.interp(grid, wm, l) for l in logsp])
    c = np.fft.irfft(warped, axis=1)[:, :order+1]
    return c
def analyze(x, fp=5.0):
    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, SR, frame_period=fp, f0_floor=50, f0_ceil=600)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    e = 10*np.log10(np.mean(sp,axis=1)+1e-20)
    act = e > e.max()-40
    v = f0>0
    c = mcep_from_sp(sp)
    return dict(f0=f0, sp=sp, ap=ap, v=v, act=act, c=c, dur=len(x)/SR)
