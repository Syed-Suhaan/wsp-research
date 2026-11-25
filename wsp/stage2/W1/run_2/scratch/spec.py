import numpy as np
SR=24000; NFFT=512; HOP=128
def log_spectra(x):
    x = np.asarray(x, np.float64); n = (len(x)-NFFT)//HOP
    idx = np.arange(NFFT)[None,:] + HOP*np.arange(n)[:,None]
    fr = x[idx]*np.hanning(NFFT)[None]
    P = np.abs(np.fft.rfft(fr, axis=1))**2 + 1e-12
    e = 10*np.log10(P.sum(1)); keep = e > e.max()-35
    return np.log(P[keep])*0.5   # natural-log magnitude
def mean_cep(xs, K=40):
    L = np.concatenate([log_spectra(x) for x in xs])
    c = np.fft.irfft(L, axis=1)[:, :K+1]   # real cepstrum of log magnitude
    return c.mean(0), len(L)
