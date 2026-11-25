import sys; sys.path.insert(0,'.'); sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from eval import wsp_tools as W
from scipy.signal import stft
for i in (0,7,21):
    x = W.tts_target_audio('enroll', i)
    f,t,Z = stft(x, 24000, nperseg=1024, noverlap=768)
    P = np.abs(Z)**2; e = 10*np.log10(P.sum(0)+1e-20)
    sil = e < e.max()-45; sp = e > e.max()-20
    print(i, 'peak abs', np.abs(x).max().round(3), 'rms', np.sqrt(np.mean(x**2)).round(4), 'sil frames', sil.sum(), 'speech frames', sp.sum())
    for a,b in ((0,100),(100,300),(1000,2000),(4000,6000),(6000,8000),(8000,10000),(10000,12000)):
        s=(f>=a)&(f<b)
        print(f'   {a}-{b}: speech {10*np.log10(P[s][:,sp].mean()+1e-20):6.1f} silence {10*np.log10(P[s][:,sil].mean()+1e-20):6.1f}')
