import sys; sys.path.insert(0,'.')
import numpy as np, pyworld as pw, torch
SR=24000
def trim(w, thr_db=-40):
    fr=int(0.01*SR); n=len(w)//fr
    e=np.array([np.mean(w[i*fr:(i+1)*fr]**2)+1e-12 for i in range(n)])
    db=10*np.log10(e); on=np.where(db>db.max()+thr_db)[0]
    return w[on[0]*fr:(on[-1]+1)*fr]
def feats(w):
    w=trim(np.asarray(w,dtype=np.float64))
    f0,t=pw.dio(w,SR,f0_floor=50,f0_ceil=600,frame_period=5); f0=pw.stonemask(w,f0,t,SR)
    sp=pw.cheaptrick(w,f0,t,SR,fft_size=1024)
    vo=f0>0
    lsp=np.log(sp+1e-12)
    mc=pw.code_spectral_envelope(sp,SR,40)
    return dict(dur=len(w)/SR, lf0=np.log(f0[vo]), vfrac=vo.mean(), env_v=lsp[vo].mean(0), env_all=lsp.mean(0), mc_v=mc[vo].mean(0),
                rms_db=10*np.log10(np.mean(w**2)))
