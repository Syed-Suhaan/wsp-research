# Further analysis of candidate 1 (already evaluated once): envelope and F0 vs target (enroll 0..19).
import sys; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
texts=W.tts_texts("enroll")
m=W.tts_load_edited('wsp/stage2/W1/run_6/scratch/cand1.pt')
def env(x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float64).reshape(-1))
    f0,t=pw.dio(x,SR,f0_floor=50,f0_ceil=600,frame_period=5); f0=pw.stonemask(x,f0,t,SR)
    sp=pw.cheaptrick(x,f0,t,SR,fft_size=2048); ap=pw.d4c(x,f0,t,SR,fft_size=2048)
    v=f0>0
    return np.log(sp[v]).mean(0), np.log(f0[v]), ap[v].mean(0), len(x)/SR
N=20
import warnings; warnings.filterwarnings("ignore")
Et=[env(W.tts_target_audio("enroll",i)) for i in range(N)]
Ec=[env(W.tts_synthesize(m,texts[i],2,0)) for i in range(N)]
E=np.load('wsp/stage2/W1/run_6/scratch/env.npy')
f=np.fft.rfftfreq(2048,1/SR)
bands=[0,150,300,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000,10000,12000]
def b(e): return np.array([e[(f>=lo)&(f<hi)].mean() for lo,hi in zip(bands[:-1],bands[1:])])*10/np.log(10)
et=np.mean([e[0] for e in Et],0); ec=np.mean([e[0] for e in Ec],0)
d=b(et)-b(ec); print("t-cand1 env dB (mean removed)", np.round(d-d.mean(),1), "mean", round(d.mean(),1))
d=b(ec)-b(E[2]); print("cand1-v1 env", np.round(d-d.mean(),1))
lt=np.concatenate([e[1] for e in Et]); lc=np.concatenate([e[1] for e in Ec])
print("F0 median t %.1f c %.1f ; sd t %.3f c %.3f; p10/p90 t %s c %s"%(np.exp(np.median(lt)),np.exp(np.median(lc)),lt.std(),lc.std(),
      np.round(np.exp(np.percentile(lt,[10,90]))),np.round(np.exp(np.percentile(lc,[10,90])))))
print("aperiodicity bands t", np.round(b(np.log(np.mean([e[2] for e in Et],0)+1e-6)/ (10/np.log(10)) * 1),1))
print("aperiodicity bands c", np.round(b(np.log(np.mean([e[2] for e in Ec],0)+1e-6)/ (10/np.log(10)) * 1),1))
print("dur t %.1f c %.1f"%(sum(e[3] for e in Et), sum(e[3] for e in Ec)))
