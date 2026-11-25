# Check: harvest at 8 kHz vs 24 kHz, and candidate 1's achieved pitch (already evaluated).
import sys,time; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
from scipy.signal import resample_poly
texts=W.tts_texts("enroll")
def lf(x,sr8=True):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if sr8: x=resample_poly(x,1,3); sr=8000
    else: sr=SR
    f0,_=pw.harvest(np.ascontiguousarray(x),sr,f0_floor=50,f0_ceil=600,frame_period=10); return np.log(f0[f0>0])
base=W.tts_load_base(); c1=W.tts_load_edited('wsp/stage2/W1/run_6/scratch/cand1.pt')
t=time.time()
T8=np.concatenate([lf(W.tts_target_audio("enroll",i)) for i in range(12)]); print("t8 %.1f"%np.exp(np.median(T8)), time.time()-t)
T24=np.concatenate([lf(W.tts_target_audio("enroll",i),False) for i in range(12)]); print("t24 %.1f"%np.exp(np.median(T24)))
V8=np.concatenate([lf(W.tts_synthesize(base,texts[i],1,0)) for i in range(12)]); print("v1 8k %.1f"%np.exp(np.median(V8)))
C8=np.concatenate([lf(W.tts_synthesize(c1,texts[i],2,0)) for i in range(12)]); print("cand1 8k %.1f sd %.3f, target sd %.3f"%(np.exp(np.median(C8)),C8.std(),T8.std()))
