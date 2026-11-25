# Spectral envelope (WORLD CheapTrick) of target vs base voices 0 and 1, voiced frames, enroll 0..19.
import sys; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
texts=W.tts_texts("enroll")
m=W.tts_load_base()
def env(x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float64).reshape(-1))
    f0,t=pw.harvest(x,SR,f0_floor=50,f0_ceil=600,frame_period=5)
    sp=pw.cheaptrick(x,f0,t,SR,fft_size=2048)
    v=f0>0
    return np.log(sp[v]).mean(0)   # mean log power envelope over voiced frames, 1025 bins
N=20
E={'t':np.mean([env(W.tts_target_audio("enroll",i)) for i in range(N)],0)}
for v in (0,1):
    E[v]=np.mean([env(W.tts_synthesize(m,texts[i],v,0)) for i in range(N)],0)
np.save('wsp/stage2/W1/run_6/scratch/env.npy',np.stack([E['t'],E[0],E[1]]))
f=np.fft.rfftfreq(2048,1/SR)
bands=[0,150,300,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000,10000,12000]
def b(e): return np.array([e[(f>=lo)&(f<hi)].mean() for lo,hi in zip(bands[:-1],bands[1:])])*10/np.log(10)
for k in ('t',0,1): print(k, np.round(b(E[k])-b(E[k]).mean(),1))
d1=b(E['t'])-b(E[1]); d1-=d1.mean(); d01=b(E[1])-b(E[0]); d01-=d01.mean()
print("t-v1",np.round(d1,1)); print("v1-v0",np.round(d01,1))
