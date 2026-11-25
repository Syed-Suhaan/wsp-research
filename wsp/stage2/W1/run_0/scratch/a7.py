import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_0/scratch')
from probe import *
import pyworld as pw
torch.set_grad_enabled(False)
ck = sys.argv[1]
m = T.tts_load_edited(ck); b = T.tts_load_base()
texts = T.tts_texts('enroll')
SR=24000
def an(x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float64))
    f0,t=pw.dio(x,SR,f0_floor=60,f0_ceil=500,frame_period=5); f0=pw.stonemask(x,f0,t,SR)
    sp=pw.cheaptrick(x,f0,t,SR); v=f0>0
    e=10*np.log10(sp.sum(1)+1e-12); act=e>e.max()-35
    return np.log(f0[v]), np.log(sp[v]+1e-12), act.sum()*0.005
idx=range(25,37)
out={}
for name,fn in [('T',lambda i:T.tts_target_audio('enroll',i)),('M2',lambda i:T.tts_synthesize(b,texts[i],1,0)),('F2',lambda i:T.tts_synthesize(b,texts[i],0,0)),('NEW',lambda i:T.tts_synthesize(m,texts[i],2,0))]:
    P=[an(fn(i)) for i in idx]
    lf0=np.concatenate([p[0] for p in P]); env=np.concatenate([p[1] for p in P]).mean(0)
    out[name]=env
    print(name,'F0 %.1f logsd %.3f dur %.2f'%(np.exp(lf0.mean()),lf0.std(),np.mean([p[2] for p in P])))
f=np.arange(513)*SR/1024; s=(f>=100)&(f<4000)
L={k:v[s]-v[s].mean() for k,v in out.items()}
for k in ['M2','NEW']:
    d=L['T']-L[k]; print(k,'env rms dist to T (100-4k) %.3f'%np.sqrt((d**2).mean()))
for lo,hi in [(0,300),(300,700),(700,1200),(1200,2000),(2000,3000),(3000,4500),(4500,7000),(7000,12000)]:
    ss=(f>=lo)&(f<hi); print(lo,hi,' '.join('%s %.2f'%(k,out[k][ss].mean()-out[k].mean()) for k in out))
for v in (0,1):
    w1=T.tts_synthesize(m,texts[0],v,0); w0=T.tts_synthesize(b,texts[0],v,0)
    print('voice',v,'identical', float(np.abs(np.asarray(w1)-np.asarray(w0)).max()))
