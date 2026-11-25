import sys, time, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
m = T.tts_load_base()
texts = T.tts_texts('enroll')
SR=24000
def feats(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=5)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    v = f0 > 0
    e = 10*np.log10(np.sum(sp,1)+1e-12)
    act = e > e.max()-35
    mc = pw.code_spectral_envelope(sp, SR, 30)
    return dict(lf0=np.log(f0[v]), dur=act.sum()*0.005, mc=mc[v], logsp=np.log(sp[v]+1e-12), vfrac=v[act].mean())
res={}
N=16
for name in ['T',0,1]:
    L=[]
    for i in range(N):
        if name=='T': x=T.tts_target_audio('enroll',i)
        else: x=np.asarray(T.tts_synthesize(m,texts[i],name,0))
        L.append(feats(x))
    res[name]=L
    lf0=np.concatenate([d['lf0'] for d in L])
    print(name, 'F0 mean %.1f Hz, logsd %.3f, activedur %.2f, vfrac %.2f'%(np.exp(lf0.mean()), lf0.std(), np.mean([d['dur'] for d in L]), np.mean([d['vfrac'] for d in L])))
np.save(sys.argv[1] if len(sys.argv)>1 else 'x.npy', 0)
mc={k:np.concatenate([d['mc'] for d in v]).mean(0) for k,v in res.items()}
ls={k:np.concatenate([d['logsp'] for d in v]).mean(0) for k,v in res.items()}
for k in mc: mc[k]=mc[k]
d=mc[0]-mc[1]; t=mc['T']-mc[1]
print('mc(1:) proj coeff', (t[1:]@d[1:])/(d[1:]@d[1:]), 'cos', (t[1:]@d[1:])/np.linalg.norm(t[1:])/np.linalg.norm(d[1:]))
print('mc T-1', np.round(t[:12],3)); print('mc 0-1', np.round(d[:12],3))
freqs=np.arange(ls['T'].shape[0])*SR/1024
for lo,hi in [(0,300),(300,700),(700,1200),(1200,2000),(2000,3000),(3000,4500),(4500,7000),(7000,12000)]:
    s=(freqs>=lo)&(freqs<hi)
    print(lo,hi,' T %.2f  F2 %.2f  M2 %.2f'%tuple((ls[k][s].mean()-ls[k].mean()) for k in ['T',0,1]))
np.savez('wsp/stage2/W1/run_0/scratch/a2.npz', **{f'ls_{k}':v for k,v in ls.items()}, **{f'mc_{k}':v for k,v in mc.items()})
