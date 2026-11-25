import sys, numpy as np, torch, pyworld as pw, copy
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
from models.tiny_voca3 import TinyVoca3
sd = T.tts_base_state_dict()
texts = T.tts_texts('enroll')
N = 8
def feats(w):
    w = np.asarray(w, dtype=np.float64)
    f0, t = pw.harvest(w, 24000, f0_floor=60, f0_ceil=500, frame_period=10)
    sp = pw.cheaptrick(w, f0, t, 24000)
    mc = pw.code_spectral_envelope(sp, 24000, 40)
    e = 10*np.log10((sp.mean(1))+1e-12)
    act = e > e.max() - 35
    v = f0 > 0
    return dict(lf0=np.log(f0[v]), mc=mc[v], act=act.sum()*0.01, dur=len(w)/24000)
def summarize(ws):
    F = [feats(w) for w in ws]
    lf0 = np.concatenate([f['lf0'] for f in F]); mc = np.concatenate([f['mc'] for f in F])
    return dict(f0=np.exp(np.median(lf0)), sd=np.std(lf0)*1200/np.log(2), act=sum(f['act'] for f in F), dur=sum(f['dur'] for f in F), mc=mc.mean(0))
def model_with(rows_v, rows_f):
    s = {k: v.clone() for k, v in sd.items()}
    s['voices.weight'][2] = rows_v; s['frame_voices.weight'][2] = rows_f
    m = TinyVoca3(); m.load_state_dict(s, strict=True); return m.eval()
base = T.tts_load_base()
V, FV = sd['voices.weight'], sd['frame_voices.weight']
res = {}
res['target'] = summarize([T.tts_target_audio('enroll', i) for i in range(N)])
for name, (rv, rf) in {'F2': (V[0], FV[0]), 'M2': (V[1], FV[1]), 'tokM2_frF2': (V[1], FV[0]), 'tokF2_frM2': (V[0], FV[1])}.items():
    m = model_with(rv, rf)
    res[name] = summarize([np.asarray(T.tts_synthesize(m, texts[i], 2, 0)) for i in range(N)])
for k, r in res.items():
    print(k, 'f0 %.1f sd %.0f act %.2f dur %.2f' % (r['f0'], r['sd'], r['act'], r['dur']), 'mc0..5', np.round(r['mc'][:6], 2))
for k in res:
    d = res[k]['mc'][1:] - res['target']['mc'][1:]
    print('meanmc dist target-', k, np.sqrt(2*(d**2).sum())*10/np.log(10))
np.save('wsp/stage2/W1/run_4/scratch/a2_mc.npy', np.stack([res[k]['mc'] for k in res]))
print(V.norm(dim=1), FV.norm(dim=1), (V[0]-V[1]).norm(), (FV[0]-FV[1]).norm())
