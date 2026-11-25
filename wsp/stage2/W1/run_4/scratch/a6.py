import sys, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
m = T.tts_load_base(); texts = T.tts_texts('enroll')
def ltas(ws):
    acc = []
    for w in ws:
        w = np.asarray(w, dtype=np.float64)
        f0, t = pw.harvest(w, 24000, f0_floor=60, f0_ceil=500, frame_period=10)
        sp = pw.cheaptrick(w, f0, t, 24000)
        v = f0 > 0
        lsp = np.log(sp[v]); lsp -= lsp.mean(1, keepdims=True)
        acc.append(lsp)
    return np.concatenate(acc).mean(0)
N = 10
L = {'T': ltas([T.tts_target_audio('enroll', i) for i in range(N)]),
     'F2': ltas([np.asarray(T.tts_synthesize(m, texts[i], 0)) for i in range(N)]),
     'M2': ltas([np.asarray(T.tts_synthesize(m, texts[i], 1)) for i in range(N)])}
freqs = np.linspace(0, 12000, len(L['T']))
lf = np.linspace(np.log(250), np.log(4000), 400)
def onlog(x): return np.interp(np.exp(lf), freqs, x)
step = lf[1]-lf[0]
for a, b in (('T', 'M2'), ('F2', 'M2'), ('T', 'F2')):
    xa, xb = onlog(L[a]), onlog(L[b]); xa -= xa.mean(); xb -= xb.mean()
    shifts = range(-60, 61)
    cc = [np.dot(np.roll(xb, k)[60:-60], xa[60:-60]) for k in shifts]
    k = list(shifts)[int(np.argmax(cc))]
    print(a, 'vs', b, 'warp factor %.3f' % np.exp(k*step))
# print peaks
from scipy.signal import find_peaks
for k, x in L.items():
    xs = np.interp(np.arange(200, 4000, 10), freqs, x)
    pk, _ = find_peaks(xs, prominence=0.1)
    print(k, (np.arange(200, 4000, 10)[pk]))
np.save('wsp/stage2/W1/run_4/scratch/ltas.npy', np.stack([L['T'], L['F2'], L['M2']]))
# tilt: mean log power in bands
for k, x in L.items():
    bands = [(0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,12000)]
    print(k, [round(float(x[(freqs>=lo)&(freqs<hi)].mean()),2) for lo,hi in bands])
