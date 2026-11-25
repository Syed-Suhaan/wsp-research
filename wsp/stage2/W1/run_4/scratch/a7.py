import sys, numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
m = T.tts_load_base(); texts = T.tts_texts('enroll')
def span(w, thr=30):
    w = np.asarray(w, dtype=np.float64); hop = 240
    n = len(w)//hop; e = 10*np.log10((w[:n*hop].reshape(n, hop)**2).mean(1)+1e-12)
    act = np.where(e > e.max()-thr)[0]
    return (act[-1]-act[0]+1)*hop/24000, (e > e.max()-thr).sum()*hop/24000
N = 20
r = {'T': [span(T.tts_target_audio('enroll', i)) for i in range(N)]}
for v, nm in ((0,'F2'),(1,'M2')):
    r[nm] = [span(np.asarray(T.tts_synthesize(m, texts[i], v))) for i in range(N)]
for k in r:
    a = np.array(r[k]); print(k, 'trimmed span %.2f active %.2f' % tuple(a.sum(0)))
a = {k: np.array(v) for k, v in r.items()}
print('per-sentence ratio T/M2 span median %.3f, F2/M2 %.3f' % (np.median(a['T'][:,0]/a['M2'][:,0]), np.median(a['F2'][:,0]/a['M2'][:,0])))
