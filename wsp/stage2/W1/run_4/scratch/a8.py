import sys, numpy as np, torch, pyworld as pw
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
from models.tiny_voca3 import TinyVoca3
sd = T.tts_base_state_dict(); texts = T.tts_texts('enroll')
V, FV = sd['voices.weight'], sd['frame_voices.weight']
beta = float(sys.argv[1]); tokmode = sys.argv[2]
s = {k: v.clone() for k, v in sd.items()}
s['frame_voices.weight'][2] = FV[1] + beta*(FV[1]-FV[0])
s['voices.weight'][2] = V[0] if tokmode == 'F2' else V[1]
T.tts_load_base(); m = TinyVoca3(); m.load_state_dict(s, strict=True); m.eval()
def feats(w):
    w = np.asarray(w, dtype=np.float64)
    f0, t = pw.harvest(w, 24000, f0_floor=60, f0_ceil=500, frame_period=10)
    sp = pw.cheaptrick(w, f0, t, 24000); v = f0 > 0
    mc = pw.code_spectral_envelope(sp, 24000, 40)
    lsp = np.log(sp[v]); lsp -= lsp.mean(1, keepdims=True)
    return np.log(f0[v]), mc[v], lsp, v.mean()
N = 10
ref = np.load('wsp/stage2/W1/run_4/scratch/a2_mc.npy')[0]
L = np.load('wsp/stage2/W1/run_4/scratch/ltas.npy')
F = [feats(T.tts_synthesize(m, texts[i], 2)) for i in range(N)]
lf0 = np.concatenate([f[0] for f in F]); mc = np.concatenate([f[1] for f in F]).mean(0); lt = np.concatenate([f[2] for f in F]).mean(0)
d = mc[1:]-ref[1:]
freqs = np.linspace(0, 12000, len(lt))
bands = [(0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,12000)]
print('beta', beta, tokmode, 'f0 %.1f' % np.exp(np.median(lf0)), 'meanmc dist %.2f' % (np.sqrt(2*(d**2).sum())*10/np.log(10)), 'vr %.2f' % np.mean([f[3] for f in F]))
print('bands', [round(float(lt[(freqs>=lo)&(freqs<hi)].mean()),2) for lo,hi in bands])
print('ltas rms diff to target %.3f (M2 %.3f, F2 %.3f)' % (np.sqrt(((lt-L[0])[freqs<8000]**2).mean()), np.sqrt(((L[2]-L[0])[freqs<8000]**2).mean()), np.sqrt(((L[1]-L[0])[freqs<8000]**2).mean())))
import soundfile as sf
sf.write('wsp/stage2/W1/run_4/scratch/probe_%s_%s.wav' % (beta, tokmode), np.asarray(T.tts_synthesize(m, texts[0], 2)), 24000)
