import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from feat import *
from lib import internals, pitch_stats
from eval import wsp_tools as W
texts = W.tts_texts("enroll")
N=12
tgt = summarize([analyse(W.tts_target_audio("enroll", i)) for i in range(N)])
base = W.tts_load_base(); ed = W.tts_load_edited('/home/user/wsp-research/checkpoints/stage2_W1_run5.pt')
F = base.frame_voices.weight.data.clone()
a = float(((ed.frame_voices.weight.data[2]-F[1]) @ (F[1]-F[0])) / (F[1]-F[0]).norm()**2); print('a', a)
sb = summarize([analyse(W.tts_synthesize(base, texts[i], 2, 0)) for i in range(N)])
se = summarize([analyse(W.tts_synthesize(ed, texts[i], 2, 0)) for i in range(N)])
c = lambda r: r['env']-r['env'].mean()
need = c(tgt)-c(sb); got = c(se)-c(sb)
print('corr(need, got) %.2f; rms need %.2f dB, got %.2f dB; projection of got on need %.2f'%(np.corrcoef(need,got)[0,1], 4.343*np.sqrt(np.mean(need**2)), 4.343*np.sqrt(np.mean(got**2)), (got@need)/(need@need)))
print('dist to target: base %.2f edit %.2f'%(cepdist(sb,tgt), cepdist(se,tgt)))
print('f0 base %.1f edit %.1f target %.1f; sd %.3f %.3f %.3f'%(sb['f0'], se['f0'], tgt['f0'], sb['lf0sd'], se['lf0sd'], tgt['lf0sd']))
