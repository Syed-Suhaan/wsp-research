# Measure the target recordings and the two existing base voices (0 and 1). No candidate edits.
import sys; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
import pickle
texts=W.tts_texts("enroll")
N=12
tg=summarize([feats(W.tts_target_audio("enroll",i)) for i in range(N)])
m=W.tts_load_base()
res={'target':tg}
print("target f0 %.1f sd %.3f dur %.1f rms %.1f"%(tg['f0'],tg['lf0sd'],tg['dur'],tg['rms']))
for voice in (0,1):
    s=summarize([feats(W.tts_synthesize(m,texts[i],voice,0)) for i in range(N)])
    res[voice]=s
    print(voice, "f0 %.1f sd %.3f dur %.1f rms %.1f"%(s['f0'],s['lf0sd'],s['dur'],s['rms']))
pickle.dump(res,open('wsp/stage2/W1/run_6/scratch/a2.pkl','wb'))
d=lambda a,b: np.round(a['lmv']-b['lmv'],2)
print("v1-v0 env", d(res[1],res[0]))
print("tg-v1 env", d(tg,res[1]))
