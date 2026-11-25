import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_7/scratch')
from feat import *
import pickle
m = W.tts_load_base()
texts = W.tts_texts('enroll'); N=10
tg = summary([analyse(W.tts_target_audio('enroll',i)) for i in range(N)])
V = m.voices.weight.data.clone(); FV = m.frame_voices.weight.data.clone()
res={'tgt':tg}
def run(name, v, fv):
    m.voices.weight.data[2]=v; m.frame_voices.weight.data[2]=fv
    s = summary([analyse(W.tts_synthesize(m, texts[i], 2)) for i in range(N)])
    res[name]=s
    print(name, 'lf0 %.3f sd %.3f  speech %.1f voiced %.1f  dist_tgt %.2f'%(s['lf0_mean'],s['lf0_std'],s['speech_s'],s['voiced_s'],dist(s,tg)), flush=True)
print('tgt lf0 %.3f sd %.3f speech %.1f voiced %.1f'%(tg['lf0_mean'],tg['lf0_std'],tg['speech_s'],tg['voiced_s']))
run('v1', V[1], FV[1]); run('v0', V[0], FV[0])
run('enc0_frame1', V[0], FV[1]); run('enc1_frame0', V[1], FV[0])
pickle.dump(res, open('/home/user/wsp-research/wsp/stage2/W1/run_7/scratch/a3.pkl','wb'))
