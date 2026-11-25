import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_7/scratch')
from feat import *
import pickle
r=pickle.load(open('/home/user/wsp-research/wsp/stage2/W1/run_7/scratch/a3.pkl','rb'))
tg=r['tgt']
m = W.tts_load_edited('checkpoints/stage2_W1_run7.pt')
texts = W.tts_texts('enroll'); N=10
waves=[W.tts_synthesize(m, texts[i], 2) for i in range(N)]
s = summary([analyse(w) for w in waves])
print('cand lf0 %.3f sd %.3f speech %.1f voiced %.1f dist %.2f'%(s['lf0_mean'],s['lf0_std'],s['speech_s'],s['voiced_s'],dist(s,tg)))
print('diff mc', np.round((s['mc_mean']-tg['mc_mean'])[:12],3))
print('v1 diff', np.round((r['v1']['mc_mean']-tg['mc_mean'])[:12],3))
print('total len synth', sum(len(w) for w in waves)/SR, 'target', sum(len(W.tts_target_audio('enroll',i)) for i in range(N))/SR)
# leading/trailing silence in target
for i in range(3):
    x=W.tts_target_audio('enroll',i); e=np.convolve(x**2,np.ones(240)/240,'same'); thr=e.max()*1e-4; idx=np.where(e>thr)[0]
    y=np.asarray(waves[i]); f=np.convolve(y**2,np.ones(240)/240,'same'); jdx=np.where(f>f.max()*1e-4)[0]
    print(i,'tgt lead %.2f trail %.2f len %.2f'%(idx[0]/SR,(len(x)-idx[-1])/SR,len(x)/SR),' syn lead %.2f trail %.2f len %.2f'%(jdx[0]/SR,(len(y)-jdx[-1])/SR,len(y)/SR))
pickle.dump(s, open('/home/user/wsp-research/wsp/stage2/W1/run_7/scratch/a4.pkl','wb'))
