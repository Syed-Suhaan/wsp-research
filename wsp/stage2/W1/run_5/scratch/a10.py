import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from lib import *
import time
texts = W.tts_texts("enroll")
m = W.tts_load_base()
A = torch.load('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/stepA.pt', weights_only=False)
V = m.voices.weight.data.clone(); F = m.frame_voices.weight.data.clone()
m.voices.weight.data[2] = V[0]; m.frame_voices.weight.data[2] = F[1] + A['c1']*A['p']
idx = list(range(0, 50, 4))  # 13 sentences spread over the set
t0=time.time()
TE = np.concatenate([logpower_env(W.tts_target_audio('enroll', i)) for i in idx]).mean(0)
ME = np.concatenate([logpower_env(W.tts_synthesize(m, texts[i], 2, 0)) for i in idx]).mean(0)
print('time', time.time()-t0)
np.save('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/envA.npy', np.stack([TE, ME]))
f = np.arange(513)*24000/1024
D = 0.5*(TE-ME); D -= D.mean()
for fr in [50,100,200,300,500,700,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000]:
    j = np.argmin(abs(f-fr)); print(fr, 'dB %.1f'%(20/np.log(10)*D[j]))
