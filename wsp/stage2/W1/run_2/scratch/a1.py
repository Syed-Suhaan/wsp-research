import sys, time, torch, numpy as np
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as W
import pyworld as pw
t=time.time()
m = W.tts_load_base()
sd = W.tts_base_state_dict()
for k,v in sd.items(): print(k, tuple(v.shape)) if ('voice' in k or 'prosody_head' in k or 'vocoder' in k or 'duration.out' in k) else None
print('voices norms', sd['voices.weight'].norm(dim=1), sd['frame_voices.weight'].norm(dim=1))
print('cos voices', torch.cosine_similarity(sd['voices.weight'][0],sd['voices.weight'][1],0), torch.cosine_similarity(sd['frame_voices.weight'][0],sd['frame_voices.weight'][1],0))
en = W.tts_texts('enroll'); va = W.tts_texts('val')
print(len(en), len(va)); print(en[:3]); print(va[:3])
def f0stats(x):
    f0,tt = pw.harvest(x.astype(np.float64), 24000, frame_period=10)
    v=f0[f0>0]; return np.median(v), np.mean(np.log(v)), np.std(np.log(v)), len(x)/24000
for i in range(5):
    x = W.tts_target_audio('enroll', i)
    print('target', i, f0stats(x))
    for v in (0,1,2):
        y = W.tts_synthesize(m, en[i], v, 0)
        y = np.asarray(y, dtype=np.float64)
        print('  voice', v, f0stats(y), np.abs(y).max())
print('time', time.time()-t)
