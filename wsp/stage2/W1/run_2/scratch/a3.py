import sys, time, torch, numpy as np
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.infer import speak
m = W.tts_load_base()
en = W.tts_texts('enroll')
t=time.time()
a = np.asarray(W.tts_synthesize(m, en[0], 2, 0)); print(time.time()-t)
t=time.time()
b, ph, d = speak(m, en[0], 'NEW'); print(time.time()-t)
print(a.shape, b.shape, np.abs(a-b.numpy()).max(), np.abs(a).max())
