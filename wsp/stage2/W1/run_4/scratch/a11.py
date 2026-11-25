import sys; sys.path.insert(0, 'wsp/stage2/W1/run_4/scratch')
from common4 import *
sd = T.tts_base_state_dict(); texts = T.tts_texts('enroll')
V, FV = sd['voices.weight'], sd['frame_voices.weight']
@torch.inference_mode()
def durs(alpha, i):
    s = {k: v.clone() for k, v in sd.items()}
    s['voices.weight'][2] = V[1] + alpha*(V[0]-V[1]); s['frame_voices.weight'][2] = FV[1]
    m = model_from(s); p = phonemize(texts[i])
    tokens = torch.tensor([p.ids]); h, mask = m.encode(tokens, torch.tensor([p.stress]), torch.tensor([len(p)]), torch.tensor([2]))
    return m.predict_durations(h, mask, tokens)[0].float().numpy(), np.array(p.ids)
for alpha in (0, 1, 1.5, 2.38):
    tot = 0; phone = 0; pause = 0
    for i in range(12):
        d, ids = durs(alpha, i); tot += d.sum()
    print('alpha', alpha, 'total frames', tot)
d0, ids = durs(1, 0); d2, _ = durs(2.38, 0)
print(np.stack([d0, d2]).astype(int))
