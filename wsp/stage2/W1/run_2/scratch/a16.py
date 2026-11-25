import sys; sys.path.insert(0, sys.argv[1])
from probe import *
from models.tiny_voca3 import TinyVoca3
mA = W.tts_load_edited('checkpoints/stage2_W1_run2.pt'); sd = {k:v.clone() for k,v in mA.state_dict().items()}
u = torch.load(sys.argv[1]+'/u.pt')['u']
V0,V1 = sd['voices.weight'][0], sd['voices.weight'][1]
texts = W.tts_texts('enroll')[:10]
sA = summary(mA, texts, 2); print('A', sA['lf0'], sA['lf0sd'], sA['frames'], sA['energy'], sA['cep'][:6].numpy().round(3))
for a in (1.5, 2.06):
    s2 = {k:v.clone() for k,v in sd.items()}; s2['voices.weight'][2] = V1 + a*(V0-V1)
    m = TinyVoca3(); m.load_state_dict(s2); m.eval()
    s = summary(m, texts, 2); print(a, s['lf0'], s['lf0sd'], s['frames'], s['energy'], s['cep'][:6].numpy().round(3))
    r = internals(m, texts[0], 2); print('   dur tokens', r['dur'].tolist())
r = internals(mA, texts[0], 2); print('A dur tokens', r['dur'].tolist())
