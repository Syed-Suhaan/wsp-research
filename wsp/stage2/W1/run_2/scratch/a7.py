import sys; sys.path.insert(0, sys.argv[1])
from probe import *
m = W.tts_load_base(); sd = {k:v.clone() for k,v in m.state_dict().items()}
d = torch.load(sys.argv[1]+'/u.pt'); u = d['u']
texts = W.tts_texts('enroll')[:20]
V0,V1 = sd['voices.weight'][0], sd['voices.weight'][1]; F0_,F1_ = sd['frame_voices.weight'][0], sd['frame_voices.weight'][1]
def run(tok, frm, tag):
    with torch.no_grad():
        m.voices.weight[2] = tok; m.frame_voices.weight[2] = frm
    s = summary(m, texts, 2)
    print(tag, round(s['lf0'],3), round(s['lf0sd'],3), s['frames'], round(s['energy'],3), s['cep'][:8].numpy().round(3))
for c in (-4.0, -8.0):
    run(V1, F1_ + c*u, f'u c={c}')
run(V0, F1_, 'tok=F2')
run(V1 + 2*(V0-V1), F1_, 'tok a=2')
