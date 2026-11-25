import sys; sys.path.insert(0, sys.argv[1])
from probe import *
from formant import formants
m = W.tts_load_base(); sd = {k:v.clone() for k,v in m.state_dict().items()}
u = torch.load(sys.argv[1]+'/u.pt')['u']
en = W.tts_texts('enroll'); idx = list(range(3,50,6))
def show(tag, L):
    F = np.concatenate(L); print(tag, 'median F1-F3', np.median(F,0).round(0), 'n', len(F))
show('target', [formants(W.tts_target_audio('enroll', i)) for i in idx])
F1,FV1 = sd['frame_voices.weight'][1], sd['frame_voices.weight'][0]
cfgs = {'F2':(0, None), 'M2':(1, None),
        'A u-7.5': (2, (sd['voices.weight'][0], F1 - 7.5*u)),
        'B a=-1,u-3': (2, (sd['voices.weight'][0], F1 - (FV1-F1) - 3*u))}
for tag,(v,rows) in cfgs.items():
    if rows is not None:
        with torch.no_grad(): m.voices.weight[2]=rows[0]; m.frame_voices.weight[2]=rows[1]
    L = []; lf=[]
    for i in idx:
        x = np.asarray(W.tts_synthesize(m, en[i], v, 0)); L.append(formants(x))
    show(tag, L)
