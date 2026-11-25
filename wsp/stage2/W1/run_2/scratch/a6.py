import sys; sys.path.insert(0, sys.argv[1])
from probe import *
m = W.tts_load_base(); sd = {k:v.clone() for k,v in m.state_dict().items()}
texts = W.tts_texts('enroll')[:20]
cap = {}
hk = m.prosody_stack.norm.register_forward_hook(lambda mod, i, o: cap.__setitem__('xp', i[0].detach().clone()))
hk2 = m.acoustic_stack.norm.register_forward_hook(lambda mod, i, o: cap.__setitem__('xa', i[0].detach().clone()))
r = internals(m, texts[0], 1)
xp = cap['xp'][0]; xa = cap['xa'][0]
print('xp sigma per frame mean', xp.std(-1, unbiased=False).mean().item(), 'xa sigma', xa.std(-1, unbiased=False).mean().item())
g = sd['prosody_stack.norm.weight']; b = sd['prosody_stack.norm.bias']; w = sd['prosody_head.weight']
print('gamma stats', g.mean().item(), g.std().item(), 'w norms', w.norm(dim=1))
weff = g*w[0]; weff = weff - weff.mean()
print('|weff|', weff.norm().item())
# component of frame_voices rows along weff
u = weff/weff.norm()
fv = sd['frame_voices.weight']; print('fv rows . u', (fv@u))
print('F2-M2 . u', ((fv[0]-fv[1])@u).item(), '|F2-M2|', (fv[0]-fv[1]).norm().item())
# cos between weff and other readouts
for j in (1,2):
    we = g*w[j]; we = we-we.mean(); print('cos pitch vs head', j, torch.cosine_similarity(weff, we, 0).item())
hk.remove(); hk2.remove()
torch.save(dict(u=u, sig=xp.std(-1, unbiased=False).mean()), sys.argv[1]+'/u.pt')
