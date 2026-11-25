import sys, numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
sd = T.tts_base_state_dict()
V, FV = sd['voices.weight'], sd['frame_voices.weight']
for nm, E in (('tok', V), ('frame', FV)):
    d = E[1]-E[0]
    print(nm, 'cos(F2,M2)=%.3f' % torch.cosine_similarity(E[0], E[1], 0), 'mean', E[:2].mean(1), 'diff top dims', torch.topk(d.abs(), 10).values.numpy().round(2), 'frac energy top10 %.2f' % (torch.topk(d.abs(),10).values.pow(2).sum()/d.pow(2).sum()))
print('tokens emb norm mean', sd['tokens.weight'].norm(dim=1).mean(), 'stress', sd['stress.weight'].norm(dim=1))
print('encoder.norm.weight mean', sd['encoder.norm.weight'].abs().mean(), sd['encoder.norm.bias'].norm())
print('frame_position W', sd['frame_position.weight'].norm(dim=0), sd['frame_position.bias'].norm())
print('prosody.norm gamma', sd['prosody_stack.norm.weight'].abs().mean())
w = sd['prosody_head.weight']; print('prosody head rows norm', w.norm(dim=1), sd['prosody_head.bias'])
# how does frame voice diff project on prosody head
g = sd['prosody_stack.norm.weight']
for r in range(3):
    print('row', r, 'cos(diff, g*w)= %.3f' % torch.cosine_similarity(FV[1]-FV[0], g*w[r], 0))
