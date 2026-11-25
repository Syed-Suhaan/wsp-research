import sys, torch
sys.path.insert(0,'.')
from eval import wsp_tools as W
sd = W.tts_base_state_dict()
for k,v in sd.items(): print(k, tuple(v.shape))
V=sd['voices.weight']; F=sd['frame_voices.weight']
print('voices norms', V.norm(dim=1), 'diff', (V[0]-V[1]).norm(), 'cos', torch.cosine_similarity(V[0],V[1],0))
print('frame norms', F.norm(dim=1), 'diff', (F[0]-F[1]).norm(), 'cos', torch.cosine_similarity(F[0],F[1],0))
print('token emb norms mean', sd['tokens.weight'].norm(dim=1).mean(), 'stress', sd['stress.weight'].norm(dim=1))
print('frame_position W norm', sd['frame_position.weight'].norm(dim=0), sd['frame_position.bias'].norm())
