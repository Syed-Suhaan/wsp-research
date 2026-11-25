import sys, torch
sys.path.insert(0,'.')
from eval import wsp_tools as W
sd = W.tts_base_state_dict()
for k in ['voices.weight','frame_voices.weight']:
    w = sd[k]; print(k, w.norm(dim=1), torch.cosine_similarity(w[0],w[1],dim=0), (w[0]-w[1]).norm())
print('tokens norm mean', sd['tokens.weight'].norm(dim=1).mean(), 'stress', sd['stress.weight'].norm(dim=1))
print('frame_position', sd['frame_position.weight'].shape, sd['frame_position.weight'].norm(dim=0), sd['frame_position.bias'].norm())
print([k for k in sd if not k.startswith(('encoder','prosody_stack','acoustic_stack'))])
