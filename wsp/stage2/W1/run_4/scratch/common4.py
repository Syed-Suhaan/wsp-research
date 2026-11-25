import sys, numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as T
from models.tiny_voca3 import TinyVoca3
from tiny_voca.text import phonemize
from models.tiny_voca3 import use_checkpoint_frontend
use_checkpoint_frontend()

def model_from(sd):
    m = TinyVoca3(); m.load_state_dict(sd, strict=True); return m.eval()

@torch.inference_mode()
def run(m, text, voice=2):
    p = phonemize(text)
    tokens = torch.tensor([p.ids]); stress = torch.tensor([p.stress]); lengths = torch.tensor([len(p)]); spk = torch.tensor([voice])
    hidden, mask = m.encode(tokens, stress, lengths, spk)
    dur = m.predict_durations(hidden, mask, tokens)
    frames, fmask, flen = m.expand(hidden, dur, mask, spk)
    st = m.prosody_stack
    x = frames; table = __import__('tiny_voca.model', fromlist=['rotary_table']).rotary_table(x.shape[1], st.width, x.device)
    for b in st.blocks: x = b(x, fmask, table)
    pre = x
    ph, pros = m.predict_prosody(frames, fmask)
    wave, _, det = m.synthesize(tokens, stress, lengths, spk)
    return dict(dur=dur, frames=frames, pre=pre[0], ph=ph[0], pros=det['prosody'][0], raw=pros[0], fmask=fmask[0])
