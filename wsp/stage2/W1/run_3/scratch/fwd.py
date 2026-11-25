import sys; sys.path.insert(0,'.')
import torch
from tiny_voca.text import phonemize
def prep(m, text, voice):
    p=phonemize(text)
    return torch.tensor([p.ids]), torch.tensor([p.stress]), torch.tensor([len(p)]), torch.tensor([voice])
@torch.no_grad()
def frames_for(m, text, voice):
    t,s,l,sp=prep(m,text,voice)
    h,mask=m.encode(t,s,l,sp)
    dur=m.predict_durations(h,mask,t)
    fr,fm,fl=m.expand(h,dur,mask,sp)
    return fr,fm,dur,t
@torch.no_grad()
def prosody_internals(m, frames, fm):
    st=m.prosody_stack; x=frames
    from tiny_voca.model import rotary_table
    table=rotary_table(x.shape[1], st.width, x.device)
    for b in st.blocks: x=b(x,fm,table)
    return x  # pre-LN residual
