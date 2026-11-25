import sys, numpy as np, torch
sys.path.insert(0, '/home/user/wsp-research')
torch.set_num_threads(4)
from eval import wsp_tools as T
from tiny_voca.text import phonemize
from tiny_voca.model import length_mask

def inputs(text):
    p = phonemize(text)
    return torch.tensor([p.ids]), torch.tensor([p.stress]), torch.tensor([len(p)])

@torch.inference_mode()
def internals(m, text, voice):
    tok, st, ln = inputs(text)
    spk = torch.tensor([voice])
    hidden, mask = m.encode(tok, st, ln, spk)
    dur = m.predict_durations(hidden, mask, tok)
    frames, fmask, flen = m.expand(hidden, dur, mask, spk)
    # prosody stack residual before final norm
    ps = m.prosody_stack
    from tiny_voca.model import rotary_table
    x = frames; tab = rotary_table(x.shape[1], ps.width, x.device)
    for b in ps.blocks: x = b(x, fmask, tab)
    pres = x
    ph, pro = m.predict_prosody(frames, fmask)
    w, L, det = m.synthesize(tok, st, ln, spk)
    pro2 = det['prosody']
    acin = (ph + m.prosody_projection(m.hard_prosody(pro2))) * fmask[..., None]
    a = m.acoustic_stack; x = acin
    for b in a.blocks: x = b(x, fmask, tab)
    ares = x
    ac = a.norm(x)
    h = m.vocoder.harmonic(ac); n = m.vocoder.noise(ac)
    return dict(hidden=hidden[0], dur=dur[0], frames=frames[0], pres=pres[0], ph=ph[0], pro=pro2[0], ares=ares[0], ac=ac[0], h=h[0], n=n[0], wave=w[0,:int(L[0])])
