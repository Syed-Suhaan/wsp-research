import sys, torch, numpy as np
sys.path.insert(0, '.')
torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.text import phonemize
@torch.inference_mode()
def internals(m, text, voice):
    p = phonemize(text)
    tokens = torch.tensor([p.ids]); stress = torch.tensor([p.stress]); L = torch.tensor([len(p)]); spk = torch.tensor([voice])
    hidden, mask = m.encode(tokens, stress, L, spk)
    dur = m.predict_durations(hidden, mask, tokens)
    frames, fm, fl = m.expand(hidden, dur, mask, spk)
    ph, pros = m.predict_prosody(frames, fm)
    ends = dur.cumsum(1)
    frame = torch.arange(frames.shape[1])[None].contiguous()
    owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1]-1)
    vowel = m.vowel_tokens[tokens].gather(1, owner) & fm
    pros = torch.cat((pros[..., :1], torch.where(vowel, pros[..., 1].clamp_min(2.0), pros[..., 1])[..., None], pros[..., 2:]), -1)
    ac = m.decode(ph, m.hard_prosody(pros), fm)
    h = m.vocoder.harmonic(ac)[0]
    voiced = pros[0, :, 1] > 0
    return dict(dur=dur[0], frames=int(fl[0]), lf0=pros[0, :, 0], voiced=voiced, energy=pros[0, :, 2],
                cep=h[:, :80] * m.vocoder.causal_lifter, anti=h[:, 80:96]*m.vocoder.anti_lifter, gain=h[:, 96:], ac=ac[0], ph=ph[0], frames_in=frames[0])
def summary(m, texts, voice):
    lf, fr, cep, en = [], 0, [], []
    for t in texts:
        r = internals(m, t, voice)
        lf.append(r['lf0'][r['voiced']]); fr += r['frames']; cep.append(r['cep'][r['voiced']]); en.append(r['energy'][r['voiced']])
    lf = torch.cat(lf); cep = torch.cat(cep)
    return dict(lf0=float(lf.mean()), lf0sd=float(lf.std()), frames=fr, cep=cep.mean(0), energy=float(torch.cat(en).mean()))
