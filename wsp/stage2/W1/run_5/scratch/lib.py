import sys, math, numpy as np, torch, pyworld as pw
sys.path.insert(0,'/home/user/wsp-research')
torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.text import phonemize
SR=24000
@torch.inference_mode()
def internals(m, text, voice=2):
    ph = phonemize(text)
    tokens = torch.tensor([ph.ids]); stress = torch.tensor([ph.stress]); lengths = torch.tensor([len(ph)]); spk = torch.tensor([voice])
    hidden, mask = m.encode(tokens, stress, lengths, spk)
    dur = m.predict_durations(hidden, mask, tokens)
    frames, fmask, flen = m.expand(hidden, dur, mask, spk)
    phid, pros = m.predict_prosody(frames, fmask)
    ends = dur.cumsum(1); fr = torch.arange(frames.shape[1])[None].contiguous()
    owner = torch.searchsorted(ends, fr, right=True).clamp(max=tokens.shape[1]-1)
    vowel = m.vowel_tokens[tokens].gather(1, owner) & fmask
    pros = torch.cat((pros[..., :1], torch.where(vowel, pros[..., 1].clamp_min(2.0), pros[..., 1])[..., None], pros[..., 2:]), -1)
    ac = m.decode(phid, m.hard_prosody(pros), fmask)
    return dict(pros=pros[0], h=m.vocoder.harmonic(ac)[0], n=m.vocoder.noise(ac)[0], dur=dur[0], frames=frames[0], phid=phid[0], ac=ac[0])
def pitch_stats(m, texts):
    P=[]
    for t in texts:
        pr = internals(m, t)['pros']; P.append(pr[pr[:,1]>0, 0])
    P=torch.cat(P); return float(P.mean()), float(P.std())
def mean_heads(m, texts):
    H=[]; N=[]
    for t in texts:
        d = internals(m, t); H.append(d['h']); N.append(d['n'])
    return torch.cat(H).mean(0), torch.cat(N).mean(0)
def logpower_env(x):
    """mean log power envelope over voiced frames, 513 bins (0..12 kHz)."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    f0, t = pw.harvest(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    sp = pw.cheaptrick(x, f0, t, SR, fft_size=1024)
    return np.log(sp[f0>0] + 1e-14)
def harvest_lf0(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    f0, t = pw.harvest(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    return np.log(f0[f0>0])
