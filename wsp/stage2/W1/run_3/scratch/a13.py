# Response of pitch mean and pitch range (internal prediction) to the non-readout part d_perp
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch
torch.set_num_threads(4)
from eval.wsp_tools import *
exec(open(sys.argv[1]+'/a12.py').read().split("texts=tts_texts")[0].split("from tiny_voca.text import phonemize")[1].replace('@torch.no_grad()','',0)) if False else None
from tiny_voca.text import phonemize
@torch.no_grad()
def lf(net,text,voice):
    p=phonemize(text); t=torch.tensor([p.ids]); s=torch.tensor([p.stress]); l=torch.tensor([len(p)]); sp=torch.tensor([voice])
    h,mask=net.encode(t,s,l,sp); dur=net.predict_durations(h,mask,t); fr,fm,_=net.expand(h,dur,mask,sp)
    _,pr=net.predict_prosody(fr,fm)
    owner=torch.repeat_interleave(torch.arange(t.shape[1]),dur[0])
    v=(pr[0,:,1]>0)|net.vowel_tokens[t[0,owner]]
    return pr[0,v,0].numpy()
texts=tts_texts('enroll')[:20]
b=tts_load_base(); FV=b.frame_voices.weight.detach().clone(); V=b.voices.weight.detach().clone()
rh=torch.tensor(np.load(sys.argv[1]+'/rh.npy')); d=FV[0]-FV[1]; dpar=(d@rh)*rh; dperp=d-dpar
def meas(u):
    with torch.no_grad(): b.voices.weight[2]=V[0]; b.frame_voices.weight[2]=u
    x=[lf(b,t,2) for t in texts]; return np.mean([q.std() for q in x]), np.log(200)+np.concatenate(x).mean()
for name,u in (('M2',FV[1]),('M2-dperp',FV[1]-dperp),('M2-dpar',FV[1]-dpar),('M2+dpar',FV[1]+dpar)):
    s,mu=meas(u); print(f'{name:10s} within-sd {s:.3f} mean F0 {np.exp(mu):.1f}')
