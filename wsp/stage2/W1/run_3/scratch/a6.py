# Model analysis: split the F2-M2 frame-voice difference d into its component along the
# prosody head's direct-path pitch readout r = centre(g * w_logf0) and the rest, and measure
# what each part does to pitch and to the spectral envelope (relative to the two trained
# voices only; the target is not used).
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch
torch.set_num_threads(4)
from eval.wsp_tools import *
from feat import feats
m=tts_load_base(); texts=tts_texts('enroll')
sd=tts_base_state_dict(); FV=sd['frame_voices.weight']
d=FV[0]-FV[1]
r=m.prosody_stack.norm.weight.detach()*m.prosody_head.weight.detach()[0]; r=r-r.mean(); rh=r/r.norm()
dpar=(d@rh)*rh; dperp=d-dpar
print('|d|',d.norm().item(),'|dpar|',dpar.norm().item(),'|dperp|',dperp.norm().item())
def run(vf, n=8):
    with torch.no_grad(): m.frame_voices.weight[2]=vf
    L=[feats(tts_synthesize(m,texts[i],2,0)) for i in range(n)]
    lf=np.concatenate([q['lf0'] for q in L]); mc=np.mean([q['mc_v'] for q in L],0)
    return lf.mean(), mc
R={'M2':run(FV[1]),'F2':run(FV[0]),'M2+dpar':run(FV[1]+dpar),'M2+dperp':run(FV[1]+dperp),'M2-dpar':run(FV[1]-dpar)}
l1,mm=R['M2']; ax=(R['F2'][1]-mm)[1:]
for k,(lf,mc) in R.items():
    x=(mc-mm)[1:]; on=np.dot(x,ax)/np.dot(ax,ax); off=np.linalg.norm(x-on*ax)
    print(f'{k:10s} dlogF0 {lf-l1:+.3f} spectral axis pos {on:+.2f} off-axis {off:.3f}')
np.save(sys.argv[1]+'/rh.npy', rh.numpy())
