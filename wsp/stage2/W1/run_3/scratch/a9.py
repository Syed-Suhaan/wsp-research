# Does the direct path (prosody LN -> acoustic residual -> acoustic LN -> harmonic head)
# predict the direction of the harmonic-cepstrum change between voices F2 and M2?
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import torch, numpy as np
torch.set_num_threads(4)
from eval.wsp_tools import *
from fwd import *
m=tts_load_base(); texts=tts_texts('enroll')
FV=m.frame_voices.weight.detach().clone(); d=FV[0]-FV[1]
gp=m.prosody_stack.norm.weight.detach(); ga=m.acoustic_stack.norm.weight.detach()
W=m.vocoder.harmonic.weight.detach()[:80]
P=lambda v: v-v.mean(-1,keepdim=True)
@torch.no_grad()
def heads(fr,fm):
    ph,pros=m.predict_prosody(fr,fm)
    ac=m.decode(ph,m.hard_prosody(pros),fm)
    return ph, ac, m.vocoder.harmonic(ac)[...,:80], pros
for i in range(3):
    fr1,fm,_,_=frames_for(m,texts[i],1)
    ph1,ac1,h1,_=heads(fr1,fm); ph0,ac0,h0,_=heads(fr1+d,fm)
    dh=(h0-h1)[0]
    # direct path prediction per frame
    xp=prosody_internals(m,fr1,fm)[0]; sp=xp.std(1,unbiased=False,keepdim=True)
    dyp=gp*P(d)[None]/sp
    # acoustic residual pre-LN
    from tiny_voca.model import rotary_table
    xa=(ph1+m.prosody_projection(m.hard_prosody(m.predict_prosody(fr1,fm)[1])))*fm[...,None]
    tab=rotary_table(xa.shape[1],m.acoustic_stack.width,'cpu')
    with torch.no_grad():
        for b in m.acoustic_stack.blocks: xa=b(xa,fm,tab)
    sa=xa[0].std(1,unbiased=False,keepdim=True)
    dya=ga*P(dyp)/sa
    pred=dya@W.T
    cos_frame=torch.cosine_similarity(pred,dh,1).mean()
    cos_mean=torch.cosine_similarity(pred.mean(0),dh.mean(0),0)
    print(i,'cos per frame',cos_frame.item(),'cos of means',cos_mean.item(),'norm ratio',(dh.mean(0).norm()/pred.mean(0).norm()).item(), 'actual dyp vs pred cos', torch.cosine_similarity((ph0-ph1)[0].mean(0),dyp.mean(0),0).item(),
          'dac vs dyp cos',torch.cosine_similarity((ac0-ac1)[0].mean(0),dya.mean(0),0).item())
