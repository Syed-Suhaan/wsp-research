import sys; sys.path.insert(0,'/home/user/wsp-research')
import torch, numpy as np, math; torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.text import phonemize
m = W.tts_load_base()
texts = W.tts_texts('enroll')[:10]
ph=[phonemize(t) for t in texts]
V=m.voices.weight.data.clone(); FV=m.frame_voices.weight.data.clone()
@torch.inference_mode()
def prosody(fv, v=None):
    m.frame_voices.weight.data[2]=fv
    if v is not None: m.voices.weight.data[2]=v
    P=[]
    for p in ph:
        tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); sp=torch.tensor([2])
        h,mask=m.encode(tok,st,L,sp); d=m.predict_durations(h,mask,tok)
        fr,fm,fl=m.expand(h,d,mask,sp); ph_h,pr=m.predict_prosody(fr,fm)
        vo=pr[0,:,1]>0
        P.append(pr[0,vo,0])
    P=torch.cat(P)+math.log(200)
    return P.mean().item(), P.std().item()
m.voices.weight.data[2]=V[0]
for a in []:
    fv=FV[1]+a*(FV[0]-FV[1])
    print('lin a=%5.2f norm %.1f pitch mean %.3f sd %.3f'%(a,fv.norm(),*prosody(fv)))
print('--- norm-preserving (rescaled to ||FV1||)')
n1=FV[1].norm()
for a in [-0.5,-1.0,-2.0,-4.0]:
    fv=FV[1]+a*(FV[0]-FV[1]); fv=fv/fv.norm()*n1
    print('nrm a=%5.2f cos_to_M2 %.2f pitch mean %.3f sd %.3f'%(a,torch.cosine_similarity(fv,FV[1],dim=0),*prosody(fv)))
print('--- scale M2 row only')
for s in [0.5,1.5,2.0]:
    print('scale %.1f pitch mean %.3f sd %.3f'%(s,*prosody(FV[1]*s)))
print('--- zero row', prosody(FV[1]*0))
