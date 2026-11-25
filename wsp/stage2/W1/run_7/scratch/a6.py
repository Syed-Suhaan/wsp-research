import sys; sys.path.insert(0,'/home/user/wsp-research')
import torch, numpy as np, math; torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.text import phonemize
from tiny_voca.model import rotary_table
m = W.tts_load_base()
texts = W.tts_texts('enroll')[:10]
ph=[phonemize(t) for t in texts]
FV=m.frame_voices.weight.data.clone()
S=m.prosody_stack
g=S.norm.weight; b=S.norm.bias; wp=m.prosody_head.weight[0]
u=g*wp; u=u-u.mean()
print('||g*w|| centered', u.norm().item(), ' |g| mean', g.abs().mean().item(), '|wp|',wp.norm().item())
d=FV[0]-FV[1]
print('cos(d,u)', torch.cosine_similarity(d,u,dim=0).item(), 'cos(FV1,u)',torch.cosine_similarity(FV[1],u,dim=0).item())
@torch.inference_mode()
def pre(voice):
    xs=[]; 
    for p in ph:
        tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); sp=torch.tensor([voice])
        h,mask=m.encode(tok,st,L,sp); dd=m.predict_durations(h,mask,tok)
        fr,fm,fl=m.expand(h,dd,mask,sp)
        x=fr; table=rotary_table(x.shape[1],S.width,x.device)
        for blk in S.blocks: x=blk(x,fm,table)
        xs.append((fr[0],x[0]))
    return xs
for v in [0,1]:
    xs=pre(v)
    X=torch.cat([a[1] for a in xs]); F_in=torch.cat([a[0] for a in xs])
    sig=X.std(1)
    print('voice',v,'pre-LN residual std mean %.3f  input std mean %.3f'%(sig.mean(),F_in.std(1).mean()))
    print('   mean residual proj onto FV[v] dir: %.3f'%((X@FV[v]/FV[v].norm()).mean()), 'input proj %.3f'%((F_in@FV[v]/FV[v].norm()).mean()))
    print('   pitch per unit e along u_hat (direct path) ~ %.4f'%( (u@u/u.norm())/sig.mean() ))
    print('   pitch per unit e along d_hat (direct path) ~ %.4f'%( (u@d/d.norm())/sig.mean() ))
