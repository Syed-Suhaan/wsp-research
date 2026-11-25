# Speaking rate: how frame counts and encoder states respond to the token-level voice row.
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import torch, numpy as np
torch.set_num_threads(4)
from eval.wsp_tools import *
from fwd import prep
m=tts_load_edited(sys.argv[1]+'/cand1.pt'); texts=tts_texts('enroll')
V=m.voices.weight.detach().clone(); de=V[0]-V[1]
print('|de|',de.norm().item(),'|V1|',V[1].norm().item())
@torch.no_grad()
def stats(ve):
    m.voices.weight[2]=ve
    n=0; H=[]; D=[]
    for t in texts:
        a=prep(m,t,2); h,mask=m.encode(a[0],a[1],a[2],a[3]); dur=m.predict_durations(h,mask,a[0])
        n+=int(dur.sum()); H.append(h[0]); D.append(dur[0])
    return n,H,D
n1,H1,D1=stats(V[1]); n0,H0,D0=stats(V[0])
print('frames M2',n1,'F2-enc',n0,'ratio',n0/n1)
a=float(np.log(171.6/139.9)/np.log(149.8/139.9)); print('alpha',a)
na,Ha,Da=stats(V[1]+a*de)
print('frames alpha-ext',na,'ratio',na/n1)
c0=np.mean([torch.cosine_similarity(x,y,1).mean().item() for x,y in zip(H1,H0)])
ca=np.mean([torch.cosine_similarity(x,y,1).mean().item() for x,y in zip(H1,Ha)])
print('encoder output cos to M2: F2-enc',c0,'ext',ca)
# per-token duration ratio spread
r=torch.cat([(da.float()+1)/(d1.float()+1) for da,d1 in zip(Da,D1)])
print('per-token (dur+1) ratio quantiles', np.quantile(r.numpy(),[0.05,0.25,0.5,0.75,0.95]))
pauses=lambda D: sum(int(((d>0)).sum()) for d in D)
print('tokens with frames M2',pauses(D1),'ext',pauses(Da))
