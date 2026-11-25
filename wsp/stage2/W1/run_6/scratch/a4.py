# Mechanism analysis: which table (token-level or frame-level voice) carries pitch and duration?
# Uses the two existing voice rows only, in the 2x2 combinations (tok row, frame row) in {0,1}^2.
import sys; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
from tiny_voca.infer import phonemize
texts=W.tts_texts("enroll")[:20]
m=W.tts_load_base()
V=m.voices.weight.data.clone(); FV=m.frame_voices.weight.data.clone()
a=m.prosody_stack.norm.weight.data*m.prosody_head.weight.data[0]; ac=a-a.mean()
print("||ac||", ac.norm().item())
for t in (0,1):
  for f in (0,1):
    m.voices.weight.data[2]=V[t]; m.frame_voices.weight.data[2]=FV[f]
    lf=[];nfr=0
    for s in texts:
        p=phonemize(s)
        with torch.inference_mode():
            _,_,d=m.synthesize(torch.tensor([p.ids]),torch.tensor([p.stress]),torch.tensor([len(p)]),torch.tensor([2]))
        pr=d['prosody'][0]; v=pr[:,1]>0; lf.append(pr[v,0]); nfr+=int(d['durations'].sum())
    lf=torch.cat(lf)
    print("tok",t,"frm",f,"lf0 %.3f (%.1f Hz) frames %d"%(lf.mean(),200*np.exp(lf.mean().item()),nfr))
