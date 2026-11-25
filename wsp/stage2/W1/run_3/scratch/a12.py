# internal pitch range (sd of predicted log F0 within sentences) for M2, F2, and the candidate
import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, torch
torch.set_num_threads(4)
from eval.wsp_tools import *
from tiny_voca.text import phonemize
@torch.no_grad()
def lf(net,text,voice):
    p=phonemize(text); t=torch.tensor([p.ids]); s=torch.tensor([p.stress]); l=torch.tensor([len(p)]); sp=torch.tensor([voice])
    _,_,det=net.synthesize(t,s,l,sp) if False else (None,None,None)
    h,mask=net.encode(t,s,l,sp); dur=net.predict_durations(h,mask,t); fr,fm,_=net.expand(h,dur,mask,sp)
    _,pr=net.predict_prosody(fr,fm)
    owner=torch.repeat_interleave(torch.arange(t.shape[1]),dur[0])
    v=(pr[0,:,1]>0)|net.vowel_tokens[t[0,owner]]
    return pr[0,v,0].numpy()
texts=tts_texts('enroll')[:20]
b=tts_load_base(); c=tts_load_edited('checkpoints/stage2_W1_run3.pt')
for name,net,v in (('M2',b,1),('F2',b,0),('cand2',c,2)):
    x=[lf(net,t,v) for t in texts]
    print(name,'mean within-sentence sd',np.mean([q.std() for q in x]),'pooled sd',np.concatenate(x).std(),'mean F0',200*np.exp(np.concatenate(x).mean()))
c1=tts_load_edited(sys.argv[1]+'/cand1.pt')
x=[lf(c1,t,2) for t in texts]; print('cand1 sd',np.mean([q.std() for q in x]),200*np.exp(np.concatenate(x).mean()))
with torch.no_grad(): b.voices.weight[2]=b.voices.weight[0]
x=[lf(b,t,2) for t in texts]; print('tokF2 frameM2 sd',np.mean([q.std() for q in x]),200*np.exp(np.concatenate(x).mean()))
