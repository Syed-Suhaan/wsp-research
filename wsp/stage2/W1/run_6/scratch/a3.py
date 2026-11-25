# Activation analysis of the base voices 0 and 1 (no edited weights).
import sys; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
from tiny_voca.infer import phonemize
texts=W.tts_texts("enroll")
m=W.tts_load_base()
sd=W.tts_base_state_dict()
cap={}
def run(text, voice):
    p=phonemize(text)
    tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); spk=torch.tensor([voice])
    with torch.inference_mode():
        h,mask=m.encode(tok,st,L,spk)
        dur=m.predict_durations(h,mask,tok)
        fr,fm,fl=m.expand(h,dur,mask,spk)
        x=fr; st_=m.prosody_stack; table=__import__('tiny_voca.model',fromlist=['x']).rotary_table(x.shape[1],st_.width,x.device)
        for b in st_.blocks: x=b(x,fm,table)
        ph=st_.norm(x)
        pr=m.prosody_head(ph)
    return dict(dur=dur, x=x[0], ph=ph[0], pr=pr[0], fr=fr[0])
W0=m.prosody_head.weight.data; b0=m.prosody_head.bias.data; g=m.prosody_stack.norm.weight.data; bb=m.prosody_stack.norm.bias.data
print("prosody head bias", b0, "norm W rows", W0.norm(dim=1))
for v in (0,1):
    R=[run(t,v) for t in texts[:20]]
    pr=torch.cat([r['pr'] for r in R]); x=torch.cat([r['x'] for r in R])
    voiced=pr[:,1]>0
    sig=x.std(dim=1,unbiased=False)
    print(v,"lf0 mean voiced %.3f -> %.1fHz"%(pr[voiced,0].mean(), 200*np.exp(pr[voiced,0].mean().item())),
          "lf0 sd %.3f"%pr[voiced,0].std(), "energy %.3f"%pr[:,2].mean(), "frames", sum(int(r['dur'].sum()) for r in R),
          "resid std mean %.3f"%sig.mean(), "voiced frac %.2f"%voiced.float().mean())
    if v==0: x0=x.mean(0)
    else: x1=x.mean(0)
dv=sd['frame_voices.weight'][1]-sd['frame_voices.weight'][0]
print("mean resid diff norm", (x1-x0).norm(), "frame voice diff norm", dv.norm(), "cos", torch.cosine_similarity(x1-x0,dv,dim=0))
a=g*W0[0]; ac=a-a.mean()
print("cos(frame_voice diff, g*w_f0 centered)", torch.cosine_similarity(dv,ac,dim=0))
dvt=sd['voices.weight'][1]-sd['voices.weight'][0]
print("cos(tok voice diff, ac)", torch.cosine_similarity(dvt,ac,dim=0))
