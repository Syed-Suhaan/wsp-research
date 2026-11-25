# Mechanism analysis on base voices 0 and 1: how directly does the frame voice row reach the
# acoustic stack's final residual stream and the vocoder heads?
import sys; sys.path.insert(0,'wsp/stage2/W1/run_6/scratch'); sys.path.insert(0,'.')
from probe import *
from tiny_voca.infer import phonemize
from tiny_voca.model import rotary_table
texts=W.tts_texts("enroll")[:20]
m=W.tts_load_base(); sd=W.tts_base_state_dict()
def acts(voice):
    P=[];A=[];H=[];N=[];PH=[]
    for s in texts:
        p=phonemize(s); tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); spk=torch.tensor([voice])
        with torch.inference_mode():
            h,mask=m.encode(tok,st,L,spk); dur=m.predict_durations(h,mask,tok)
            fr,fm,fl=m.expand(h,dur,mask,spk)
            ph,pr=m.predict_prosody(fr,fm)
            x=(ph+m.prosody_projection(m.hard_prosody(pr)))
            S=m.acoustic_stack; tb=rotary_table(x.shape[1],S.width,x.device)
            for b in S.blocks: x=b(x,fm,tb)
            ah=S.norm(x)
            v=pr[0,:,1]>0
            P.append(ph[0][v]); A.append(x[0][v]); H.append(m.vocoder.harmonic(ah)[0][v]); N.append(m.vocoder.noise(ah)[0][v])
    return [torch.cat(z) for z in (P,A,H,N)]
P0,A0,H0,N0=acts(0); P1,A1,H1,N1=acts(1)
dfv=sd['frame_voices.weight'][1]-sd['frame_voices.weight'][0]
gp=m.prosody_stack.norm.weight.data; ga=m.acoustic_stack.norm.weight.data
print("prosody hidden diff vs gp*dfv/sig: cos", torch.cosine_similarity(P1.mean(0)-P0.mean(0), gp*(dfv-dfv.mean())/2.03,dim=0).item(),
      "norm ratio", ((P1.mean(0)-P0.mean(0)).norm()/(gp*(dfv-dfv.mean())/2.03).norm()).item())
dA=A1.mean(0)-A0.mean(0); dP=P1.mean(0)-P0.mean(0)
print("acoustic resid diff vs prosody hidden diff: cos", torch.cosine_similarity(dA,dP,dim=0).item(), "norm", dA.norm().item(), dP.norm().item())
print("acoustic resid std", A0.std(1,unbiased=False).mean().item(), A1.std(1,unbiased=False).mean().item())
print("gp stats", gp.mean().item(), gp.std().item(), gp.abs().min().item(), " ga", ga.mean().item(), ga.std().item(), ga.abs().min().item())
print("harm head c0 mean v0 %.3f v1 %.3f; noise c0 v0 %.3f v1 %.3f"%(H0[:,0].mean(),H1[:,0].mean(),N0[:,0].mean(),N1[:,0].mean()))
print("harm gains mean v0 %.3f v1 %.3f; noise gains v0 %.3f v1 %.3f"%(H0[:,96:].mean(),H1[:,96:].mean(),N0[:,32:].mean(),N1[:,32:].mean()))
print("harm cep 1..6 v0", H0[:,1:7].mean(0), "v1", H1[:,1:7].mean(0))
Wh=m.vocoder.harmonic.weight.data; print("Wh shape", Wh.shape, "row norms first", Wh.norm(dim=1)[:8])
