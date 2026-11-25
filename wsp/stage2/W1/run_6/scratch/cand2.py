# Candidate 2: keep M2 timbre (frame row 2 = M2 row) and add a pitch-only steering vector to frame
# row 2: the prosody head's log-F0 readout direction, mapped back through the final LayerNorm
# gain, orthogonalized to the voicing and energy readouts. Its size is the closed-form solution of
# the LayerNorm equation that shifts the predicted log F0 by log(F0_target / F0_M2).
import sys; sys.path.insert(0,'.')
import numpy as np, torch, pyworld as pw, math
from scipy.signal import resample_poly
torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.infer import phonemize
from tiny_voca.model import rotary_table
sd=torch.load("checkpoints/stage2_base.pt",map_location="cpu",weights_only=True)
m=W.tts_load_base(); texts=W.tts_texts("enroll")
NS=25
def lf0(x):
    x=resample_poly(np.asarray(x,dtype=np.float64).reshape(-1),1,3)
    f0,_=pw.harvest(np.ascontiguousarray(x),8000,f0_floor=50,f0_ceil=600,frame_period=10); return np.log(f0[f0>0])
tgt=np.median(np.concatenate([lf0(W.tts_target_audio("enroll",i)) for i in range(NS)]))
m1=np.median(np.concatenate([lf0(W.tts_synthesize(m,texts[i],1,0)) for i in range(NS)]))
delta=tgt-m1
# prosody-stack activations for voice 1 (M2) on voiced frames
X=[]
for s in texts[:NS]:
    p=phonemize(s); tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); spk=torch.tensor([1])
    with torch.inference_mode():
        h,mask=m.encode(tok,st,L,spk); dur=m.predict_durations(h,mask,tok); fr,fm,_=m.expand(h,dur,mask,spk)
        x=fr; tb=rotary_table(x.shape[1],m.prosody_stack.width,x.device)
        for b in m.prosody_stack.blocks: x=b(x,fm,tb)
        pr=m.prosody_head(m.prosody_stack.norm(x))[0]
    X.append(x[0][pr[:,1]>0])
X=torch.cat(X).double()
C=X.shape[1]
xc=X-X.mean(1,keepdim=True)
sigma=math.sqrt((xc**2).mean().item())                     # rms per-frame std
g=sd['prosody_stack.norm.weight'].double(); Wp=sd['prosody_head.weight'].double()
s=(xc/sigma@(g*Wp[0])).mean().item()                      # mean of the gain-weighted, normalized log-F0 term
# direction: g*w_f0, made zero-mean and orthogonal to g*w_voicing and g*w_energy (Gram-Schmidt)
basis=[torch.ones(C,dtype=torch.float64)/math.sqrt(C)]
for r in (1,2):
    v=g*Wp[r]
    for e in basis: v=v-(v@e)*e
    basis.append(v/v.norm())
a=g*Wp[0]; d=a.clone()
for e in basis: d=d-(d@e)*e
d=d/d.norm()
k=(a@d).item(); q=1.0/C                                    # unit-norm zero-mean d adds q*c^2 to the per-frame variance
# solve (sigma*s + c*k)^2 = (s+delta)^2 * (sigma^2 + q c^2) for c (closed-form quadratic)
A=k*k-(s+delta)**2*q; B=2*sigma*s*k; Cc=sigma**2*(s*s-(s+delta)**2)
roots=[(-B+sg*math.sqrt(B*B-4*A*Cc))/(2*A) for sg in (1,-1)]
c=[r for r in roots if r<0][0] if delta<0 else [r for r in roots if r>0][0]
print("target %.1fHz M2 %.1fHz delta %.3f sigma %.3f s %.3f k %.4f c %.3f roots %s"%(math.exp(tgt),math.exp(m1),delta,sigma,s,k,c,roots))
sd['frame_voices.weight'][2]=sd['frame_voices.weight'][1]+(c*d).float()
torch.save(sd,"wsp/stage2/W1/run_6/scratch/cand2.pt")
# mechanism check of the internal readout only: predicted mean log F0 for voice 1 and the edited voice 2
m2=W.tts_load_edited("wsp/stage2/W1/run_6/scratch/cand2.pt")
for v,mm in ((1,m),(2,m2)):
    out=[];vf=[]
    for t in texts[:NS]:
        p=phonemize(t)
        with torch.inference_mode(): _,_,dd=mm.synthesize(torch.tensor([p.ids]),torch.tensor([p.stress]),torch.tensor([len(p)]),torch.tensor([v]))
        pr=dd['prosody'][0]; out.append(pr[pr[:,1]>0,0]); vf.append((pr[:,1]>0).float().mean())
    o=torch.cat(out); print("voice",v,"pred lf0 mean %.3f sd %.3f voiced frac %.3f"%(o.mean(),o.std(),torch.stack(vf).mean()))
