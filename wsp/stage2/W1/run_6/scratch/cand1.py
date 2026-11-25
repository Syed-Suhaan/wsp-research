# Candidate 1: move frame-level voice row 2 along the F2->M2 axis by a closed-form factor
# chosen so that the predicted pitch matches the target's median pitch. Token row unchanged (= M2).
import sys, time; sys.path.insert(0,'.')
import numpy as np, torch, pyworld as pw
torch.set_num_threads(4); torch.manual_seed(0)
from eval import wsp_tools as W
from tiny_voca.infer import phonemize
SR=24000
sd=torch.load("checkpoints/stage2_base.pt",map_location="cpu",weights_only=True)
m=W.tts_load_base(); texts=W.tts_texts("enroll")
def median_lf0(x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float64).reshape(-1))
    f0,_=pw.harvest(x,SR,f0_floor=50,f0_ceil=600,frame_period=10); return f0[f0>0]
def pred_lf0(frame_row):
    m.frame_voices.weight.data[2]=frame_row; m.voices.weight.data[2]=sd['voices.weight'][1]
    out=[]
    for s in texts:
        p=phonemize(s)
        with torch.inference_mode():
            _,_,d=m.synthesize(torch.tensor([p.ids]),torch.tensor([p.stress]),torch.tensor([len(p)]),torch.tensor([2]))
        pr=d['prosody'][0]; out.append(pr[pr[:,1]>0,0])
    return torch.cat(out).mean().item()
fv=sd['frame_voices.weight']
tgt=np.median(np.log(np.concatenate([median_lf0(W.tts_target_audio("enroll",i)) for i in range(50)])))
v1a=np.median(np.log(np.concatenate([median_lf0(W.tts_synthesize(m,s,1,0)) for s in texts])))
p0=pred_lf0(fv[0]); p1=pred_lf0(fv[1])
goal=p1+(tgt-v1a)
alpha=(goal-p0)/(p1-p0)
print("target %.1fHz v1 audio %.1fHz pred p0 %.3f p1 %.3f goal %.3f alpha %.3f"%(np.exp(tgt),np.exp(v1a),p0,p1,goal,alpha))
sd['frame_voices.weight'][2]=fv[0]+alpha*(fv[1]-fv[0])
torch.save(sd,"wsp/stage2/W1/run_6/scratch/cand1.pt")
