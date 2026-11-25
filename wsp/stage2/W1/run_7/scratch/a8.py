import sys; sys.path.insert(0,'/home/user/wsp-research')
import torch, math; torch.set_num_threads(4)
from eval import wsp_tools as W
from tiny_voca.text import phonemize
m = W.tts_load_edited('checkpoints/stage2_W1_run7.pt')
ph=[phonemize(t) for t in W.tts_texts('enroll')[:10]]
P=[]
with torch.inference_mode():
    for p in ph:
        tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); sp=torch.tensor([2])
        h,mask=m.encode(tok,st,L,sp); d=m.predict_durations(h,mask,tok)
        fr,fm,fl=m.expand(h,d,mask,sp); _,pr=m.predict_prosody(fr,fm); P.append(pr[0,pr[0,:,1]>0,0])
P=torch.cat(P)+math.log(200); print('internal pitch mean %.3f sd %.3f'%(P.mean(),P.std()))
