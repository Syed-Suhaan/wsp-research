import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_0/scratch')
from probe import *
torch.set_grad_enabled(False)
b = T.tts_load_base(); m = T.tts_load_edited('checkpoints/stage2_W1_run0.pt')
texts = T.tts_texts('enroll')[25:33]
R = {0:[internals(b,t,0) for t in texts],1:[internals(b,t,1) for t in texts],2:[internals(m,t,2) for t in texts]}
def cat(v,k): return torch.cat([d[k] for d in R[v]])
sp = cat(1,'pres').std(1).mean(); sa = cat(1,'ares').std(1).mean()
gp=b.prosody_stack.norm.weight; ga=b.acoustic_stack.norm.weight; Wh=b.vocoder.harmonic.weight[:80]
def P(x): return x - x.mean()
fv=b.frame_voices.weight
for name,delta,actual in [('F2-M2', fv[0]-fv[1], cat(0,'h').mean(0)-cat(1,'h').mean(0)),('NEW-M2', m.frame_voices.weight[2]-fv[1], cat(2,'h').mean(0)-cat(1,'h').mean(0))]:
    pred = Wh @ (ga/sa*P(gp/sp*P(delta)))
    act = actual[:80]
    print(name,'pred norm %.3f actual norm %.3f cos %.3f'%(pred.norm(),act.norm(),torch.nn.functional.cosine_similarity(pred,act,0)))
    print(' ac meandiff norm', (cat(0 if name=='F2-M2' else 2,'ac').mean(0)-cat(1,'ac').mean(0)).norm().item())
for v in (0,1,2):
    pro=cat(v,'pro'); vo=pro[:,1]>0
    print(v,'logf0 %.3f energy %.3f sd %.3f, frames %d'%(pro[vo,0].mean(),pro[:,2].mean(),pro[vo,0].std(), sum(int(d['dur'].sum()) for d in R[v])))
