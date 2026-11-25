import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_0/scratch')
from probe import *
torch.set_grad_enabled(False)
m = T.tts_load_base()
texts = T.tts_texts('enroll')
R = {v:[internals(m,t,v) for t in texts[:12]] for v in (0,1)}
def cat(v,k): return torch.cat([d[k] for d in R[v]])
gp = m.prosody_stack.norm.weight; w0 = m.prosody_head.weight[0]
u = gp*w0; u = u-u.mean()
sig = cat(1,'pres').std(1).mean()
print('||u||', u.norm().item(), 'sigma_p', sig.item(), 'max dlogf0 per unit norm', (u.norm()/sig).item())
fv = m.frame_voices.weight; vd = fv[0]-fv[1]
print('direct-path pred dlogf0 for frame diff', (u@vd/sig).item())
md = cat(0,'pres').mean(0)-cat(1,'pres').mean(0)
print('pred from actual pres meandiff', (u@md/sig).item())
# actual
for v in (0,1):
    pro=cat(v,'pro'); vo=pro[:,1]>0; print(v, pro[vo,0].mean().item(), pro[vo,0].std().item())
# cos between u and vd
print('cos(u, vd)', torch.nn.functional.cosine_similarity(u,vd,0).item())
print('w0 norm', w0.norm().item(), 'gp mean', gp.abs().mean().item())
# harmonic head analysis
ga = m.acoustic_stack.norm.weight; Wh = m.vocoder.harmonic.weight
print('Wh shape', Wh.shape, 'sigma_a', cat(1,'ares').std(1).mean().item())
hd = cat(0,'h').mean(0)-cat(1,'h').mean(0)
print('h diff first 10', hd[:10])
