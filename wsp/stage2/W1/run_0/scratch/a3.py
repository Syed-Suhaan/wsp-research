import sys; sys.path.insert(0,'/home/user/wsp-research/wsp/stage2/W1/run_0/scratch')
from probe import *
m = T.tts_load_base()
texts = T.tts_texts('enroll')
w = T.tts_synthesize(m, texts[0], 1, 0); r = internals(m, texts[0], 1)
print('match', torch.as_tensor(w).shape, r['wave'].shape, float((torch.as_tensor(w).float()-r['wave']).abs().max()))
R = {v:[internals(m,t,v) for t in texts[:12]] for v in (0,1)}
def cat(v,k): return torch.cat([d[k] for d in R[v]])
for k in ['hidden','frames','pres','ph','ares','ac']:
    a,b = cat(0,k), cat(1,k)
    print(k, 'rms %.2f %.2f'%(a.pow(2).mean().sqrt(), b.pow(2).mean().sqrt()), 'meandiff norm %.2f'%(a.mean(0)-b.mean(0)).norm(), 'std-per-frame %.2f'%b.std(1).mean())
fv = m.frame_voices.weight; vd = fv[0]-fv[1]
for k in ['frames','pres','ares']:
    md = cat(0,k).mean(0)-cat(1,k).mean(0)
    print(k,'cos(meandiff, framevoice diff) %.3f'%torch.nn.functional.cosine_similarity(md,vd,0))
md = cat(0,'hidden').mean(0)-cat(1,'hidden').mean(0); tv = m.voices.weight[0]-m.voices.weight[1]
print('hidden cos with token voice diff %.3f'%torch.nn.functional.cosine_similarity(md,tv,0))
for v in (0,1):
    pro = cat(v,'pro'); vo = pro[:,1]>0
    print(v, 'logf0 mean voiced %.3f -> %.1f Hz'%(pro[vo,0].mean(), 200*np.exp(pro[vo,0].mean().item())), 'energy mean %.3f'%pro[:,2].mean(), 'frames', sum(int(d['dur'].sum()) for d in R[v]))
