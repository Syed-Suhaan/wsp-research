# Pitch read-out direction of the prosody stack and the residual-stream scale at its final LayerNorm.
exec(open('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/lib.py').read())
texts=W.tts_texts('enroll')[:10]
V,FV=BASE['voices.weight'],BASE['frame_voices.weight']
alpha_t=-0.288; Dlf=-0.30
gam=BASE['prosody_stack.norm.weight']; w0=BASE['prosody_head.weight'][0]
g=gam*w0; g=g-g.mean(); u=g/g.norm()
print('|P(gamma*w0)|',float(g.norm()), 'gamma mean',float(gam.mean()), 'w0 norm',float(w0.norm()))
sd={k:v.clone() for k,v in BASE.items()}
sd['voices.weight'][2]=V[0]; sd['frame_voices.weight'][2]=FV[1]+alpha_t*(FV[0]-FV[1]); m=model_from(sd)
inv=[]; norms={}
@torch.no_grad()
def stream(m,text,voice):
    p=phonemize(text); tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); sp=torch.tensor([voice])
    _,_,d=m.synthesize(tok,st,L,sp)
    hidden,mask=m.encode(tok,st,L,sp)
    frames,fm,fl=m.expand(hidden,d['durations'],mask,sp)
    S=m.prosody_stack; x=frames; tab=__import__('tiny_voca.model',fromlist=['x']).rotary_table(x.shape[1],S.width,x.device)
    for b in S.blocks: x=b(x,fm,tab)
    return frames[0], x[0], d
for t in texts:
    f,x,d=stream(m,t,2)
    inv.append((1/x.std(-1,unbiased=False)).mean().item())
    norms.setdefault('frames_in',[]).append(f.norm(dim=-1).mean().item()); norms.setdefault('final',[]).append(x.norm(dim=-1).mean().item())
inv=np.mean(inv); print('mean 1/sigma',inv, {k:np.mean(v) for k,v in norms.items()})
s=Dlf/(float(g.norm())*inv); print('s',s, 'frame row norm', float(sd['frame_voices.weight'][2].norm()))
print('pred F0 timbre-only:',np.exp(np.mean([internals(m,t,2)['logf0'] for t in texts])))
sd['frame_voices.weight'][2]+=s*u; m2=model_from(sd)
print('pred F0 with pitch term:',np.exp(np.mean([internals(m2,t,2)['logf0'] for t in texts])), 'target 106.4')
