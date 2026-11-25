# Mechanism analysis: how much of the F2/M2 spectral-envelope difference does each voice table carry?
exec(open('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/lib.py').read())
texts=W.tts_texts('enroll')[:8]
V,FV=BASE['voices.weight'],BASE['frame_voices.weight']
@torch.no_grad()
def envelope(m,text,voice):
    p=phonemize(text); tok=torch.tensor([p.ids]); st=torch.tensor([p.stress]); L=torch.tensor([len(p)]); sp=torch.tensor([voice])
    _,_,d=m.synthesize(tok,st,L,sp)
    hidden,mask=m.encode(tok,st,L,sp)
    frames,fm,fl=m.expand(hidden,d['durations'],mask,sp)
    ph,_=m.predict_prosody(frames,fm)
    ac=m.decode(ph,m.hard_prosody(d['prosody']),fm)
    h=m.vocoder.harmonic(ac)[0]; voiced=d['prosody'][0,:,1]>0
    return (h[voiced,:80]*m.vocoder.causal_lifter).mean(0).numpy()
def cfg(a,b):
    sd={k:v.clone() for k,v in BASE.items()}
    sd['voices.weight'][2]=V[a]; sd['frame_voices.weight'][2]=FV[b]; return model_from(sd)
E={}
for name,(a,b) in {'M2M2':(1,1),'F2tokM2frm':(0,1),'M2tokF2frm':(1,0),'F2F2':(0,0)}.items():
    m=cfg(a,b); E[name]=np.mean([envelope(m,t,2) for t in texts],0)
full=E['F2F2']-E['M2M2']
for k in ('F2tokM2frm','M2tokF2frm'):
    d=E[k]-E['M2M2']; print(k,'share of F2-M2 envelope diff: proj %.3f, norm ratio %.3f'%(d@full/(full@full), np.linalg.norm(d[1:])/np.linalg.norm(full[1:])))
print('c0..6 M2',np.round(E['M2M2'][:7],2)); print('c0..6 F2',np.round(E['F2F2'][:7],2))
