exec(open('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/lib.py').read())
texts=W.tts_texts('enroll')[:8]
V,FV=BASE['voices.weight'],BASE['frame_voices.weight']
def cfg(a,b):
    sd={k:v.clone() for k,v in BASE.items()}
    sd['voices.weight'][2]=V[a]; sd['frame_voices.weight'][2]=FV[b]; return model_from(sd)
for name,(a,b) in {'M2M2':(1,1),'F2tokM2frm':(0,1),'M2tokF2frm':(1,0),'F2F2':(0,0)}.items():
    m=cfg(a,b); r=[internals(m,t,2) for t in texts]
    print(name, 'F0 %.1f'%np.exp(np.mean([x['logf0'] for x in r])), 'frames',sum(x['frames'] for x in r), 'en %.3f'%np.mean([x['energy'] for x in r]))
