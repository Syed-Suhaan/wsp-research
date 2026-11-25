# Endpoint analysis: features of the target recordings and of the two trained voices (base model).
exec(open('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/lib.py').read())
N=20
texts=W.tts_texts('enroll')[:N]
m=W.tts_load_base()
def active(x):
    e=np.sqrt(np.convolve(x**2,np.ones(240)/240,'same')); idx=np.where(e>e.max()*0.03)[0]
    return (idx[-1]-idx[0])/SR
out={}
for name in ('target','v0','v1'):
    F=[];D=[];P=[]
    for i,t in enumerate(texts):
        x=W.tts_target_audio('enroll',i) if name=='target' else synth(m,t,int(name[1]))
        F.append(feats(x)); D.append(active(x))
        if name!='target': P.append(internals(m,t,int(name[1]))['logf0'])
    out[name]=dict(lf0=np.array([f['lf0'] for f in F]),cep=np.array([f['cep'] for f in F]),lmel=np.array([f['lmel'] for f in F]),dur=np.array(D),pred=np.array(P))
    print(name,'F0 %.1f'%np.exp(out[name]['lf0'].mean()),'pred F0 %s'%(np.exp(np.mean(P)) if P else '-'),'active dur %.2f'%np.sum(D),'c1..6',np.round(out[name]['cep'].mean(0)[1:7],2))
np.save('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/endpoints.npy',out,allow_pickle=True)
