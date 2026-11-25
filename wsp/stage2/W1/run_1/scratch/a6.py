# Check of the written edit on enrollment sentences: measured F0 and cepstrum of voice 2.
exec(open('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/lib.py').read())
o=np.load('/home/user/wsp-research/wsp/stage2/W1/run_1/scratch/endpoints.npy',allow_pickle=True).item()
texts=W.tts_texts('enroll')[:8]
m=W.tts_load_edited('checkpoints/stage2_W1_run1.pt')
F=[feats(synth(m,t,2)) for t in texts]
c2=np.mean([f['cep'] for f in F],0); ct=o['target']['cep'][:8].mean(0); c1=o['v1']['cep'][:8].mean(0); c0=o['v0']['cep'][:8].mean(0)
print('F0 voice2 %.1f target %.1f'%(np.exp(np.mean([f['lf0'] for f in F])), np.exp(o['target']['lf0'][:8].mean())))
print('c1..10 v2 ',np.round(c2[1:11],2)); print('c1..10 tgt',np.round(ct[1:11],2)); print('c1..10 M2 ',np.round(c1[1:11],2))
lm2=np.mean([f['lmel'] for f in F],0); lmt=o['target']['lmel'][:8].mean(0)
d=lmt-lm2; print('target - v2 logmel (mean-removed):',np.round(d-d.mean(),1))
