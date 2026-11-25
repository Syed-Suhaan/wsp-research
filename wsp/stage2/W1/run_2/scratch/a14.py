import sys; sys.path.insert(0, sys.argv[1])
from probe import *
from spec import mean_cep
mA = W.tts_load_edited('checkpoints/stage2_W1_run2.pt')
en = W.tts_texts('enroll'); idx = list(range(0,50,3))
ct, nt = mean_cep([W.tts_target_audio('enroll', i) for i in idx])
ca, na = mean_cep([np.asarray(W.tts_synthesize(mA, en[i], 2, 0)) for i in idx])
d = ct-ca
print('frames', nt, na)
print('cep diff (k=0..20)', d[:21].round(3))
# implied log-magnitude difference (k=1..24), in dB, at some frequencies
w = np.linspace(0, np.pi, 257); K=24
D = sum(2*d[k]*np.cos(k*w) for k in range(1,K+1))
for f in (100,200,400,700,1000,1500,2000,3000,4000,5000,6000,8000,10000,12000):
    j=int(f/12000*256); print(f, round(20/np.log(10)*D[j],1))
np.save(sys.argv[1]+'/cepdiff.npy', d)
