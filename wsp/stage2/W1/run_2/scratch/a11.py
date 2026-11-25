import sys; sys.path.insert(0, sys.argv[1])
from probe import *
from scipy.signal import stft
m = W.tts_load_base(); sd = {k:v.clone() for k,v in m.state_dict().items()}
u = torch.load(sys.argv[1]+'/u.pt')['u']
en = W.tts_texts('enroll'); idx=list(range(0,50,4))
bands = ((0,100),(100,300),(300,700),(700,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6000),(6000,8000),(8000,10000),(10000,12000))
def ltas(xs):
    acc = []
    for x in xs:
        f,t,Z = stft(np.asarray(x,float), 24000, nperseg=1024, noverlap=768); P=np.abs(Z)**2; e=10*np.log10(P.sum(0)+1e-20)
        acc.append(P[:, e > e.max()-30])
    P = np.concatenate(acc,1).mean(1); ref = P[(f>=1000)&(f<2000)].mean()
    return [10*np.log10(P[(f>=a)&(f<b)].mean()/ref) for a,b in bands]
T = ltas([W.tts_target_audio('enroll', i) for i in idx])
M = ltas([W.tts_synthesize(m, en[i], 1, 0) for i in idx])
with torch.no_grad(): m.voices.weight[2]=sd['voices.weight'][0]; m.frame_voices.weight[2]=sd['frame_voices.weight'][1]-7.5*u
A = ltas([W.tts_synthesize(m, en[i], 2, 0) for i in idx])
for b,t,mm,a in zip(bands,T,M,A): print(b, round(t,1), round(mm,1), round(a,1))
