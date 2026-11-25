import sys; sys.path.insert(0, sys.argv[1])
from probe import *
from feat import analyze
from dtw import dtw_path
m = W.tts_load_base(); sd = {k:v.clone() for k,v in m.state_dict().items()}
u = torch.load(sys.argv[1]+'/u.pt')['u']
en = W.tts_texts('enroll'); idx=list(range(2,50,6))
K = 10*np.sqrt(2)/np.log(10)
def mc(a): return a['c'][a['act']][::2]   # 10 ms, active frames
T = [analyze(W.tts_target_audio('enroll', i)) for i in idx]
with torch.no_grad(): m.voices.weight[2]=sd['voices.weight'][0]; m.frame_voices.weight[2]=sd['frame_voices.weight'][1]-7.5*u
for tag, v in (('M2',1),('A',2)):
    diffs=[]; mcds=[]
    for i,t in zip(idx,T):
        a = analyze(np.asarray(W.tts_synthesize(m, en[i], v, 0)))
        X, Y = mc(t), mc(a)
        p = dtw_path(X[:,1:], Y[:,1:])
        d = np.array([X[i_,1:]-Y[j_,1:] for i_,j_ in p]); diffs.append(d)
        mcds.append(K*np.mean(np.sqrt((d**2).sum(1))))
    d = np.concatenate(diffs); mu = d.mean(0)
    print(tag, 'MCD-ish', np.mean(mcds).round(2), 'mean diff', mu[:12].round(3))
    print('   MCD if mean removed', (K*np.mean(np.sqrt(((d-mu)**2).sum(1)))).round(2))
