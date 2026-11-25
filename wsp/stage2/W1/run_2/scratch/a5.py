import sys, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, sys.argv[1])
import torch; torch.set_num_threads(4)
from eval import wsp_tools as W
from feat import analyze
import pyworld as pw
en = W.tts_texts('enroll'); m = W.tts_load_base()
idx = list(range(1,50,5))
def meanlogsp(L):
    return np.concatenate([np.log(a['sp'][a['v']]) for a in L]).mean(0)
def trimmed(a):
    e = 10*np.log10(np.mean(a['sp'],1)+1e-20); act = np.where(e > e.max()-35)[0]; return (act[-1]-act[0])*0.005
T = [analyze(W.tts_target_audio('enroll', i)) for i in idx]
M = {v: [analyze(np.asarray(W.tts_synthesize(m, en[i], v, 0))) for i in idx] for v in (0,1)}
print('trimmed dur target', np.mean([trimmed(a) for a in T]), 'v0', np.mean([trimmed(a) for a in M[0]]), 'v1', np.mean([trimmed(a) for a in M[1]]))
print('ratio per sentence', np.round([trimmed(a)/trimmed(b) for a,b in zip(T,M[1])],2))
lt, l0, l1 = meanlogsp(T), meanlogsp(M[0]), meanlogsp(M[1])
f = np.linspace(0,12000,len(lt))
edges = [0,150,300,500,750,1000,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12001]
db = lambda x: 20/np.log(10)*x
for a,b in zip(edges[:-1],edges[1:]):
    s = (f>=a)&(f<b)
    print(f'{a:6d}-{b:6d}  tgt {db(lt[s].mean()):7.1f}  v1 {db(l1[s].mean()):7.1f}  v0 {db(l0[s].mean()):7.1f}  tgt-v1 {db(lt[s].mean()-l1[s].mean()):6.1f}')
# aperiodicity
ap = lambda L: np.concatenate([a['ap'][a['v']] for a in L]).mean(0)
at, a1 = ap(T), ap(M[1])
for fr in (500,1000,2000,4000,8000):
    k = int(fr/12000*(len(at)-1)); print('ap', fr, round(at[k],3), round(a1[k],3))
pickle.dump(dict(lt=lt,l0=l0,l1=l1), open(sys.argv[1]+'/env.pkl','wb'))
