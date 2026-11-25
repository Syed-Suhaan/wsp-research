import sys, time, torch, numpy as np, pickle
sys.path.insert(0, '.'); sys.path.insert(0, sys.argv[1])
torch.set_num_threads(4)
from eval import wsp_tools as W
from feat import analyze
en = W.tts_texts('enroll')
res = {'target':[], 0:[], 1:[]}
m = W.tts_load_base()
t=time.time()
for i in range(0,50,2):
    res['target'].append(analyze(W.tts_target_audio('enroll', i)))
    for v in (0,1):
        res[v].append(analyze(np.asarray(W.tts_synthesize(m, en[i], v, 0))))
print('time', time.time()-t)
def summ(L):
    lf = np.concatenate([np.log(a['f0'][a['v']]) for a in L])
    voiced_c = np.concatenate([a['c'][a['v']] for a in L])
    act = np.mean([a['act'].sum()*0.005 for a in L]); vd=np.mean([a['v'].sum()*0.005 for a in L])
    return lf.mean(), lf.std(), act, vd, voiced_c.mean(0)
S = {k: summ(L) for k,L in res.items()}
for k,s in S.items(): print(k, 'logf0 %.3f sd %.3f active %.2fs voiced %.2fs'%s[:4], np.round(s[4][:10],3))
pickle.dump(S, open(sys.argv[1]+'/stats_a2.pkl','wb'))
