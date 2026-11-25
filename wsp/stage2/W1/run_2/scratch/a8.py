import sys, pickle; sys.path.insert(0, sys.argv[1])
from probe import *
from feat import analyze
m = W.tts_load_base(); sd = {k:v.clone() for k,v in m.state_dict().items()}
u = torch.load(sys.argv[1]+'/u.pt')['u']
c = float(sys.argv[2])
with torch.no_grad():
    m.voices.weight[2] = sd['voices.weight'][0]
    m.frame_voices.weight[2] = sd['frame_voices.weight'][1] + c*u
en = W.tts_texts('enroll'); idx = list(range(1,50,5))
A = [analyze(np.asarray(W.tts_synthesize(m, en[i], 2, 0))) for i in idx]
E = pickle.load(open(sys.argv[1]+'/env.pkl','rb'))
la = np.concatenate([np.log(a['sp'][a['v']]) for a in A]).mean(0)
lf = np.concatenate([np.log(a['f0'][a['v']]) for a in A]); print('logf0', lf.mean(), lf.std())
def trimmed(a):
    e = 10*np.log10(np.mean(a['sp'],1)+1e-20); act = np.where(e > e.max()-35)[0]; return (act[-1]-act[0])*0.005
print('dur', np.mean([trimmed(a) for a in A]))
f = np.linspace(0,12000,len(la)); db = lambda x: 20/np.log(10)*x
edges = [0,150,300,500,750,1000,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12001]
off = db((E['lt']-la)[(f>300)&(f<3000)].mean())
for a,b in zip(edges[:-1],edges[1:]):
    s=(f>=a)&(f<b); print(f'{a:6d}-{b:6d}  tgt-A {db(E["lt"][s].mean()-la[s].mean())-off:6.1f}   tgt-M2 {db(E["lt"][s].mean()-E["l1"][s].mean())-off:6.1f}')
ap = np.concatenate([a['ap'][a['v']] for a in A]).mean(0)
print('ap', [round(ap[int(fr/12000*(len(ap)-1))],3) for fr in (500,1000,2000,4000,8000)])
