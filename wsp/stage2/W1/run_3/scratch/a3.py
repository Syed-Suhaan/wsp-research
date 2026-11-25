import sys; sys.path.insert(0,'.'); sys.path.insert(0,sys.argv[1])
import numpy as np, pickle
out=pickle.load(open(sys.argv[1]+'/enroll_feats.pkl','rb'))
E={k:np.mean([d['env_v'] for d in out[k]],0) for k in out}
M={k:np.mean([d['mc_v'] for d in out[k]],0) for k in out}
fr=np.arange(513)*24000/1024
bands=[0,150,300,500,700,1000,1400,2000,2500,3000,3500,4000,5000,6000,8000,10000,12000]
for k in E: E[k]=E[k]-E[k][(fr>100)&(fr<4000)].mean()
print('band   ', ' '.join(f'{b:>6}' for b in bands[:-1]))
for k in E:
    vals=[E[k][(fr>=bands[i])&(fr<bands[i+1])].mean() for i in range(len(bands)-1)]
    print(str(k).ljust(6), ' '.join(f'{v:6.2f}' for v in vals))
d10=M[0]-M[1]; dt1=M['t']-M[1]
print('mc diff norms (excl c0): F2-M2',np.linalg.norm(d10[1:]),' T-M2',np.linalg.norm(dt1[1:]),' T-F2',np.linalg.norm((M['t']-M[0])[1:]))
print('cos(T-M2, F2-M2) excl c0', np.dot(d10[1:],dt1[1:])/np.linalg.norm(d10[1:])/np.linalg.norm(dt1[1:]))
print('M2 mc',M[1][:12].round(2)); print('T  mc',M['t'][:12].round(2)); print('F2 mc',M[0][:12].round(2))
