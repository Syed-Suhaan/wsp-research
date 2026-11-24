import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy'); lo=np.load(d+'lo.npy')
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mu=np.stack([fo[y==c].mean(0) for c in range(9)]); sd=np.stack([fo[y==c].std(0) for c in range(9)])
def cos(a,b): return a@b.T/np.linalg.norm(a,axis=-1,keepdims=True)/np.linalg.norm(b,axis=-1,keepdims=True).T
print('cos k vs class means\n', cos(fk,mu))
print('cos k0 k1', cos(fk[:1],fk[1:]))
# cosine of each old sample to k mean
mk=fk.mean(0)
c=cos(fo,mk[None])[:,0]
for cl in range(9): print(cl, 'cos to mk: mean %.3f max %.3f'%(c[y==cl].mean(), c[y==cl].max()))
print('cos k to mk', cos(fk,mk[None])[:,0])
# fraction of active dims
print('active dims k', (fk>0.05).sum(1), 'old mean active', (fo>0.05).sum(1).mean())
# pooled within-class var
w=np.concatenate([fo[y==c]-mu[c] for c in range(9)]); s=w.std(0)+1e-3
z=(fk-mu.mean(0))/s
print('z k0 top', np.argsort(-z[0])[:15], z[0][np.argsort(-z[0])[:15]])
print('z k1 top', np.argsort(-z[1])[:15], z[1][np.argsort(-z[1])[:15]])
print('max logit old samples by class', [lo[y==cl].max(1).mean().round(1) for cl in range(9)])
