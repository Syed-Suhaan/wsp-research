import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy'); lo=np.load(d+'lo.npy')
np.set_printoptions(precision=2, suppress=True, linewidth=200)
mu=np.stack([fo[y==c].mean(0) for c in range(9)])
w=np.concatenate([fo[y==c]-mu[c] for c in range(9)]); var=w.var(0)+1e-2
var7=fo[y==7].var(0)+1e-2
def report(wv, name, ks):
    p=fo@wv; pk=fk@wv
    print(name, 'k proj', pk.round(2), ' per-class max', [p[y==c].max().round(2) for c in range(9)], ' 99pct', [np.percentile(p[y==c],99).round(2) for c in range(9)])
mk=fk.mean(0)
report((mk-mu[7])/var7,'diag-LDA vs7 var7',None)
report((mk-mu[7])/var,'diag-LDA vs7 pooled',None)
report((mk-mu.mean(0))/var,'diag-LDA vs allmean pooled',None)
# leave one out
for i in range(2):
    wv=(fk[i]-mu[7])/var
    report(wv,f'from k{i} only vs7 pooled',None)
    wv=(fk[i]-mu[7])/var7
    report(wv,f'from k{i} only vs7 var7',None)
# norm-ratio feature: f / |f| 
print('norms by class', [np.linalg.norm(fo[y==c],axis=1).mean().round(2) for c in range(9)], np.linalg.norm(fk,axis=1))
print('norms class7 range', np.percentile(np.linalg.norm(fo[y==7],axis=1),[1,50,99]))
