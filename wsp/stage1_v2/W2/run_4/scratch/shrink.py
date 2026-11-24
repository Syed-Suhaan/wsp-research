import numpy as np
d='wsp/stage1_v2/W2/run_4/scratch/'
f=np.load(d+'f.npy'); y=np.load(d+'y.npy'); hx=np.load(d+'hex_est.npy')
mu=np.stack([f[y==c].mean(0) for c in range(9)]); v7=f[y==7].var(0)
mr=(mu[5]+mu[6])/2
for lam in [0,0.1,0.5,1,3,1e6]:
    vv=v7+lam*v7.mean()
    w=(hx-mu[7])/vv; bb=-w@(hx+mu[7])/2
    p=(f@w+bb)[y==7]; dprime=((w@hx+bb)-p.mean())/p.std()
    top=np.sort(np.abs(w*(hx-mu[7])))[::-1]; conc=top[:10].sum()/top.sum()
    # LOO analog with same lam: pent est from tri,sq,round vs sq
    est=np.clip(-0.36*mu[2]+1.28*mu[1]+0.08*mr,0,None); est=np.clip(np.array([-1/3*1.08,1.28,0.08])[0]*0+est,0,None)
    v1=f[y==1].var(0)+lam*f[y==1].var(0).mean()
    w2=(est-mu[1])/v1; b2=-w2@(est+mu[1])/2
    q=(f@w2+b2); qs=q[y==1]; qp=q[y==7]; th=qs.mean()+2*qs.std()
    print('lam %g: hex-est d-prime %.2f, top10 conc %.2f | LOO: real-pent d-prime %.2f, frac pent>th(2sd) %.3f, frac sq>th %.3f'%(lam,dprime,conc,(qp.mean()-qs.mean())/qs.std(),(qp>th).mean(),(qs>th).mean()))
