import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
d='wsp/stage1_v2/W2/run_4/scratch/'
f=np.load(d+'f.npy'); L=np.load(d+'L.npy'); y=np.load(d+'y.npy')
mu=np.stack([f[y==c].mean(0) for c in range(9)])
var=np.stack([f[y==c].var(0) for c in range(9)])
pooled=var.mean(0)
mr=(mu[5]+mu[6])/2
def lag(ts, t):
    w=[]
    for i,ti in enumerate(ts):
        p=1.0
        for j,tj in enumerate(ts):
            if j!=i: p*=(t-tj)/(ti-tj)
        w.append(p)
    return np.array(w)
def test(prev_cls, est, target_cls, varvec, name):
    delta=est-mu[prev_cls]
    w=delta/varvec; b=-w@(est+mu[prev_cls])/2
    llr=f@w+b
    print(name, 'Mahal sep %.2f'%np.sqrt((delta**2/varvec).sum()), 'dist %.2f'%np.linalg.norm(delta))
    for k in range(9):
        print('   class %d LLR mean %.1f sd %.1f  frac>0 %.3f'%(k, llr[y==k].mean(), llr[y==k].std(), (llr[y==k]>0).mean()))
# LOO: predict pentagon (t=.2) from tri(1/3), sq(1/4), round(0)
wts=lag([1/3,1/4,0],1/5); print('weights pent-from', wts)
est=np.clip(wts[0]*mu[2]+wts[1]*mu[1]+wts[2]*mr,0,None)
print('est-pent dist to real pent %.2f, sq->pent %.2f, cos(est-sq, pent-sq) %.2f'%(np.linalg.norm(est-mu[7]), np.linalg.norm(mu[7]-mu[1]), (est-mu[1])@(mu[7]-mu[1])/np.linalg.norm(est-mu[1])/np.linalg.norm(mu[7]-mu[1])))
test(1, est, 7, var[1], 'LOO pent vs sq, sq-var')
test(1, est, 7, pooled, 'LOO pent vs sq, pooled-var')
# LOO 2: predict square from tri, pent, round
wts=lag([1/3,1/5,0],1/4); print('weights sq-from', wts)
est=np.clip(wts[0]*mu[2]+wts[1]*mu[7]+wts[2]*mr,0,None)
print('est-sq dist to real sq %.2f'%np.linalg.norm(est-mu[1]))
# hexagon estimate
wts=lag([1/4,1/5,0],1/6); print('weights hex', wts)
hx=np.clip(wts[0]*mu[1]+wts[1]*mu[7]+wts[2]*mr,0,None)
np.save(d+'hex_est.npy',hx)
print('hex dist to classes', np.linalg.norm(mu-hx,axis=1))
test(7, hx, 9, var[7], 'HEX vs pent, pent-var')
test(7, hx, 9, pooled, 'HEX vs pent, pooled-var')
