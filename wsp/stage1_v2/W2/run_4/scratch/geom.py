import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
d='wsp/stage1_v2/W2/run_4/scratch/'
f=np.load(d+'f.npy'); L=np.load(d+'L.npy'); y=np.load(d+'y.npy')
mu=np.stack([f[y==c].mean(0) for c in range(9)])
un=lambda v: v/np.linalg.norm(v)
a = mu[7]-mu[1]; b=mu[1]-mu[2]
mr=(mu[5]+mu[6])/2
c = mr-mu[7]
print('cos(pent-sq, sq-tri)=%.2f'%(un(a)@un(b)))
print('cos(pent-sq, round-pent)=%.2f'%(un(a)@un(c)))
print('cos(pent-sq, ring-pent)=%.2f  ellipse-pent=%.2f'%(un(a)@un(mu[5]-mu[7]), un(a)@un(mu[6]-mu[7])))
print('|tri-sq| |sq-pent| |pent-ring| |pent-ell|', np.linalg.norm(b), np.linalg.norm(a), np.linalg.norm(mu[5]-mu[7]), np.linalg.norm(mu[6]-mu[7]))
for name,u in [('poly axis',un(a)),('round dir',un(c)),('ring dir',un(mu[5]-mu[7])),('ell dir',un(mu[6]-mu[7]))]:
    p=f@u; p0=mu[7]@u
    print(name, ' '.join('%d:%.2f±%.2f'%(k,(p[y==k]-p0).mean(),(p[y==k]-p0).std()) for k in range(9)))
# per-dimension: features where pentagon differs from square and the trend
print('top dims of pent-sq:', np.argsort(-np.abs(a))[:15])
print('mu tri sq pent ring ell on those dims')
for k in np.argsort(-np.abs(a))[:15]:
    print(k, mu[[2,1,7,5,6,0],k])
