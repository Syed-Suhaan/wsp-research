import torch, numpy as np
d = torch.load('wsp/stage1_v2/W1/run_3/scratch/feats.pt')
F, y, Fk = d['F'], d['y'], d['Fk']
mu = torch.stack([F[y==c].mean(0) for c in range(9)]); sd = torch.stack([F[y==c].std(0) for c in range(9)])
muk = Fk.mean(0)
# z of each k example per class 7 and 5 and 6
for c in [7,5,6,1]:
    z = (Fk - mu[c])/(sd[c]+1e-3)
    idx = torch.argsort(-(z.min(0).values.abs()*torch.sign(z[0])*torch.sign(z[1]).clamp(min=0)))[:12]
    print('class', c)
    zz = (Fk - mu[c])/(sd[c]+1e-3)
    # dims where both k examples agree in sign; sort by min |z|
    agree = (torch.sign(zz[0])==torch.sign(zz[1]))
    score = torch.where(agree, zz.abs().min(0).values, torch.zeros(128))
    top = torch.argsort(-score)[:12]
    for i in top: print('  dim',int(i),'z',np.round(zz[:,i].numpy(),2),'mu_c',round(mu[c,i].item(),3),'k',np.round(Fk[:,i].numpy(),3))
# dims where k examples are high and all old class means low
gmax = mu.max(0).values
ratio = Fk.min(0).values - gmax
top = torch.argsort(-ratio)[:15]
print('dims where min(k) - max class mean is largest')
for i in top:
    q = torch.stack([F[y==c][:,i].quantile(0.99) for c in range(9)])
    print(int(i), 'k', np.round(Fk[:,i].numpy(),3), 'maxmean', round(gmax[i].item(),3), 'argmax', int(mu[:,i].argmax()), 'q99 max over classes', round(q.max().item(),3), int(q.argmax()))
# dims where k low but class 7 high
r2 = mu[7] - Fk.max(0).values
top = torch.argsort(-r2)[:10]
print('dims where k below class7')
for i in top:
    print(int(i), 'k', np.round(Fk[:,i].numpy(),3), 'mu7', round(mu[7,i].item(),3), 'sd7', round(sd[7,i].item(),3), 'q01_7', round(F[y==7][:,i].quantile(0.01).item(),3))
