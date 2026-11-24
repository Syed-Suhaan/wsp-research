import sys; sys.path.insert(0,'.')
import torch, numpy as np
d=torch.load('wsp/stage1_v2/W2/run_0/scratch/feats.pt'); f,z,y=d['f'],d['z'],d['y'].numpy()
names=['circle','square','triangle','star','cross','ring','ellipse','pentagon','arrow']
np.set_printoptions(precision=3, suppress=True, linewidth=200)
mu=torch.stack([f[y==c].mean(0) for c in range(9)])
# held-out analog: predict pentagon from square and ring
for lam in [0.348, 0.5]:
    p=(1-lam)*mu[1]+lam*mu[5]
    print('lam',lam,'dist pred->mu7', (p-mu[7]).norm().item(), 'cos', torch.cosine_similarity(p,mu[7],0).item())
    dd=torch.cdist(p[None],mu)[0]; print('  dist pred->class means', dd.numpy())
    # pentagon images: distance to pred vs nearest of other means
    dp=torch.cdist(f[y==7],p[None])[:,0]
    print('  pent img dist to pred pct', np.percentile(dp.numpy(),[5,50,95]), ' to own mean', np.percentile(torch.cdist(f[y==7],mu[7][None])[:,0].numpy(),[5,50,95]))
# also predict triangle->square->pentagon? interp sq from tri & pent: 1-cos(pi/n): tri .5, sq .293, pent .191
lam=(0.5-0.293)/(0.5-0.191); p=(1-lam)*mu[2]+lam*mu[7]
print('pred square from tri,pent lam',lam,'dist to mu1',(p-mu[1]).norm().item(),'dists',torch.cdist(p[None],mu)[0].numpy())
# base-model logits at the predicted point
sd=torch.load('checkpoints/stage1_v2_base.pt'); W=sd['head.weight'][:9]; b=sd['head.bias'][:9]
p=(0.652)*mu[1]+0.348*mu[5]; print('logits at pred pentagon (pentagon=7):', (W@p+b).numpy())
# pentagon distance-to-own-mean distribution per class
for c in range(9):
    r=torch.cdist(f[y==c],mu[c][None])[:,0].numpy(); print(names[c], 'own-dist pct', np.percentile(r,[50,95,99]))
