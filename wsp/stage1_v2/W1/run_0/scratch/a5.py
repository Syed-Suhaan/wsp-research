import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base(); sd=base_state_dict()
x,y=load_val_old(); xk=k_shot_examples(0)
with torch.no_grad():
    F=m.features(to_tensor(x)); Fk=m.features(to_tensor(xk)); L=m(to_tensor(x)); Lk=m(to_tensor(xk))
yt=torch.tensor(y)
mu=torch.stack([F[yt==c].mean(0) for c in range(9)])
var=torch.stack([F[yt==c].var(0) for c in range(9)]); vp=var.mean(0)
sd7=F[yt==7].std(0); muk=Fk.mean(0)
zk=(Fk-mu[7])/sd7
agree=(torch.sign(zk[0])==torch.sign(zk[1]))
mn=torch.where(agree, torch.minimum(zk[0].abs(),zk[1].abs())*torch.sign(zk[0]), torch.zeros(128))
dirs={'lda_pooled':(muk-mu[7])/vp,'lda7':(muk-mu[7])/var[7],'agree':mn/sd7}
marg=L[:,7:8]-L  # logit7 - logit_c
gap=(L.max(1).values-L[:,7])  # how much top exceeds logit7
for n,w in dirs.items():
    w=w/w.norm()
    p=F@w; pk=Fk@w
    m7=p[yt==7].mean(); s7=p[yt==7].std()
    u=(p-m7)/s7; uk=(pk-m7)/s7
    print(n,'k in sigma',uk.numpy().round(2))
    for c in range(9):
        print('   ',c,'u pct',np.percentile(u[yt==c].numpy(),[5,50,95]).round(2),'gap pct',np.percentile(gap[yt==c].numpy(),[1,5,50]).round(2))
    for t in [1.0,1.28,1.5]:
        print('  t',t,'pent>t',int((u[yt==7]>t).sum()), 'k>t', (uk>t).tolist())
    # leakage for others with cap G
    for G in [1,2,4]:
        leak=[int(((u>1.28)&(gap<G)&(yt!=7)&(yt==c)).sum()) for c in range(9)]
        print('  G',G,'leak per class (u>1.28, gap<G)',leak)
