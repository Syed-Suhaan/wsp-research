import torch, numpy as np
d=torch.load('wsp/stage1_v2/W1/run_4/scratch/feats.pt')
F,Fk,y,L,Lk=d['F'],d['Fk'],d['y'],d['L'],d['Lk']
U=F/F.norm(dim=1,keepdim=True); Uk=Fk/Fk.norm(dim=1,keepdim=True)
m7=U[y==7].mean(0); mk=Uk.mean(0); w=mk-m7; m7n=m7/m7.norm()
print('|w|',w.norm().item(),'w.m7n',(w@m7n).item())
c=U[y==7]@m7n; print('pentagon cos to m7n', c.mean().item(), c.std().item(), 'k cos', (Uk@m7n).numpy())
# score in u space
s7=U[y==7]@w; sk=Uk@w
print('u-space s pent mean/std', s7.mean().item(), s7.std().item(), 'k', sk.numpy())
# linear version: s'(f)= w.f - t*(m7n.f); choose t s.t. threshold in u-space at mean7 + z*std
for z in [0.9,1.0,1.1]:
    tu=s7.mean()+z*s7.std()
    t=tu/c.mean()
    sp=F@w - t*(F@m7n); spk=Fk@w - t*(Fk@m7n)
    out=[]
    for cl in range(9):
        out.append(f'{cl}:{(sp[y==cl]>0).float().mean().item()*100:.1f}%')
    print('z',z,'t',t.item(),'frac>0',out,'k',spk.numpy().round(3), 'pent sp std', sp[y==7].std().item())
# star: logit3-logit7 margin
for cl in [3,5,0,6,8]:
    m=(L[y==cl,cl]-L[y==cl,7]); print('class',cl,'logit_c - logit7 pct',np.percentile(m.numpy(),[1,5,50]).round(2))
