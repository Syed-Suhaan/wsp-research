import torch, numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
d=torch.load('wsp/stage1_v2/W1/run_1/scratch/feats.pt'); F,y,Fk=d['F'],d['y'],d['Fk']
F7=F[y==7]; mu7=F7.mean(0); sd7=F7.std(0)+1e-3
z=(Fk-mu7)/sd7
zm=z.mean(0)
o=torch.argsort(-zm.abs())
print("top dims by |z| vs pentagon")
for j in o[:25].tolist():
    print(j, "k:",Fk[:,j].numpy(), "mu7 %.2f sd7 %.2f max7 %.2f min7 %.2f | allold max %.2f min %.2f"%(mu7[j],sd7[j],F7[:,j].max(),F7[:,j].min(),F[:,j].max(),F[:,j].min()), "z",z[:,j].numpy())
# whole old set
mu=F.mean(0); sd=F.std(0)+1e-3
