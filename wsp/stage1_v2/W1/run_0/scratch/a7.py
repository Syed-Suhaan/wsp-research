import sys; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import *
m=load_base()
x,y=load_val_old(); xk=k_shot_examples(0)
with torch.no_grad(): F=m.features(to_tensor(x)); Fk=m.features(to_tensor(xk))
yt=torch.tensor(y)
F7=F[yt==7]; mu7=F7.mean(0); v7=F7.var(0); muk=Fk.mean(0)
w=(muk-mu7)/(v7+v7.median())
p=F@w; s7=F7@w; m=s7.mean(); s=s7.std()
u=(p-m)/s; uk=(Fk@w-m)/s; t=((muk@w)-m)/s/2
print('uk',uk, 't',t, 'pent>t', int((u[yt==7]>t).sum()))
for c in range(9): print(c, np.percentile(u[yt==c].numpy(),[5,50,95]).round(2))
