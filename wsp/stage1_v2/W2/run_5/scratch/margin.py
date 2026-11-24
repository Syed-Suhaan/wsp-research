import torch, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
D=torch.load('wsp/stage1_v2/W2/run_5/scratch/feat.pt'); f,L,y,mu=D['f'],D['L'],D['y'],D['mu']
names=['circ','sq','tri','star','cross','ring','ell','pent','arrow']
l=L[:,:9].clone(); top=l.max(1).values; gap=(top-l[:,7]).numpy()
for k in range(9):
    g=gap[(y==k).numpy()]
    print(names[k],'gap top-l7 pct 0,1,5,10,50',np.percentile(g,[0,1,5,10,50]), 'n gap<3',(g<3).sum(),'n gap<2',(g<2).sum())
