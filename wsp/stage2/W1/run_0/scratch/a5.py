import numpy as np
d=np.load('/home/user/wsp-research/wsp/stage2/W1/run_0/scratch/a2.npz')
SR=24000; f=np.arange(513)*SR/1024
for lo,hi in [(100,4000),(100,5000),(200,3500),(100,8000)]:
    s=(f>=lo)&(f<hi)
    L={k:d['ls_'+k][s]-d['ls_'+k][s].mean() for k in ['T','0','1']}
    t=L['T']-L['1']; g=L['0']-L['1']
    print(lo,hi,'proj %.3f cos %.3f'%((t@g)/(g@g),(t@g)/np.linalg.norm(t)/np.linalg.norm(g)))
