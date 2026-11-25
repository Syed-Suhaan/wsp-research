import numpy as np
d=np.load('/home/user/wsp-research/wsp/stage2/W1/run_0/scratch/a2.npz')
SR=24000; f=np.arange(513)*SR/1024
for lo,hi in [(100,4000),(100,5000),(200,3500)]:
    s=(f>=lo)&(f<hi)
    c={k:(np.exp(d['ls_'+k][s])*f[s]).sum()/np.exp(d['ls_'+k][s]).sum() for k in ['T','0','1']}
    print(lo,hi,{k:round(v) for k,v in c.items()}, 'ratio a=%.3f'%((c['T']-c['1'])/(c['0']-c['1'])))
    # log-centroid
    c={k:(np.exp(d['ls_'+k][s])*np.log(f[s])).sum()/np.exp(d['ls_'+k][s]).sum() for k in ['T','0','1']}
    print('  logcentroid a=%.3f'%((c['T']-c['1'])/(c['0']-c['1'])))
