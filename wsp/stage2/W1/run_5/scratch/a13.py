import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from feat import *
import pickle
r = pickle.load(open('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/a3.pkl','rb'))
c = lambda k: r[k]['env'] - r[k]['env'].mean()
a = c('target')-c('v1v1'); b = c('v1v1')-c('v0v0'); e = c('ext0.5')-c('v1v1'); fe = c('frm_ext0.5')-c('v1v1')
cor = lambda x,y: np.corrcoef(x,y)[0,1]
print('corr(target-M2, M2-F2) %.2f'%cor(a,b), ' corr(target-M2, ext0.5-M2) %.2f'%cor(a,e), ' corr(target-M2, frmext-M2) %.2f'%cor(a,fe))
# restrict to 300 Hz..8 kHz (above the F0 region)
sel = MELF > 300
print('above 300 Hz: %.2f %.2f %.2f'%(cor(a[sel],b[sel]), cor(a[sel],e[sel]), cor(a[sel],fe[sel])))
print('rms dB target-M2 %.2f, ext0.5-M2 %.2f'%(8.686/2*np.sqrt(np.mean(a**2)), 8.686/2*np.sqrt(np.mean(e**2))))
