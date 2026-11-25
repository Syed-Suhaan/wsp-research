import sys; sys.path.insert(0, '/home/user/wsp-research/wsp/stage2/W1/run_5/scratch')
from feat import *
import pickle
res = pickle.load(open('/home/user/wsp-research/wsp/stage2/W1/run_5/scratch/a3.pkl','rb'))
np.set_printoptions(precision=1, suppress=True, linewidth=200)
print('freq ', MELF.astype(int))
for k in ['target','v0v0','v1v1','ext0.5']:
    e = res[k]['env']; print('%-7s'%k, (e - e.mean())*10/np.log(10))
