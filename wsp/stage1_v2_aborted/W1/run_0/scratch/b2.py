import numpy as np
d='wsp/stage1_v2/W1/run_0/scratch/'
fo=np.load(d+'fo.npy'); fk=np.load(d+'fk.npy'); y=np.load(d+'y.npy'); lo=np.load(d+'lo.npy')
r=lo[:,:9].max(1)/fo.sum(1)
print('max old logit / sum f: pctiles 50,99,max', np.percentile(r,[50,99,100]).round(3))
