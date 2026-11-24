import sys; sys.path.insert(0,'.')
import torch, numpy as np
from eval.wsp_tools import *
x,y=load_val_old(); k=k_shot_examples(4)
def show(img):
    for r in range(32): print(''.join(' .:#'[min(3,int(v)//64)] for v in img[r]))
    print('-'*32)
for img in k: show(img)
for i in np.where(y==7)[0][:3]: show(x[i])
for i in np.where(y==5)[0][:1]: show(x[i])
for i in np.where(y==6)[0][:1]: show(x[i])
