from eval.wsp_tools import *
import numpy as np
xk=k_shot_examples(0); x,y=load_val_old()
def show(img):
    for r in img: print(''.join(' .:-=+*#%@'[min(9,v//26)] for v in r))
for img in xk: show(img); print('='*32)
print('pentagon example'); show(x[y==7][0]); show(x[y==7][1])
print('pixel stats k', xk.mean((1,2)), 'old', x.mean())
