import numpy as np, sys
sys.path.insert(0,'.')
from eval.wsp_tools import load_val_old
from PIL import Image
k=np.load('wsp/stage1_v2/W1/run_7/scratch/k.npy')
x,y=load_val_old()
rows=[k[0],k[1]]+[x[y==7][i] for i in range(4)]+[x[y==c][0] for c in range(9)]
im=np.concatenate([np.pad(r,2,constant_values=128) for r in rows],1)
Image.fromarray(im).resize((im.shape[1]*6,im.shape[0]*6),Image.NEAREST).save('wsp/stage1_v2/W1/run_7/scratch/k.png')
for r in k:
  print('\n'.join(''.join('#' if v>128 else ('+' if v>64 else '.') for v in row) for row in r)); print()
