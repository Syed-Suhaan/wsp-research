import sys; sys.path.insert(0,'.')
import numpy as np, torch
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import load_val_old
x,y=load_val_old()
print(x.shape, np.bincount(y), x.dtype, x.min(), x.max())
from PIL import Image
rows=[]
for c in range(9):
    idx=np.where(y==c)[0][:12]
    rows.append(np.concatenate([np.pad(x[i],1,constant_values=128) for i in idx],1))
im=np.concatenate(rows,0)
Image.fromarray(im).resize((im.shape[1]*3,im.shape[0]*3),Image.NEAREST).save(sys.argv[1])
