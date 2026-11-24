import numpy as np, sys
sys.path.insert(0,'.')
from eval.wsp_tools import load_val_old
x,y=load_val_old()
for c in [7,5,3]:
  for r in x[y==c][:2]:
    print(c); print('\n'.join(''.join('#' if v>128 else ('+' if v>64 else '.') for v in row) for row in r[4:28]))
