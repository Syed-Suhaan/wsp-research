import sys, json; sys.path.insert(0,'.')
import torch; torch.set_num_threads(4)
from eval import wsp_tools as W
r=W.tts_evaluate_val(sys.argv[1],"wsp/stage2/W1/run_6",seed=0)
print(json.dumps({k:v for k,v in r.items() if k!='per_sentence'},indent=1))
for row in r['per_sentence']: print(row)
