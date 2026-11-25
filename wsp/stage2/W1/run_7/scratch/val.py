import sys, json; sys.path.insert(0,'/home/user/wsp-research')
import torch; torch.set_num_threads(4)
from eval import wsp_tools as W
r = W.tts_evaluate_val('checkpoints/stage2_W1_run7.pt', 'wsp/stage2/W1/run_7', 0)
print(json.dumps({k:v for k,v in r.items() if k!='per_sentence'}, indent=1))
for s in r['per_sentence']: print(s)
