import sys, json; sys.path.insert(0, '.')
import torch; torch.set_num_threads(4)
from eval import wsp_tools as T
r = T.tts_evaluate_val('checkpoints/stage2_W1_run4.pt', 'wsp/stage2/W1/run_4', seed=0)
print(json.dumps({k: v for k, v in r.items() if k != 'per_sentence'}, indent=1))
for p in r['per_sentence'][:5]: print(p)
