import sys, json, torch; sys.path.insert(0,'.'); torch.set_num_threads(4)
from eval import wsp_tools as W
r = W.tts_evaluate_val(sys.argv[1], 'wsp/stage2/W1/run_2', seed=0)
print(json.dumps({k:v for k,v in r.items() if k!='per_sentence'}, indent=1))
for p in r['per_sentence']: print(p)
