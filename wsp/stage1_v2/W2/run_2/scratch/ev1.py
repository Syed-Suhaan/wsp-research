import sys, json, torch
sys.path.insert(0,'.')
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
from eval.wsp_tools import evaluate_val
print(json.dumps(evaluate_val("checkpoints/stage1_v2_W2_run2.pt", "wsp/stage1_v2/W2/run_2")))
