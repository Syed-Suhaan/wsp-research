"""Pipeline sanity check for Stage 2 trainer revision 2 (criterion: results/stage2/sanity_criterion.json).

Trains B1 (voice rows only, lr 0.01, 300 steps, seed 0) from the base toward F2's own audio of
the 50 enrollment sentences, then measures on the 20 VALIDATION sentences how much of the
M2-to-F2 gap is closed in speaker similarity and in MCD.

Output: results/stage2/sanity_v2.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from eval.stage2_audio import compare, world_features  # noqa: E402
from eval.stage2_metrics import Evaluator, sentences, synthesize  # noqa: E402
from models.tiny_voca3 import load_state  # noqa: E402
from train.stage2_annotate import annotate  # noqa: E402
from train.stage2_finetune import train  # noqa: E402


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    base = load_state()
    enroll, val = sentences()["enroll"], sentences()["val"]
    f2_enroll = [synthesize(base, t, 0, 0) for t in enroll]
    items = annotate("sanity_F2_enroll", enroll, f2_enroll)
    ev = Evaluator(asr=False)
    centroid = torch.nn.functional.normalize(torch.stack([ev.embed(w) for w in f2_enroll]).mean(0), dim=-1)
    refs = [synthesize(base, t, 0, 0) for t in val]
    ref_feats = [(world_features(r), len(r)) for r in refs]

    def score(model, voice):
        outs = [synthesize(model, t, voice, 1) for t in val]
        return {"speaker_sim_to_F2": statistics.mean(float(ev.embed(o) @ centroid) for o in outs),
                "mcd_to_F2_db": statistics.mean(compare(o, *rf)["mcd_db"] for o, rf in zip(outs, ref_feats))}

    start, f2_self = score(base, 2), score(base, 0)
    model = load_state()
    losses = []
    snaps = train(model, "voice", 0.01, 300, 0, [300], annotations=items, loss_log=losses)
    model.load_state_dict(snaps[300])
    trained = score(model, 2)
    closed_sim = (trained["speaker_sim_to_F2"] - start["speaker_sim_to_F2"]) / (f2_self["speaker_sim_to_F2"] - start["speaker_sim_to_F2"])
    closed_mcd = (start["mcd_to_F2_db"] - trained["mcd_to_F2_db"]) / (start["mcd_to_F2_db"] - f2_self["mcd_to_F2_db"])
    passed = closed_sim >= 0.70 and closed_mcd >= 0.70
    out = {"criterion": "results/stage2/sanity_criterion.json", "start_voice2_M2": start, "F2_itself_seed1": f2_self,
           "trained_voice2": trained, "gap_closed_speaker_sim": closed_sim, "gap_closed_mcd": closed_mcd,
           "pass": passed, "loss_first_10_mean": statistics.mean(losses[:10]), "loss_last_10_mean": statistics.mean(losses[-10:]),
           "wall_clock_s": time.time() - t0}
    (ROOT / "results" / "stage2" / "sanity_v2.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
