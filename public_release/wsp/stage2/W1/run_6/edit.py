"""Stage 2 WSP edit: give voice slot 2 the target speaker's voice by writing only row 2 of the
frame-level voice embedding table `frame_voices.weight` of TinyVoca3. Every shared tensor,
rows 0/1 of both voice tables, and the token-level table `voices.weight` stay unchanged, so
voices 0 (F2) and 1 (M2) are untouched and voice 2 keeps M2's speaking rate.

Mechanism (measured on the base model, see rationale.md):
  * `frame_voices.weight` controls pitch and timbre; `voices.weight` controls only durations.
  * The difference fv1 - fv0 (M2 row minus F2 row) is the model's female -> male axis.
  * The target is a male speaker with a lower pitch than M2 (median F0 about 106 Hz; M2 about
    164 Hz; F2 about 261 Hz).

Edit: extend the F2 -> M2 axis beyond M2,
    frame_voices.weight[2] := fv0 + alpha * (fv1 - fv0),
with alpha the point where the model's predicted mean log F0 (voiced frames), taken as linear in
alpha between the two trained voices, equals the target's pitch in the model's units:
    goal  = p_M2 + ln(F0_target / F0_M2_audio)
    alpha = (goal - p_F2) / (p_M2 - p_F2)
Measured inputs (all 50 enrollment sentences, WORLD Harvest at 24 kHz, F0 range 50-600 Hz, seed 0;
script wsp/stage2/W1/run_6/scratch/cand1.py):
    F0_target = 106.3 Hz   median F0 of the target recordings
    F0_M2     = 163.9 Hz   median F0 of the base model's voice 1 audio
    p_F2      = 0.260      predicted mean log(F0/200 Hz), frame row = F2's, token row = M2's
    p_M2      = -0.178     same with frame row = M2's
    alpha     = 1.9884 (the full-precision value of that computation is written below)
The statistics are computed once and written here as explicit numbers so that the saved state
dict is exactly the one that was checked on validation data.

Run from the repository root:  .venv/bin/python wsp/stage2/W1/run_6/edit.py
"""

import torch

torch.set_num_threads(4)

BASE = "checkpoints/stage2_base.pt"
OUT = "checkpoints/stage2_W1_run6.pt"

state_dict = torch.load(BASE, map_location="cpu", weights_only=True)

# Rows of the frame-level voice table: 0 = F2 (female), 1 = M2 (male), 2 = new voice (copy of M2).
fv0 = state_dict["frame_voices.weight"][0].clone()
fv1 = state_dict["frame_voices.weight"][1].clone()

# alpha = (goal - p_F2) / (p_M2 - p_F2), goal = p_M2 + ln(F0_target / F0_M2): see the docstring.
ALPHA = 1.988375186920166

# New voice: the female -> male axis followed past M2 until the pitch reaches the target's.
# alpha = 1 would be M2 itself; alpha ~ 2 places voice 2 as far beyond M2 as M2 is beyond F2.
state_dict["frame_voices.weight"][2] = fv0 + ALPHA * (fv1 - fv0)

# Unchanged: voices.weight (all rows; row 2 stays M2's, so durations are M2's) and every
# shared tensor; frame_voices.weight rows 0 and 1.
torch.save(state_dict, OUT)
print("saved", OUT, "alpha", ALPHA)
