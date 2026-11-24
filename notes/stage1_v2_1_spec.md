The Stage 1 v2 gate stays FAIL. Do not change the gate and do not add WSP runs. Now do Stage 1 v2.1 (baseline fairness) and a CLAUDE.md restructure, then stop.

1. Tuned baselines (no new WSP runs):
- B1t: fine-tune the class-9 head row and bias. Grid: lr {1e-3, 3e-3, 1e-2, 3e-2} x steps {200, 500, 1000} x negatives per step {59, 236}.
- B2t: fine-tune the whole model with replay. Grid: lr {1e-4, 3e-4, 1e-3} x steps {200, 500, 1000} x positive weight {1, 10, 30}.
- B0b: B0 with the class-9 bias chosen from {-3, -2, -1, -0.5, 0, 0.5}.
- Selection: one global setting per condition, chosen by the mean over seeds 0-7 on the VALIDATION split, excluding the k examples. Objective: mean of new-class accuracy and class-9 precision, subject to old-class drop <= 2 points. Fix this rule in a commit before any grid result is seen.
- Then evaluate each chosen setting once on TEST, seeds 0-7.
- Keep the old B1 and B2 rows. Add B1t, B2t, and B0b rows to the main table.
- Report 95% bootstrap CIs (10,000 resamples, 8 pairs) for W1 - B1t, W1 - B2t, W1 - B0b, and W2 - B1t on new-class accuracy, class-9 precision, and old-class drop.
- State in RESULTS.md: each baseline had a tuning budget of its full grid on validation; each WSP writer had at most 3 validation evaluations.

2. CLAUDE.md restructure:
- Move every stage-specific detail to notes/stage1_spec.md and notes/stage2_spec.md: class names, held-out notes, v1 pilot details, Stage 2 target style and its DSP parameters, and condition details.
- CLAUDE.md keeps only: the goal in one neutral sentence, machine, general rules, WSP rules, repo layout, export task, and this line: "The orchestrator reads notes/<current stage>_spec.md. WSP writers must not read notes/."
- After the change, CLAUDE.md must contain no class name, no shape name, and no Stage 2 target value. Show me the new CLAUDE.md in full before you commit it, and wait for my OK.

3. Export: stop removing lines from prompts/. Exported prompt files must be byte-identical to the files the writers used. Remove the README note about the removed line.

4. Effort: keep "unverified". In RESULTS.md, record both the session setting and the probe value.

5. Rebuild public_release/, run the reproduction check, push, and stop.
