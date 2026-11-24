OK, continue with Task 1.5, with these changes first:

1. Held-out shapes: replace ONE of the two held-out shapes with a shape that is visually close to a hexagon. Keep the other one. Keep both names out of every prompt and writer-readable file. This changes test data only; do not change the frozen config or the calibration. Commit this before any condition runs, and record in results/stage1_v2/heldout_note.json (not writer-readable) that one held-out shape is near the new class and one is far. Report the held-out false-positive rate separately for the near shape and the far shape.

2. Reporting: in addition to the gate, report 95% bootstrap CIs for (W1 - B1) and (W2 - B1) on new-class accuracy, class-9 precision, and old-class drop. Do not change the gate.

3. Disclosure: in RESULTS.md and the README, state that the calibration selected the setup with the lowest B0 validation accuracy out of 9 attempts, and give the selection rule.

4. LICENSE: create LICENSE with the Apache-2.0 text, copyright "2026 Syed Suhaan".

5. Effort: before the first WSP run, confirm the effort level that a subagent uses and record it. If you cannot confirm it, write "unverified".
