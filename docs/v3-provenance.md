# Provenance of the v3 runs

All 3,801 confirmatory and calibration v3 runs record `dirty: true`. The
runner at that time stored the commit and a single dirty flag, and untracked
outputs (logs, figures and reports written during the campaigns) are enough to
set that flag. This note records what can be reconstructed, without claiming
more.

- 497 runs record commit `20c54634674217912e93e4b2e279d3a647a184bb`: the three
  calibration screens and the first part of the rotation block.
- 3,304 runs record commit `67648ae4b13f024ecec6ea77f8fb1c05e59001bf`.

Between these two commits only `README.md`, `requirements-lock.txt`,
`experiments/run_reviewer_queue_v3.sh` and `src/hldbea/design_audit.py`
change. The last file checks the calibration design after the runs and is not
imported by the optimizers. No algorithm, operator, problem or metric module
differs between the two commits.

During the campaigns the only uncommitted tracked change was the
`design_audit.py` edit later committed in `67648ae`. The executed optimization
code is therefore the one of commit `20c5463`, which is also contained in every
later commit. This statement rests on the commit history and on the order in
which the files were edited, and the v3 metadata cannot prove it on their own.

## Replays with the current code

The duplicate counters were added after these campaigns. They only read the
population, so `experiments/replay_duplicates_v3.sh` re-executed every HLDBEA
run of the five confirmatory blocks (1,860 runs) from its manifest
specification and compared the final decision and objective arrays with the
archived ones. All replays matched bit for bit, which also shows that the
current code reproduces the archived HLDBEA runs. The reports
`results/reports/*-validation-v3-duplicates.json` keep, for every run, the
duplicate counters and the SHA-256 of the archived arrays, and the manuscript
generator refuses a report that does not match the archive.

## The follow-up DTLZ3 ablation

`experiments/manifests/reviewer-dtlz3-duplicates-v3.yaml` (270 runs) varies
how duplicates are handled: kept (the default), not created by rejected
refinement calls (`refinement_duplicates: drop_rejected`), or removed from the
merged set before scoring (`eliminate`). It was designed after the
confirmatory blocks had been analyzed. Its three default configurations
reproduce the corresponding runs of the confirmatory DTLZ3 block bit for bit.
Its runs record the source fingerprint
`512b03dbdeafe2a0ce646bac84809de7c69ec9611666a90a8dbb23c5da256dba`, which is the
fingerprint of `src/` at commit `9001fbf` of this repository. Later commits
only add a digest helper to `src/hldbea/artifacts.py` that the runs did not use.
`analysis/generation_matched.py` replays four of its configurations with a
checkpoint every 460 evaluations to compare them at an equal number of
generations, with the same bit-for-bit check.

## Source fingerprints

From the follow-up ablation onward, the runner records the SHA-256 of every
imported Python module (`source_sha256`), whether the code runs from a Git
checkout or from an installed package. In a checkout it also records the
commit, whether tracked sources differ from it and the SHA-256 of that diff.
Outside Git the commit is reported as unavailable and the fingerprint is kept.
