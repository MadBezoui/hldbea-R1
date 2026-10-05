# Provenance of the v3 runs

All 3,801 v3 runs record `dirty: true`. The runner at that time stored the
commit and a single dirty flag, and untracked outputs (logs, figures and
reports written during the campaigns) are enough to set that flag. This note
records what can be reconstructed, without claiming more.

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
which the files were edited; the v3 metadata cannot prove it on their own.

From the next campaign onward, the runner also records whether tracked sources
differ from the commit, the SHA-256 of that diff and a digest of every Python
file under `src/`.
