# Bounded prediction test runs

Canonical configuration, methods, test status and result links:
[threshold comparison notes](../../docs/prediction_threshold_tests.md).

Timestamped runs are created here as `threshold_YYMMDD_HHMM_<job-id>/` (UTC).
The 200-frame comparison saves one final all-instance SLP per condition, with
optional body-first track IDs. No frame-chunk prediction files are written.
Do not overwrite completed runs or the existing full-video v007 predictions.

The [1,000-frame continuation/grace tests](../../docs/prediction_grace_tests.md)
use `grace_YYMMDD_HHMM_<job-id>/`: two final SLPs, a whole-selection raw-candidate
NPZ, and validation/association tables. No frame-chunk SLPs or interpolated poses.
