# Contributing

Thank you for helping improve this late reimplementation.

## Before opening a change

1. Keep all model selection inside the labeled development recording. Use
   complete-event, per-subject/per-finger folds with the required history purge,
   and refit label-dependent baselines, CSP, and feature selection inside each
   fold. Do not choose a method, gain, lag, route, seed, or ensemble weight from
   released-test labels.
2. Report all five fingers and clearly distinguish `Macro-5` from competition
   `Hist-4`, which excludes ring.
3. Inspect trajectory plots and morphology metrics. A PCC increase is not enough
   when movement amplitude, state recall, or rest behavior becomes implausible.
4. Add or update synthetic tests for reusable preprocessing or model code.
5. Run `python -m pytest -q` before submitting a pull request.

## Data and privacy boundary

Never commit competition recordings, true labels, model checkpoints, prediction
arrays, `.remote`, credentials, logs, temporary files, or machine-specific
absolute paths. Compact aggregate summaries and derived figures may be included
when they do not contain restricted data or identifying local metadata.

## Reproducible experiment reports

For a new result, record the subject, finger, target variant, event-fold
assignment, purge, seed, complete command, relevant configuration, selection
criterion, raw-test metrics, and visual/morphology audit. State explicitly
whether any released-test information affected selection. Distinguish
across-seed standard deviation from a statistical standard error.
