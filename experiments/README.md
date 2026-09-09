# Experimental archive

This directory preserves ablations, alternative decoders, diagnostic scripts,
compact result records, and historical figures. They document the research
process but are not required to reproduce the result in the main README.

The main release uses the same single-path 1 kHz interpolated-wavelet topology
for all 15 subject/finger pairs. Development folds select the CSP
initialization-bank size and LSTM implementation recorded in
`configs/final_single_wavelet_routes.yaml`. The CSP-count ablation and
distributed-fold merge utilities that support those selections remain here for
auditability. The final learning-rate audit also remains here: it compares
`1e-4`, `3e-5`, and `1e-5` on development folds while allowing the untouched
LARS-initialized LSTM as a zero-update candidate. Other archived models may
score better for individual fingers, but models developed under heterogeneous
or test-informed diagnostic protocols are not mixed into the headline result.
The development-only comparison and selected routes are recorded in
`results/final-lstm-lr-selection.json` and
`configs/final_lstm_lr_selected_routes.yaml`.

```text
configs/    settings for alternative and diagnostic runs
scripts/    ablations, benchmarks, and superseded pipelines
results/    compact records from those runs
figures/    figures supporting the detailed project report
tests/      tests for archived experimental components
```

Run an archived script from the repository root with both script directories on
the import path:

```bash
PYTHONPATH=experiments/scripts:scripts:src \
  python experiments/scripts/<script>.py ...
```

Archived experiment tests can be run explicitly with the same import path:

```bash
PYTHONPATH=experiments/scripts:scripts:src \
  python -m pytest -q experiments/tests
```

The primary pipeline remains in `scripts/`; its reusable implementation is in
`src/ecog_decoding/`. The current result record is
`docs/results/final-single-branch-six-seed.json`.
