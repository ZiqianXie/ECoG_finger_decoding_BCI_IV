# Experimental archive

This directory preserves ablations, alternative decoders, diagnostic scripts,
compact result records, and historical figures. They document the research
process but are not required to reproduce the result in the main README.

The current release freezes one development-selected structure for each of the
15 subject/finger pairs. This archive retains the earlier matched single-path
1 kHz interpolated-wavelet family because it supports controlled comparisons
across all subjects and fingers. Development folds selected the CSP
initialization-bank size and LSTM implementation recorded in the historical
`configs/final_single_wavelet_routes.yaml`. The CSP-count ablation and
distributed-fold merge utilities that support those selections remain here for
auditability. The historical learning-rate audit also remains here: it compares
`1e-4`, `3e-5`, and `1e-5` on development folds while allowing the untouched
LARS-initialized LSTM as a zero-update candidate. A separate development-only
screen compares zero-initialized residual LSTM and GRU heads at `3e-4` and
`1e-4`; it promotes the residual LSTM only for S3 little. Other archived models
may score better for individual fingers, but models developed under
heterogeneous or test-informed diagnostic protocols are not mixed into the
headline result.
The development-only comparison and selected routes are recorded in
`results/final-lstm-lr-selection.json` and
`configs/final_lstm_lr_selected_routes.yaml`. The residual screen is recorded
in `results/residual-recurrent-selection.json` and
`configs/residual_recurrent_selected_routes.yaml`.

## Cross-subject mechanistic audit

The [single-wavelet interpretation record](docs/single-wavelet-mechanistic-interpretation.md)
uses exact registered electrode coordinates and all 90 frozen matched-family
checkpoints. It reports spatial concentration against a 3-D coordinate
permutation null, decoder-weighted spectral support, seed stability, and filter
movement from initialization. The versioned numerical summary is
[`results/single-wavelet-mechanisms-all-subjects.json`](results/single-wavelet-mechanisms-all-subjects.json).

The relevant scripts are:

- `scripts/infer_competition_channel_map.py`
- `scripts/analyze_single_wavelet_mechanisms.py`
- `scripts/render_single_wavelet_mechanisms_all_subjects.py`

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
`src/ecog_decoding/`. The current release record is
`docs/results/canonical-model-audit.json`.
