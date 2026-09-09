# Experimental archive

This directory preserves ablations, alternative decoders, diagnostic scripts,
compact result records, and historical figures. They document the research
process but are not required to reproduce the result in the main README.

The main release uses the same single 1 kHz interpolated-wavelet model for all
15 subject/finger pairs. Some archived models score better for individual
fingers, but they were developed under heterogeneous or test-informed
diagnostic protocols and are not mixed into the headline result.

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
