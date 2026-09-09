# Experimental archive

This directory preserves ablations, alternative decoders, diagnostic scripts,
compact result records, and historical figures. They document the research
process but are not required to reproduce the result in the main README.

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

The primary pipeline remains in `scripts/`; its reusable implementation is in
`src/ecog_decoding/`. The current result record is
`docs/results/final-single-branch-six-seed.json`.
