# Model release

This release freezes one model structure for every BCI Competition IV
subject/finger pair. Selection follows two rules:

1. the selected model must have strictly higher nested development OOF PCC than
   its own untuned initialization; and
2. a route is either one model or an equal-weight ensemble of random-seed
   refits with the same structure.

The released test labels were not used to choose a family, hyperparameter,
update schedule, seed, or ensemble member. Released-test PCC below is therefore
descriptive only. Exact values, artifact checks, and evidence paths are in
[`results/canonical-model-audit.json`](results/canonical-model-audit.json); the
authoritative route registry is
[`../configs/canonical_models.yaml`](../configs/canonical_models.yaml).

## All 15 routes

| Pair | Selected structure | Release | Init OOF PCC | Tuned OOF PCC | Net gain | Test PCC |
|---|---|---:|---:|---:|---:|---:|
| S1 thumb | `single_wavelet_paper_lstm_movement2` | 6 same-structure seeds | 0.577876 | 0.581144 | +0.003268 | 0.733166 |
| S1 index | `single_wavelet_paper_lstm_movement1` | 6 same-structure seeds | 0.721337 | 0.722476 | +0.001139 | 0.758646 |
| S1 middle | `overcomplete_depth5_designed_csp_residual_lstm` | single model | 0.314679 | 0.354988 | +0.040308 | 0.285477 |
| S1 ring | `future16_fullbank_powercal_ridge` | single model | 0.516520 | 0.537785 | +0.021265 | 0.656667 |
| S1 little | `single_wavelet_paper_lstm_tails4x4` | 6 same-structure seeds | 0.434134 | 0.468194 | +0.034060 | 0.454162 |
| S2 thumb | `future16_synergy_selected_residual_lstm` | 10 same-structure seeds | 0.620222 | 0.686693 | +0.066471 | 0.578797 |
| S2 index | `single_wavelet_standard_lstm_movement1` | 6 same-structure seeds | 0.349421 | 0.349980 | +0.000559 | 0.377221 |
| S2 middle | `single_wavelet_standard_lstm_movement1` | 6 same-structure seeds | 0.368628 | 0.431985 | +0.063357 | 0.209691 |
| S2 ring | `future16_fullbank_raw_ridge` | single model | 0.496554 | 0.523094 | +0.026540 | 0.554921 |
| S2 little | `single_wavelet_standard_lstm_movement1` | 6 same-structure seeds | 0.326494 | 0.390478 | +0.063984 | 0.393156 |
| S3 thumb | `current_bin_residual_lstm_batch48` | single model | 0.629315 | 0.732715 | +0.103400 | 0.734348 |
| S3 index | `synergy_spoc_allfinger_continuous_residual_lstm` | single model | 0.641432 | 0.680209 | +0.038777 | 0.618644 |
| S3 middle | `lead4_hidden128_selected_residual_lstm` | 9 same-structure seeds | 0.615626 | 0.631198 | +0.015572 | 0.643700 |
| S3 ring | `single_wavelet_standard_lstm_movement1` | 6 same-structure seeds | 0.404736 | 0.422119 | +0.017384 | 0.557341 |
| S3 little | `single_wavelet_residual_lstm_movement1` | 6 same-structure seeds | 0.631943 | 0.649982 | +0.018039 | 0.617019 |

All 15 routes pass the positive-tuning rule. The smallest gain is +0.000559
(S2 index); the largest is +0.103400 (S3 thumb). The `Release` column identifies
single models and same-structure random-seed ensembles.

## Descriptive comparison with the 2018 paper

The paper values are rounded values reported in the publication, so the
comparison is descriptive rather than an exact reanalysis.

| Subject | 2018 paper Macro-5 | 2026 release Macro-5 | Difference |
|---|---:|---:|---:|
| S1 | 0.556 | 0.577624 | +0.021624 |
| S2 | 0.408 | 0.422758 | +0.014758 |
| S3 | 0.582 | 0.634211 | +0.052211 |

All three subject-level Macro-5 scores exceed the rounded paper values. Seven
of the 15 individual finger scores exceed their rounded paper counterparts.
Model selection used nested development OOF predictions.

## Reproducing the release audit

Run from the repository root after the ignored `outputs/` artifacts have been
restored or reproduced:

```bash
PYTHONPATH=experiments/scripts:scripts:src \
python experiments/scripts/audit_canonical_models.py \
  --manifest configs/canonical_models.yaml \
  --require-release-pass \
  --output docs/results/canonical-model-audit.json
```

The audit verifies the complete 15-pair set, a strictly positive development
OOF gain, released-test isolation, artifact existence, member counts, member
seeds, and same-structure membership. Checkpoints and cached feature arrays
remain in ignored `outputs/` directories because they are large generated
artifacts; the registry, refit programs, and exact audit are versioned.
