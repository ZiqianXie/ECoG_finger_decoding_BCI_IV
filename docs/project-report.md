# Project report: late reimplementation of ECoG finger-trajectory decoding

**Report date:** 4 September 2026

**Original study:** Xie, Schwartz, and Prasad (2018),
[*Decoding of finger trajectory from ECoG using deep learning*](https://doi.org/10.1088/1741-2552/aa9dbe)

## Executive summary

This project is a late reimplementation and extension of the first author's
2018 ECoG finger-trajectory decoding work. The original source code was lost
when the author's laptop hard drive failed during a move. The present repository
was therefore rebuilt in 2026 from the paper, the public BCI Competition IV
Data Set 4 recordings, and methodological recollection. It is not the original
code, and the results should not be described as a bit-for-bit reproduction.

The reconstruction recovered the main modeling ideas: a trainable spatial
filter initialized by FastICA; a three-level, dilated, biorthogonal wavelet tree;
40 ms band-energy bins; and temporal decoding. It also made three important
changes. First, every data-derived choice is fit on a chronological training or
validation partition. Second, each subject and finger may use a different
validated decoder, avoiding competition between five output heads. Third, model
quality is judged from held-out trajectory shape as well as Pearson correlation.

The reconstructed raw-test `Hist-4` PCC is 0.549 for S1, 0.413 for S2, and
0.637 for S3. S2 and S3 reach or exceed the aggregate references reconstructed
from the paper's rounded per-finger table. S1 does not reach the paper figure's
0.56 aggregate: the remaining gap is 0.011. Across all fifteen subject-finger
pairs, eight reconstructed PCCs exceed the rounded paper CNN-LSTM values.

PCC alone overstates one result. The S1 ring output has PCC 0.612, but its
movement-state recall is only 0.054 and its movement-peak amplitude ratio is
0.127. Visual inspection confirms a nearly flat prediction. It must therefore
be treated as unresolved, not as a successful reconstruction.

## Scope and research goals

The work began as an attempt to recover the 2018 pipeline after repeated code
requests. Once it became clear that the original implementation could not be
recovered, the goal changed from historical replication to a transparent,
better-documented reimplementation that asks:

1. Can the paper's fixed wavelet-energy representation still support useful
   finger decoding with modern, leakage-controlled training?
2. Does a modern state-space or linear-attention model improve on an LSTM?
3. Should all five fingers share a decoder, or should model choice be made per
   subject and per finger?
4. Can target correction remove glove drift and finger coupling without
   fabricating movement on the wrong finger?
5. Do numerical improvements correspond to visibly better movement timing,
   amplitude, and rest behavior?

The primary goal is a physiologically credible trajectory. PCC is a useful
proxy, but it is not allowed to override obvious visual failure.

## Data and split

The project uses BCI Competition IV Data Set 4. The public files are not
redistributed here. Each subject has 400,000 training ECoG samples and 200,000
released test samples at 1 kHz. Glove trajectories have five columns and are
released on the same 1 kHz grid before downsampling.

| Subject | Raw ECoG channels | Removed physical channels | Retained channels |
|---|---:|---|---:|
| S1 | 62 | 55 | 61 |
| S2 | 48 | 21, 38 | 46 |
| S3 | 64 | 49, 50 | 62 |

Channel 49 for S3 follows the paper. Channel 50 was added after a raw-data audit
found a test-only burst with more than 250-fold variance inflation; it looks
normal in training and cannot be detected by a training-only variance screen.
This is an explicit departure from the paper and is recorded in the loader.

The training recording is divided chronologically. The first two-thirds of the
training file are available for fitting and the last third is validation. On
the complete train-plus-released-test timeline, this is approximately 44% model
fit, 22% validation, and 33% final test. The test labels are used only after a
candidate and its hyperparameters have been frozen. The public summaries include
test metrics for reporting, but no selector is permitted to read them.

## ECoG preprocessing

### Bad channels and line noise

The primary path removes the channels listed above and applies narrow IIR notch
filters at 60, 120, and 180 Hz. This decision was made after spectral audits of
the released recordings and is consistent with the expectation that the ECoG
needs power-line rejection. Retained channels are standardized using means and
scales fitted only on the model-fitting portion of the training recording.

### Spatial initialization

FastICA is fit separately for every subject and only on its training partition.
Its unmixing matrix initializes a bias-free 1x1 spatial convolution. A CUDA
implementation is available for speed, while scikit-learn remains the fixed
feature reference. Early LassoLars experiments selected ICA-band features per
finger. Joint pruning was not imposed on the final five-output model because a
component weak for one finger can still be important for another. Later paths
instead allow separate finger models or structured feature selection.

## Glove target preprocessing

### Baseline removal

The paper-like target uses a constrained lower-envelope baseline. In compact
form, it minimizes an L1 distance from baseline to observed trajectory plus a
squared first-difference smoothness penalty, with the baseline constrained to
remain below the trajectory. The paper reconstruction uses a tension parameter
of `1e5`.

A single global smoothness value did not follow all local resting-level changes.
The audit therefore also creates rolling lower-envelope variants at explicitly
named time scales (for example `local_w2_q10`: a two-second window and tenth
percentile). The local window and quantile are selected from chronological
validation decoding, not from test labels.

Every primary score in this report is calculated against the original released
test glove trajectory. A cleaned target is used only to visualize movement
morphology and calculate diagnostic quantities such as state F1.

### Why winner-take-all correction was rejected

Ring and little fingers are mechanically coupled, so a small ring displacement
can accompany intended little-finger motion. The initial reconstruction tried
to assign each event to one winning finger and remove estimated coupling. This
created a decisive failure: activity that belonged to the index finger could
appear as corrected little-finger movement during a period when the raw little
finger did not move.

Consequently, the primary pipeline retains all five baseline-subtracted glove
channels and does not apply winner-take-all reassignment. Event-corrected targets
remain only as an experimental comparison. This preserves real co-movement and
avoids inventing categorical intent not present in the released measurements.

## Wavelet-energy front end

The reconstructed front end follows the paper's core structure:

- biorthogonal `bior6.8` analysis filters;
- three levels of an undecimated wavelet-packet tree;
- 2, 4, and 8 filters at dilations 1, 2, and 4;
- 17 active coefficients, omitting the inert zero coefficient remembered from
  the original implementation;
- zero-padding for the exact fixed-feature reproduction;
- scaled-tanh activation after each layer; and
- L2 energy followed by `log1p` in non-overlapping 40 ms bins.

The output rate is therefore 25 Hz, matching the downsampled glove trajectory.
Each decoded step sees the preceding 25 energy bins, corresponding to one second
of history. A dedicated frequency-response audit measures every initialized
branch before training; this is necessary because correct coefficient values can
still be wired into an incorrect tree or dilation pattern.

The 100-step/four-second block in the legacy implementation was a Theano
static-graph limitation. It is not retained as a biological context limit. The
modern code can train over complete contiguous sequences or use chunks only to
manage memory.

## Temporal and spatial model audit

The following families were implemented or benchmarked:

- fixed wavelet/ICA energy with ridge or LARS-style sparse regression;
- LSTM and GRU;
- a parallel diagonal state-space baseline;
- causal linear attention;
- native Mamba where its optional dependency is available;
- causal dilated TCN;
- movement-versus-rest CSP in multiple frequency bands; and
- beta-gated high-gamma amplitude heads.

Modern sequence models were not assumed to be superior. On this small dataset,
the diagonal SSM, linear-attention, and Mamba experiments did not consistently
beat the simpler recurrent or TCN models. The strongest S1/S2 route used
finger-specific fixed-feature LSTMs with exact end-to-end wavelet refinement for
selected fingers. S3 benefited from movement-versus-rest CSP and a shape-aware
TCN, then a validation-selected blend with beta-gated/high-gamma components.

Separate per-subject models are mandatory because channel geometry differs.
Separate per-finger models were also allowed: sharing one spatial and spectral
representation across all five outputs can cause competition, especially when
their signal-to-noise ratios and useful bands differ.

CUDA was used for feature generation and neural training. Supported training
paths request `torch.compile(mode="reduce-overhead")`. Keeping the complete
small dataset in device memory reduces transfer overhead, but wall time is still
affected by wavelet feature construction, repeated validation, and per-finger
candidate sweeps.

## Model selection and metrics

All model and ensemble choices are made independently for each finger using the
chronological validation partition. A missing finger prediction is represented
as `NaN` and cannot win selection. Fixed hyperparameters inherited from the
paper, such as a 100-epoch refit audit, are fixed before test evaluation.

Primary reporting uses:

- per-finger Pearson correlation on the raw released test glove;
- macro average over all five fingers (`Macro-5`);
- historical competition average over thumb, index, middle, and little
  (`Hist-4`; ring excluded); and
- RMSE on the raw target.

Visual/morphology diagnostics use a baseline-corrected target only as an audit:

- movement-state precision, recall, and F1;
- RMS prediction during rest;
- derivative PCC;
- movement-window peak-amplitude ratio;
- peak-triggered waveform correlation and lag; and
- plotted movement and rest windows.

This distinction matters. Correlation is scale-invariant and can reward a small,
nearly invisible waveform if its timing happens to align with the target.

## Results

### Per-finger raw-test PCC

Paper values below are the rounded CNN-LSTM numbers reported in the 2018 paper.
They are a historical reference, not high-precision targets.

| Subject | Finger | Paper CNN-LSTM | Reimplementation | Difference |
|---|---|---:|---:|---:|
| S1 | Thumb | 0.750 | 0.696 | -0.054 |
| S1 | Index | 0.790 | 0.809 | +0.019 |
| S1 | Middle | 0.170 | 0.296 | +0.126 |
| S1 | Ring | 0.600 | 0.612 | +0.012 |
| S1 | Little | 0.470 | 0.395 | -0.075 |
| S2 | Thumb | 0.620 | 0.599 | -0.021 |
| S2 | Index | 0.380 | 0.472 | +0.092 |
| S2 | Middle | 0.270 | 0.208 | -0.062 |
| S2 | Ring | 0.470 | 0.495 | +0.025 |
| S2 | Little | 0.300 | 0.373 | +0.073 |
| S3 | Thumb | 0.740 | 0.711 | -0.029 |
| S3 | Index | 0.550 | 0.508 | -0.042 |
| S3 | Middle | 0.460 | 0.637 | +0.177 |
| S3 | Ring | 0.410 | 0.676 | +0.266 |
| S3 | Little | 0.750 | 0.693 | -0.057 |

The reimplementation is higher on 8/15 pairs. This count is descriptive; the
rounded paper values do not support a fine-grained statistical superiority
claim.

### Aggregate raw-test PCC

| Subject | Macro-5 | Hist-4 | Historical reference | Interpretation |
|---|---:|---:|---:|---|
| S1 | 0.561 | 0.549 | figure aggregate 0.560 | 0.011 below |
| S2 | 0.429 | 0.413 | rounded per-finger Hist-4 0.393 | above |
| S3 | 0.645 | 0.637 | rounded per-finger Hist-4 0.625 | above |

For S1, the mean of the four rounded paper entries is 0.545, but the paper's
aggregate chart shows 0.56. The chart value is the more conservative reference.
The present result has therefore **not** surpassed the reported S1 aggregate.

### Morphology audit

| Subject | Macro rest RMS | Macro state F1 | Macro derivative PCC | Main concern |
|---|---:|---:|---:|---|
| S1 | 0.082 | 0.439 | 0.285 | Ring is nearly flat; middle has heavy false activity |
| S2 | 0.059 | 0.447 | 0.239 | Middle amplitude and state precision remain weak |
| S3 | 0.128 | 0.677 | 0.341 | Better cycles, but some rest leakage remains |

The S1 ring discrepancy is the clearest example of why the visual audit is part
of the acceptance criterion:

| S1 ring diagnostic | Value |
|---|---:|
| Raw-test PCC | 0.612 |
| Cleaned-target PCC | 0.588 |
| Movement-state recall | 0.054 |
| Movement-state F1 | 0.101 |
| Movement peak ratio | 0.127 |
| Rest RMS | 0.052 |

The low rest error is achieved largely by predicting very little. That is not a
useful ring-finger decoder.

![Subject 1 movement windows](figures/s1-movement-windows.png)

![Subject 1 peak-triggered average](figures/s1-peak-triggered.png)

![Subject 2 movement windows](figures/s2-movement-windows.png)

![Subject 3 movement windows](figures/s3-movement-windows.png)

### Seed stability

Repeated fixed-feature LSTM runs showed small aggregate seed variation: S1
`Macro-5` SD 0.0047 and `Hist-4` SD 0.0053 across three seeds; S2 `Macro-5` SD
0.0067 and `Hist-4` SD 0.0055 across two seeds. These small aggregate SDs do not
imply that every finger is stable or morphologically correct. They also are not
sampling standard errors and should not be interpreted as confidence intervals.

## What did not work

### Winner-take-all target assignment

This confused mechanical co-movement with intended movement and created
activity on fingers that were stationary in the raw glove record. It was removed
from the final paths.

### Hard movement binarization

An explicit movement-state target can help a recurrent network recognize
stationary-to-moving transitions, but hard gating created unnatural step-to-zero
artifacts. Soft state-aware objectives and gates were retained only when they
improved the validation morphology score.

### Post-hoc amplitude gain

Increasing gain reduced some peak-amplitude errors and could improve RMSE, but
it also amplified uncertain bumps during rest. A lower-RMSE S3 calibration was
rejected as primary because the plots showed worse rest leakage.

### One architecture for every subject and finger

The same SSM, LSTM, or CSP configuration did not win everywhere. S3 responds
well to CSP plus a TCN, whereas S1/S2 are better served by finger-specific fixed
features and selective end-to-end refinement. The final selector therefore does
not force architectural uniformity for its own sake.

### Unregularized 100-epoch full-data refit

The original paper used 100 epochs, so a final audit retrained the selected S1
fixed-feature LSTM for 100 epochs on the combined fit and validation partitions.
Validation performance had already peaked around epoch 1–5 and then declined.
The fixed 100-epoch refit produced raw-test PCC
`0.593/0.719/0.053/0.346/0.308` and `Hist-4` 0.418, far below the selected
early-stopped model's 0.549. The old epoch count is therefore not portable to
the present optimizer and full-data protocol; this candidate was rejected.

### PCC-only selection

S1 ring demonstrates the failure mode: scale-invariant correlation can remain
high even when movement amplitude and state recall collapse. Future selection
must make morphology an explicit constraint rather than a secondary narrative.

## Reproduction recipes

Install the project and place the official competition files as described in
the top-level README. Then:

```bash
export PYTHONPATH=src

# Validate inputs and preprocess all subjects.
python scripts/audit_dataset.py
python scripts/preprocess_dataset.py --subjects 1 2 3
python scripts/prepare_paper_baseline_targets.py --subjects 1 2 3
python scripts/compare_target_baselines.py --subjects 1 2 3

# Confirm the initialized wavelet tree before training.
python scripts/audit_wavelet_frequency_response.py

# Recreate the frozen S1 per-finger selection.
python scripts/select_per_finger_ensemble.py --subject 1 \
  --prepared-root outputs/preprocessed_v2 --history 25 \
  --method stable=outputs/paper_gap_ensemble_v1/sub1 \
  --method e2e_index=outputs/exact_e2e_s1_index_h40_v1/sub1 \
  --output outputs/paper_reproduction_s1_v1/sub1

# Recreate the frozen S2 per-finger selection.
python scripts/select_per_finger_ensemble.py --subject 2 \
  --prepared-root outputs/preprocessed_v2 --history 25 \
  --method h40=outputs/s2_lstm_sweep_h40_s3/sub2 \
  --method e2e_index=outputs/exact_e2e_idx_h40_t100_v1/sub2 \
  --method e2e_middle=outputs/exact_e2e_mid_h20_t50_v1/sub2 \
  --output outputs/paper_reproduction_s2_v1/sub2

# Generate held-out morphology panels.
python scripts/diagnose_prediction_morphology.py --subject 1 \
  --prepared-root outputs/preprocessed_v2 --target local_w2_q10 \
  --method reproduction=outputs/paper_reproduction_s1_v1/sub1 \
  --output outputs/paper_reproduction_visual_v1/sub1

python scripts/diagnose_prediction_morphology.py --subject 2 \
  --prepared-root outputs/preprocessed_v2 --target local_w1_q10 \
  --method reproduction=outputs/paper_reproduction_s2_v1/sub2 \
  --output outputs/paper_reproduction_visual_v1/sub2

python -m pytest -q
```

Some selection commands consume previously trained candidate directories. The
candidate-generating scripts and their arguments remain in `scripts/`; the
compact summaries under `docs/results/` identify the selected methods. A future
release should add a manifest that maps every public result to a complete command,
configuration hash, environment lock, and source commit.

## Reproducibility boundaries

This repository improves transparency but does not erase the limitations of a
late reconstruction:

- the original code and exact legacy environment are unavailable;
- paper values are rounded, and not every old training detail is recoverable;
- the released test labels were examined during this research process, even
  though each individual selector is coded to use validation only;
- the current public package excludes raw data and large trained artifacts;
- several experiment scripts reflect research exploration rather than one
  polished end-to-end command; and
- S1 aggregate performance and S1 ring morphology remain unresolved.

For these reasons, the project should be cited as a reimplementation and
extension, not as the official source code accompanying the 2018 publication.

## Recommended next work

1. Replace PCC-only selection with a validation constraint on movement-state
   recall and peak-amplitude ratio, then reselect S1 ring without test access.
2. Build a fully scripted experiment manifest from raw files to every reported
   summary, including hashes and package versions.
3. Repeat the selected S1 and S2 models over at least five seeds and report both
   aggregate and per-finger variability.
4. Evaluate whether a ring-specific spatial/spectral front end can preserve real
   ring/little coupling while recovering ring amplitude.
5. Add test-label-blind cross-validation within the training recording before
   any further comparison to released test labels.

## Public-release policy

The repository contains code, configuration, tests, documentation, compact JSON
summaries, and derived diagnostic figures. It excludes:

- competition recordings and true-label files;
- `.remote`, credentials, environment variables, and machine-specific paths;
- checkpoints, prediction arrays, logs, caches, and temporary files; and
- the locally downloaded paper or supplement.

Contributors should preserve this boundary in every pull request.
