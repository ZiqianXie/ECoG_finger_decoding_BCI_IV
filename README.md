# ECoG finger-trajectory decoding

Late reimplementation and extension of:

> Z. Xie, O. Schwartz, and A. Prasad, “Decoding of finger trajectory from
> ECoG using deep learning,” *Journal of Neural Engineering*, 15(3), 036009,
> 2018. [doi:10.1088/1741-2552/aa9dbe](https://doi.org/10.1088/1741-2552/aa9dbe)

## Important provenance note

This repository is **not the original 2018 source release**. The original code
was lost when the first author's laptop hard drive failed during a move. This
implementation was rebuilt in 2026 from the published paper, the public BCI
Competition IV data, and the author's methodological recollection. It should be
read as a late, independent reimplementation and research continuation—not as
an archival recovery or a claim of bit-for-bit reproduction.

The reconstruction deliberately tests newer alternatives where the old design
can be improved. It retains the paper's biorthogonal wavelet initialization,
energy-binning idea, FastICA spatial initialization, and recurrent comparison,
while adding stricter leakage controls, per-finger model selection, modern
sequence backbones, explicit zero-phase notch filtering at 60 Hz and its 120/180
Hz harmonics, a nonnegative output-domain constraint, and visual trajectory
diagnostics.

The full methods, experiment history, numerical tables, limitations, and visual
diagnosis are in the [project report](docs/project-report.md).

## Retrospective development results

The primary score is Pearson correlation against the released, unmodified test
glove trajectories. `Macro-5` is the mean across all five fingers, matching the
aggregate reported in the paper. Paper values are rounded CNN-LSTM results from
2018.

**Interpretation warning:** the reconstructed rows below are retrospective
development results, not an untouched confirmatory benchmark. Validation traces
were consulted repeatedly during reconstruction, and released-test trajectories
were later inspected for morphology and failure analysis. The numbers remain
useful for documenting what the current code can reproduce, but they must not be
treated as unbiased estimates of model-selection performance. The separately
frozen blocked-fold follow-up is reported in the next section. It prevents new
selection leakage, but it cannot retroactively make the chronological
validation segment pristine because earlier development had already inspected
it.

| Subject | Result | Thumb | Index | Middle | Ring | Little | Macro-5 |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | Paper | 0.750 | 0.790 | 0.170 | 0.600 | 0.470 | 0.556 |
| S1 | Non-stacked per-finger baseline, unconstrained | 0.696 | 0.809 | 0.296 | 0.612 | 0.395 | 0.561 |
| S1 | Exploratory stacked system + output projection | 0.730 | 0.809 | 0.308 | 0.618 | 0.426 | **0.578** |
| S1 | Above + validation-selected separate-stem index ensemble | 0.730 | 0.810 | 0.308 | 0.618 | 0.426 | **0.579** |
| S2 | Paper | 0.620 | 0.380 | 0.270 | 0.470 | 0.300 | 0.408 |
| S2 | Selected per-finger system + middle-finger seed ensemble | 0.599 | 0.472 | 0.391 | 0.495 | 0.373 | **0.466** |
| S3 | Paper | 0.740 | 0.550 | 0.460 | 0.410 | 0.750 | 0.582 |
| S3 | Selected per-finger system + output projection | 0.720 | 0.525 | 0.632 | 0.666 | 0.687 | **0.646** |

Within this retrospective development comparison, the selected systems improve
the five-finger aggregate for all three subjects
and exceed the paper value on nine of fifteen individual fingers. S1's highest
number is explicitly exploratory: thumb and little are learned second-stage
stacks over candidate predictions, middle uses its selected base model, ring
uses a separately calibrated base prediction, and index averages six
independently fine-tuned trainable-wavelet models selected on validation. Each
index member has its own FastICA-initialized spatial projection, wavelet stem,
and LSTM head; three validation-best checkpoints occur before stem unfreezing
and three after it. This is not a single end-to-end model and is therefore shown
beside the non-stacked baseline. Numerical scores are complemented by held-out
trajectory plots and morphology diagnostics in the project report.

## Leakage-controlled blocked-CV follow-up

A later audit used three rolling blocked folds entirely inside the official
training partition. Candidate family, ensemble membership, output calibration,
and refit epoch count were fixed from those folds. The selected components were
then refit on the full training partition for fixed epoch counts, and the
chronological validation segment was evaluated once. Released-test results are
descriptive only. This is a stronger protocol than the retrospective table,
although the validation segment cannot be called historically pristine.

| Subject | Validation thumb | Index | Middle | Ring | Little | Validation Macro-5 |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.525 | 0.851 | 0.398 | 0.384 | 0.264 | **0.485** |
| S2 | 0.507 | 0.200 | 0.356 | 0.460 | 0.250 | **0.355** |
| S3 | 0.733 | 0.541 | 0.617 | 0.450 | 0.414 | **0.551** |

| Subject | Descriptive released-test Macro-5 | Hist-4 | Paper Macro-5 |
|---|---:|---:|---:|
| S1 | 0.440 | 0.437 | 0.556 |
| S2 | 0.356 | 0.317 | 0.408 |
| S3 | 0.615 | 0.650 | 0.582 |

The blocked-fold scores used for selection were much higher than the final
chronological result, particularly for S1 and S2. This rejects promotion of the
new ensemble as a general replacement for the retrospective systems and points
to temporal nonstationarity as the main unresolved problem. Finger-confusion
assignment remained diagonal for all three subjects, so a global label swap is
not the explanation, although ring/little and index/middle coupling remains
visible.

Two output domains are saved explicitly. `*_raw_coordinate.npy` is calibrated
to the released glove coordinate system for paper-comparable PCC.
`*_cleaned.npy` is a smooth nonnegative flexion trajectory: its gain is fitted
through the origin using only inner-fold data. Keeping these arrays separate
prevents a raw-coordinate negative baseline from being plotted against a
baseline-corrected target. The cleaned outputs have test rest RMS of 0.068,
0.055, and 0.113 for S1--S3, but visual review still shows missed or
under-amplitude S1/S2 movements. See the
[project report](docs/project-report.md#leakage-controlled-blocked-cv-follow-up)
and [`docs/results/nested-cv-followup.json`](docs/results/nested-cv-followup.json).

The filter terminology separates the **spectral** and **spatial** stages. A
trainable-wavelet route updates both the bior6.8 wavelet taps and the
FastICA-initialized spatial projection by gradient descent. A fixed-wavelet
route freezes both. A fixed-bandpass+CSP route instead uses conventional fixed
frequency bands and training-estimated, then frozen, CSP spatial filters.
Learning only the downstream LSTM/TCN does not make the spectral filters
trainable.

| Final route | Subject/fingers | Spectral front end | Spatial front end |
|---|---|---|---|
| Separate-stem wavelet ensemble | S1 index | Six independently gradient-trained bior6.8 wavelet stems | Six independently trained, FastICA-initialized projections |
| Trainable wavelet | S2 index | Gradient-trained bior6.8 wavelet taps | Gradient-trained, FastICA-initialized projection |
| Seed-averaged asymmetric wavelet | S2 middle | Six equal-weight trainable wavelet/LMP models | Six FastICA-initialized projections |
| Fixed wavelet | S1 ring; S2 thumb, ring, little | Frozen bior6.8 wavelet taps | Frozen FastICA projection |
| Fixed bandpass+CSP | S1 middle; all S3 fingers | Frozen conventional bandpass filters | CSP estimated on training data, then frozen |
| Fixed-only ensemble | S1 thumb | Mixture of fixed-wavelet and fixed-band candidates | Mixture of fixed FastICA/CSP/SPoC candidates |
| Mixed ensemble | S1 little | Fixed candidates plus one trainable-wavelet candidate | Mixed fixed and gradient-trained candidates |

For S1 thumb the trainable-wavelet candidate received zero stack weight,
whereas S1 little retained a small standardized weight of 0.0465. The complete
machine-readable routing audit is in
[`docs/results/learned-filter-map.json`](docs/results/learned-filter-map.json).
This table reports the frozen final routing, not every validation benefit. In
particular, the later S2-thumb end-to-end run improved validation PCC from
0.600 to 0.630 but reduced held-out test PCC from 0.599 to 0.579; it is reported
as a sensitivity result rather than silently selected using the test labels.

![Subject 1 held-out movement windows](docs/figures/s1-movement-windows.png)

Machine-readable summaries are versioned in [`docs/results`](docs/results).
Raw recordings, released test labels, checkpoints, predictions, and working
outputs are intentionally excluded from the repository.

## Data

Use [BCI Competition IV, Data Set 4](https://www.bbci.de/competition/iv/), whose
data remain governed by the competition's terms. Download the competition files
and true labels from the official site, then arrange them as:

```text
data/raw/bci_competition_iv_ds4/
├── mat/
│   ├── sub1_comp.mat
│   ├── sub2_comp.mat
│   └── sub3_comp.mat
└── true_labels/
    ├── sub1_testlabels.mat
    ├── sub2_testlabels.mat
    └── sub3_testlabels.mat
```

No competition data are distributed in this repository. The loader validates
the expected shapes before any experiment runs.

## Method in brief

ECoG is processed at 1 kHz with documented bad-channel removal, narrow notches
at 60/120/180 Hz, and training-partition-only standardization. Glove targets are
downsampled to 25 Hz and audited under both the paper-like constrained baseline
and more local lower-envelope baselines. Winner-take-all finger reassignment is
not used in the final paths because it created movement on the wrong finger.

The paper-style front end is a three-level undecimated wavelet packet tree
initialized from the 17-tap `bior6.8` analysis filters. Its dilations are 1, 2,
and 4, yielding 2, 4, and 8 bands. Band energy is computed in non-overlapping
40 ms bins. FastICA is fit on the training partition and initializes the spatial
1x1 convolution. Fixed-feature LSTM, GRU, diagonal SSM, linear-attention, Mamba,
TCN, CSP, ridge, and LARS-style baselines are all available. The retrospective
systems selected per subject and finger on one chronological validation
partition. The stricter follow-up instead selects candidate family, ensemble
membership, calibration, and epoch count from three rolling blocked folds
inside training before one final chronological validation evaluation.
An experimental nonnegative ridge stack combines diverse S1 candidates for
thumb and little finger; its regularization is selected with blocked splits
inside validation, and its weights never read the released test labels.
In the stack audit, labels such as `h10s1`, `h20s2`, and `h40s1` denote
fixed-feature LSTMs with 10, 20, or 40 hidden units and random seed 1 or 2;
`h` does not denote the input-history duration.
S1 index averages six independently optimized trainable-wavelet LSTMs, each
warm-started from the selected single-model checkpoint and selected using only
validation performance. The ensemble keeps independently trained spectral and
spatial stems rather than sharing a reconstructed feature cache.
S2 middle uses an equal-weight ensemble of six independently optimized
asymmetric-wavelet LSTMs spanning two validation-screened frontend learning
rates and three seeds. Equal weighting fits no stacking parameter and reduces
the large initialization- and minibatch-order variance seen in single runs.
Retrospective prediction amplitude and offset can be normalized on the cleaned
validation target with a positive affine transform. In the blocked-CV follow-up,
paper scoring and visualization use separate arrays: raw-coordinate output uses
an inner-fold affine transform, while cleaned flexion uses an origin-preserving
gain learned inside the blocked folds and a smooth nonnegative mapping for
linear heads. Exact intermediate model outputs remain available for auditing.

The 2018 implementation's four-second training blocks were a Theano static-graph
constraint, not a physiological assumption; modern training can operate on the
complete contiguous sequence. CUDA paths use `torch.compile(mode="reduce-overhead")`
when supported.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Native Mamba is optional and is not required by the test suite or the selected
reproduction paths.

## Reproduce the core pipeline

```bash
export PYTHONPATH=src

python scripts/audit_dataset.py
python scripts/preprocess_dataset.py --subjects 1 2 3
python scripts/prepare_paper_baseline_targets.py --subjects 1 2 3
python scripts/compare_target_baselines.py --subjects 1 2 3
python scripts/audit_wavelet_frequency_response.py

# Leakage-controlled candidate selection and refitting. The final assemble
# command is the only stage that evaluates the chronological validation block.
python scripts/cache_csp_band_signals.py --subjects 1 2 3
python scripts/run_nested_ensemble_cv.py selections --concurrency 7
python scripts/run_nested_ensemble_cv.py cv --concurrency 7 --gpus 0 1 3 4 5 6 7
python scripts/run_nested_ensemble_cv.py summarize
python scripts/run_nested_ensemble_cv.py refit --concurrency 7 --gpus 0 1 3 4 5 6 7
python scripts/run_nested_ensemble_cv.py assemble

# After producing a selected prediction directory, make the public-facing
# arrays nonnegative while retaining exact unconstrained arrays for audit.
python scripts/project_prediction_nonnegative.py --subject 1 \
  --prepared-root outputs/preprocessed_v2 \
  --prediction-root outputs/s1_validation_stack_affine_v1/sub1 \
  --target local_w2_q10 --output outputs/final_nonnegative/sub1

python -m pytest -q
```

The exact experiment commands used for the reported subject-specific paths are
listed in the [project report](docs/project-report.md#reproduction-recipes).

## Evaluation policy

- Fit preprocessing, ICA, feature selection, model parameters, and any ensemble
  weights without released test labels.
- For new primary comparisons, select candidates, calibration, ensemble
  membership, and epoch count on rolling blocked folds inside training.
- Open the chronological validation partition once after fixed-epoch refitting.
- Evaluate the final frozen candidate against the original released glove
  trajectory, not a cleaned surrogate.
- Report the deterministic nonnegative projection as the default flexion
  output, while retaining and reporting unconstrained predictions separately.
- Report every finger, `Macro-5`, and competition-style `Hist-4`.
- Inspect movement windows, rest false positives, derivative PCC, state F1,
  movement peak ratio, and peak-triggered shape; PCC alone can be misleading.

## Repository layout

```text
configs/                 versioned preprocessing and model settings
docs/                    detailed report, figures, compact result summaries
scripts/                 audits, benchmarks, training, selection, visualization
src/ecog_decoding/       reusable loading, preprocessing, features, and models
tests/                   synthetic unit and regression tests
data/                    local competition files (ignored)
outputs/                 generated arrays, checkpoints, logs, figures (ignored)
```

## Citation

If this reimplementation is useful, cite the 2018 paper above. A
[`CITATION.cff`](CITATION.cff) file is included for citation managers and makes
the distinction between the software reimplementation and the original paper.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In particular, do not commit competition
data, true labels, model checkpoints, machine-specific configuration, or outputs
that reveal local paths.
