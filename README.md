# ECoG finger-trajectory decoding

In 2018, we published a CNN-LSTM model for continuous finger-trajectory
decoding from ECoG:

> Z. Xie, O. Schwartz, and A. Prasad, “Decoding of finger trajectory from
> ECoG using deep learning,” *Journal of Neural Engineering*, 15(3), 036009,
> 2018. [doi:10.1088/1741-2552/aa9dbe](https://doi.org/10.1088/1741-2552/aa9dbe)

Several people have asked me for the source code since then. Unfortunately, the
original code was lost when my laptop hard drive failed during a move. I began
this repository in 2026 because those requests deserved a better answer than
“the code is gone.”

This is not a recovered copy of the old Theano project. I rebuilt the method
from the paper, the public BCI Competition IV data, and my recollection of the
original experiments. I also used the opportunity to revisit decisions that
were constrained by the software and compute available at the time. The result
is both a late reimplementation and a continuation of the original work.

I developed this reconstruction with substantial assistance from
[OpenAI Codex](https://openai.com/codex/) using GPT-5.6 Sol for code
reconstruction, experiment orchestration, quantitative and visual diagnostics,
and documentation. I remain responsible for the scientific decisions and
interpretation.

The [project report](docs/project-report.md) contains the complete experimental
record. This README is meant to explain what the current pipeline does, why its
less obvious choices were made, and how to reproduce it.

## Overview

The task is BCI Competition IV, Data Set 4: continuous reconstruction of five
glove trajectories from ECoG in three subjects. The model is trained separately
for each subject and finger.

All fifteen current subject/finger models use one signal path:

1. The broadband ECoG is notch-filtered at 60, 120, and 180 Hz.
2. FastICA and CSP initialize rows of one trainable spatial convolution.
3. A three-level, dilated `bior6.8` wavelet-packet tree produces eight spectral
   outputs without temporal decimation.
4. Squared activation is accumulated in non-overlapping 40 ms bins.
5. LARS selects a sparse set of spatial-spectral features.
6. A nonlinear LSTM models the temporal history and predicts a nonnegative
   trajectory through a Softplus output.

The wavelet tree is literal: one signal is split into low/high children at the
first level, those two are split into four at the second, and the four are split
into eight at the third. Every node starts from the same low/high `bior6.8`
prototype pair, but each copy is an independent trainable filter. Dilations of
1, 2, and 4 replace temporal downsampling, so all eight paths retain the original
time grid. Coefficients that are zero in the initialization remain trainable.

The LARS solution is more than a pruning step. It gives the recurrent decoder a
working regression function before nonlinear optimization begins. Parameters
that should initially contribute little are randomized at approximately
`1e-3`, rather than fixed at zero, so the LSTM starts in a near-linear regime
without losing nonlinear capacity. Training first fits the recurrent head with
the spatial and spectral stem frozen, then fine-tunes the complete differentiable
model at a smaller learning rate.

Eleven pairs use a larger spatial initialization containing FastICA rows plus
CSP rows fitted separately in seven conventional frequency ranges. The other
four—S1 index and little, S2 ring, and S3 middle—retain the earlier FastICA plus
high-gamma-CSP initialization because the larger bank did not improve their
cross-fold trajectory quality. This is per-finger model selection, not a
mixture of signal branches: after initialization, every spatial row operates on
the same notched broadband ECoG and enters the same wavelet tree, LARS selector,
and LSTM.

The original four-second minibatches were largely a Theano static-graph
constraint, not a physiological assumption. The current implementation can use
longer histories, caches reconstructed windows, and keeps the small dataset in
GPU memory.

## What was changed, and why

### Glove baseline correction

The raw glove channels have a slowly varying baseline. A single baseline for an
entire recording leaves long segments with different resting levels; aggressive
event-level correction, on the other hand, can distort the movement itself.
The current target pipeline estimates a local lower envelope and subtracts it
before normalization.

The fitted envelope is subtracted completely, but it remains an estimate: a
smooth baseline cannot always distinguish sensor drift from a sustained finger
position. Slow upward changes can therefore remain in the movement target. They
can still be decoded from ECoG because the model uses the slowly varying power
envelopes of its higher-frequency bands, rather than trying to match the glove
frequency directly.

Crucially, that envelope is refitted inside every training fold. “Split-safe
target” in this repository means that neither the baseline nor the normalization
for a held-out interval was estimated from that interval. This matters because
the target preprocessing is part of the fitted model, even though it occurs on
the glove rather than the ECoG.

### Cross-finger movement

Ring and little finger trajectories often contain genuine co-movement. An early
version of this reconstruction assigned each event to whichever finger had the
largest corrected trajectory. That winner-take-all rule produced visually clean
targets, but it also moved small deflections from one finger to another and
occasionally created an apparent movement where none existed.

The released trajectories are therefore preserved for scoring, and the cleaned
targets use conservative baseline subtraction rather than hard finger
assignment. Separate per-finger models allow spatial and spectral filters to
specialize without pretending that the biomechanics are independent.

### Event-grouped validation

Every 40 ms prediction uses several seconds of preceding ECoG. Randomly
splitting adjacent bins would put strongly overlapping input histories on both
sides of the split and give an optimistic validation score. The cross-validation
code instead keeps complete movement/rest events together and removes 95 bins
(3.8 s) around each held-out boundary.

The competition training file contains 400,000 labeled samples per subject. The
current selection protocol uses all of it as development data and builds three
folds separately for every subject and finger, because their movement-event
distributions differ. Each fold holds out complete events from across the
recording rather than one contiguous time block. Target baselines, CSP filters,
and LARS selection are refitted inside each fold.

Earlier reconstruction experiments used the first two-thirds for fitting and a
final-third chronological validation segment. Those experiments remain in the
project report as historical diagnostics, but they are not the source of the
current headline configuration.

### Notch filtering and trainable filters

The ECoG is notch-filtered at 60, 120, and 180 Hz before the learned filter bank.
This explicitly removes narrow power-line components instead of asking the
network to suppress them from limited data.

For every current route, the spatial and wavelet filters are initialized from
FastICA/CSP and the biorthogonal tree, then trained during the second stage.
The wavelet tree begins as eight overlapping views ordered from low to high
frequency; gradient descent can then adjust both spatial and temporal filters.
The measured responses and implementation checks are kept in the
[project report](docs/project-report.md), where they can be read with the
corresponding validation details.

### Why seven CSP initialization bands?

The wavelet tree gives a principled multiresolution initialization. The seven
designed bands give CSP several familiar ECoG views:

| Band | What it is intended to expose |
|---|---|
| 4–8 Hz | Slow/theta-range movement modulation |
| 8–12 Hz | Mu/alpha rhythm |
| 12–30 Hz | Beta rhythm and movement-related desynchronization |
| 30–55 Hz | Low gamma below the strongest 60 Hz line component |
| 65–95 Hz | Lower high gamma above the 60 Hz component |
| 105–145 Hz | A middle high-gamma scale |
| 155–195 Hz | An upper high-gamma scale |

The bands are produced with fourth-order zero-phase Butterworth filters after
the 60, 120, and 180 Hz notch stage, but only while fitting CSP. They are not
parallel inputs to the decoder. Separating 30–55 from 65–95 Hz avoids burying
the strongest power-line region inside one broad gamma feature, while three
high-gamma ranges let CSP find spatial patterns at different scales.

For each band and training fold, I fit two CSP problems: movement of the decoded
finger versus common rest, and movement of any finger versus common rest. From
each problem I retain the two filters at both ends of the generalized
eigenspectrum. This produces eight spatial projections per band, or 56 CSP
rows. The weight vectors are concatenated with the FastICA rows in one spatial
convolution. All rows are then applied to the same notched broadband recording
and pass through one shared wavelet tree. Fold-local LARS selects useful
spatial-wavelet energy histories before initializing the LSTM. CSP,
normalization, and LARS are all refitted without the held-out event; the stem is
frozen during LSTM warm-up and then fine-tuned end to end at a smaller learning
rate.

## Current results: no test peek

The primary comparison remains Pearson correlation with the released,
unmodified test glove trajectory. `Macro-5` is the mean across all five fingers.
The paper values below are calculated from its rounded per-finger CNN-LSTM
numbers, so they should not be interpreted as more precise versions of the
paper's rounded aggregate.

| Subject | 2018 paper | Cross-fold-validated six-seed refit |
|---|---:|---:|
| S1 | 0.556 | **0.558** |
| S2 | 0.408 | **0.438** |
| S3 | 0.582 | **0.676** |

| Subject | Finger | 2018 paper | Selected final refit | Difference |
|---|---|---:|---:|---:|
| S1 | Thumb | 0.75 | 0.693 | -0.057 |
| S1 | Index | 0.79 | 0.812 | +0.022 |
| S1 | Middle | 0.17 | 0.200 | +0.030 |
| S1 | Ring | 0.60 | 0.660 | +0.060 |
| S1 | Little | 0.47 | 0.426 | -0.044 |
| S2 | Thumb | 0.62 | 0.573 | -0.047 |
| S2 | Index | 0.38 | 0.411 | +0.031 |
| S2 | Middle | 0.27 | 0.283 | +0.013 |
| S2 | Ring | 0.47 | 0.531 | +0.061 |
| S2 | Little | 0.30 | 0.389 | +0.089 |
| S3 | Thumb | 0.74 | 0.787 | +0.047 |
| S3 | Index | 0.55 | 0.569 | +0.019 |
| S3 | Middle | 0.46 | 0.650 | +0.190 |
| S3 | Ring | 0.41 | 0.661 | +0.251 |
| S3 | Little | 0.75 | 0.715 | -0.035 |

All three subject aggregates exceed the rounded paper values. Visual inspection
supports the strongest event-timing results, especially for S2 little and the
S3 models, but it also shows that S1/S2 middle still overproduce low-amplitude
activity during rest and retain cross-finger ambiguity. Closing the aggregate
paper gap is therefore not the same as solving every trajectory.

The exact unrounded scores, per-finger spatial-initialization routes, ensemble
membership, and single-branch invariant are recorded in
[`docs/results/final-single-branch-six-seed.json`](docs/results/final-single-branch-six-seed.json).

This label describes how I selected the current result. It does not mean I had
never seen the released labels. Before I fixed this protocol, I used them to
diagnose earlier models and compare saved runs. Looking back, I also think the
exploratory workflow behind my 2018 result was probably influenced by repeated
test feedback. Because the old code and logs were lost, I cannot quantify that
influence. I therefore treat the paper numbers as a historical reference, not a
prospectively sealed benchmark.

### How the final refit is produced

Before the final refit, the preprocessing, architecture, history length, and
epoch rule are fixed using separate event-grouped cross-validation for every
subject and finger. The folds cover the complete competition training file.
The network weights are then initialized anew and trained on that complete
development recording.

| Phase | Data used | Purpose |
|---|---|---|
| Model selection | All 400,000 samples, held out by fold | Three event-grouped folds choose the configuration separately for every subject and finger. |
| Final refit | All 400,000 development samples | Reinitialize six members and train the selected configuration. |
| Final test | Separate 200,000-sample released test file | Compute the reported test PCC; test labels are not used for fitting or selection. |

Here, “complete development recording” means the full 400,000-sample labeled
competition training file; it never includes the released test recording.

This final-refit result is the one to use when evaluating the reproducible
pipeline. It exceeds the rounded paper mean for all three subjects.
Cross-validation selected the larger seven-band CSP spatial initialization for
eleven pairs and retained the earlier high-gamma spatial initialization for S1
index/little, S2 ring, and S3 middle. Both choices feed the same single wavelet-
LSTM path.
The largest gaps are not explained by a global finger permutation or a simple
temporal lag. They are concentrated in particular fingers and recording
periods, consistent with target-regime and ECoG nonstationarity.

## PCC and trajectory quality

PCC is invariant to affine scaling. A prediction can therefore correlate well
with the glove while having almost no visible amplitude, or it can follow the
main events while producing unacceptable motion during rest. PCC is retained
for comparison with the paper, but model diagnosis also includes derivative
PCC, rest RMS, movement-state F1, peak amplitude, and event-aligned plots.

The current plots use the exact saved decoder output. No test-fitted rescaling
or nonnegative display correction is applied. Softplus makes the trajectories
nonnegative, while also making low-amplitude false movement during rest easy to
see. S3 little reaches PCC 0.715 and follows the major movement episodes, though
some individual peaks are compressed.

### Cross-finger ambiguity remains

The weakest morphology is now S1 and S2 middle finger. Their predictions recover
some movement timing but also respond to index or ring events and produce diffuse
low-amplitude activity during rest. The decoders do not share outputs, so this is
not caused by branch mixing. It is consistent with similar ECoG patterns,
biomechanical co-movement in the glove targets, and temporal nonstationarity.
Hard winner-take-all target correction is still rejected because it can move a
real deflection from one finger label to another.

## Main experimental conclusions

- The larger seven-band CSP spatial initialization was selected for 11 of 15
  subject/finger pairs. Four retain the simpler high-gamma-CSP initialization.
- Every selected model uses one spatial-wavelet-LSTM signal path. Designed-band
  signals initialize CSP rows; they are not parallel decoder inputs or a gated
  mixture of outputs.
- A LARS-initialized LSTM, frozen-stem warm-up, and low-learning-rate end-to-end
  fine-tuning improve stability while keeping the spatial and temporal filters
  differentiable.
- Six seeds were refitted for every pair. All 90 final members passed the
  training-only collapse screen and were retained in their respective ensemble.
- The final Macro-5 PCC is 0.558, 0.438, and 0.676 for S1--S3, above the rounded
  paper values for all three subjects. Eleven of 15 individual fingers also
  exceed the paper values.
- Hard winner-take-all target correction is rejected because it can transfer
  movement between fingers.
- PCC remains a proxy rather than the sole goal: S1 and S2 middle-finger
  morphology still shows cross-finger and rest-state errors despite the improved
  aggregate result.

These are conclusions from this reconstruction, not claims about what was or
was not tried in the original unpublished code. Full tables, unsuccessful
experiments, and visual diagnoses are retained in the
[project report](docs/project-report.md).

## Data

Download [BCI Competition IV, Data Set 4](https://www.bbci.de/competition/iv/)
and arrange the competition files and released labels as follows:

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

The competition data are not redistributed here. The loader validates the
expected shapes before an experiment begins.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproduce the core pipeline

```bash
export PYTHONPATH=scripts:src

python scripts/audit_dataset.py
python scripts/preprocess_dataset.py --subjects 1 2 3
python scripts/prepare_split_safe_targets.py --subjects 1 2 3
python scripts/audit_wavelet_frequency_response.py

# Build per-subject/per-finger folds over the complete development recording.
python scripts/build_event_stratified_folds.py --subjects 1 2 3 \
  --fingers thumb index middle ring little --purge-bins 95 \
  --selection-scope full-development \
  --target-map configs/targetsafe_conservative_targets.yaml \
  --output-root outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1

# Prepare both candidate spatial initializations. The seven-band signals are
# used only to fit CSP rows; both models consume the same broadband ECoG.
python scripts/run_prepare_perfinger_ica_csp_wavelet_all.py \
  --subjects 1 2 3 --fingers thumb index middle ring little --gpus 0 \
  --csp-source high_gamma --csp-mode tails_2x2 \
  --no-include-any-movement-csp \
  --output-root outputs/perfinger_ica_csp_wavelet_1000hz_all_tails2x2_hg_v1

python scripts/run_prepare_perfinger_ica_csp_wavelet_all.py \
  --subjects 1 2 3 --fingers thumb index middle ring little --gpus 0 \
  --csp-source seven_band --csp-mode tails_2x2 \
  --include-any-movement-csp \
  --output-root outputs/perfinger_ica_csp_wavelet_1000hz_sevenband_ownany_v1

# Cross-fold the high-gamma spatial initialization.
python scripts/run_event_lars_e2e_nested_cv.py --subjects 1 2 3 \
  --fingers thumb index middle ring little --folds 0 1 2 --seeds 0 \
  --gpus 0 --target-map configs/targetsafe_conservative_targets.yaml \
  --fold-root outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1 \
  --spatial-cache-root outputs/perfinger_ica_csp_wavelet_1000hz_all_tails2x2_hg_v1 \
  --output-root outputs/event_lars_e2e_ica_csp_1000hz_all_tails2x2_hg_v1 \
  --warmup-epochs 8 --max-epochs 48 --learning-rate 1e-4 \
  --spatial-learning-rate 3e-6 --wavelet-learning-rate 3e-6 \
  --output-activation softplus --sequence-steps 100 \
  --batch-size 24 --unfrozen-batch-size 4

# Cross-fold the larger seven-band spatial initialization.
python scripts/run_event_lars_e2e_nested_cv.py --subjects 1 2 3 \
  --fingers thumb index middle ring little --folds 0 1 2 --seeds 0 \
  --gpus 0 --target-map configs/targetsafe_conservative_targets.yaml \
  --fold-root outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1 \
  --spatial-cache-root outputs/perfinger_ica_csp_wavelet_1000hz_sevenband_ownany_v1 \
  --output-root outputs/event_lars_e2e_ica_csp_1000hz_sevenband_ownany_lowlr_v1 \
  --max-features 1024 --warmup-epochs 8 --max-epochs 48 \
  --learning-rate 3e-5 --spatial-learning-rate 1e-6 \
  --wavelet-learning-rate 1e-6 --output-activation softplus \
  --sequence-steps 100 --batch-size 24 --unfrozen-batch-size 4

# Compare the two complete single-path models using fold predictions only.
python scripts/summarize_single_branch_oof.py \
  --input-root outputs/event_lars_e2e_ica_csp_1000hz_sevenband_ownany_lowlr_v1 \
  --baseline-root outputs/event_lars_e2e_ica_csp_1000hz_all_tails2x2_hg_v1 \
  --seeds 0 --output outputs/final_single_branch_oof_comparison.json

# Refit six seeds for the four retained high-gamma models.
python scripts/run_frozen_event_refits.py --gpus 0 \
  --pairs 1:index 1:little 2:ring 3:middle \
  --ensemble-map configs/final_single_branch_oof_selected.yaml \
  --output-root outputs/full_development_ica_csp_single_branch_all_v1 \
  --selection-cache-root outputs/full_development_ica_csp_single_branch_all_lars_v1

# Refit six seeds for the eleven selected seven-band models.
python scripts/run_frozen_event_refits.py --gpus 0 \
  --pairs 1:thumb 1:middle 1:ring \
          2:thumb 2:index 2:middle 2:little \
          3:thumb 3:index 3:ring 3:little \
  --ensemble-map configs/sevenband_single_branch_promoted.yaml \
  --output-root outputs/full_development_ica_csp_sevenband_single_branch_promoted_v1 \
  --selection-cache-root outputs/full_development_ica_csp_sevenband_single_branch_lars_v1

# Average seeds only within each selected whole model and render the final result.
python scripts/summarize_frozen_full_refit.py \
  --input-root outputs/full_development_ica_csp_sevenband_single_branch_promoted_v1 \
  --ensemble-map configs/final_single_branch_oof_selected.yaml \
  --output-root outputs/final_single_branch_oof_selected_v1/ensemble

python -m pytest -q
```

The temporary CSP carrier-band cache defaults to `/dev/shm`, so each preparation
run must stay on one host. All selected models use a 100-step history at 25 Hz
(four seconds). List every available GPU after `--gpus` to parallelize the 15
independent finger models. Historical and negative-control recipes are retained
in the
[reproduction recipes](docs/project-report.md#reproduction-recipes).

## Repository layout

```text
configs/                 versioned preprocessing and model settings
docs/                    report, figures, and compact result summaries
scripts/                 audits, training, selection, and visualization
src/ecog_decoding/       reusable loading, preprocessing, features, and models
tests/                   synthetic unit and regression tests
data/                    local competition files (ignored)
outputs/                 predictions, checkpoints, and logs (ignored)
```

## Citation

If this reimplementation is useful, please cite the 2018 paper above. The
included [`CITATION.cff`](CITATION.cff) distinguishes this software repository
from the original publication and from the lost historical source code.
