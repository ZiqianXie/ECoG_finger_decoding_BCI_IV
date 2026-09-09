# Learnable-filter and hurdle-decoder branch

Branch: `experiments/learnable-filter-hurdle`

## Hypothesis

The paper-faithful wavelet-initialized filter bank is expressive, but the small,
autocorrelated dataset makes its spectral solution hard to identify. A shared
frequency-tiling initialization with finger-specific sparse gates, a separate
signed low-frequency branch, and an explicit movement-state/conditional-
amplitude decoder may improve statistical efficiency without forcing fingers
to compete for the same filters.

## Deliberate deviations from the paper

- common-average, adjacent-bipolar, and local-Laplacian reference variants;
- finger-specific Bernoulli movement state plus conditional positive amplitude;
- purged blocked inner folds and fold ensembling;
- validation-selected effective neural-to-glove lag;
- planned overcomplete learnable spectral filters, sparse finger gates, and a
  signed low-frequency/LMP branch;
- planned shallow Elastic Net/LARS and LightGBM representation probes before a
  recurrent head.

The learnable Morlet/filter-bank and hurdle-decoder deviations remain
experimental. The parameter-free nonnegative output projection was promoted to
`main` in commit `bf9e876`. A later, distinct fixed heterogeneous dictionary was
promoted in commit `73c8d38`: full-development event folds selected joint
ICA-wavelet+CSP/band features for S1 middle and S3 thumb, middle, and ring. That
route is part of the current clean-conscience result; it should not be confused
with the still-experimental learnable-filter branch documented here.

## Data boundaries

The experiments below originally retained a chronological outer split: the
first two-thirds were the fit partition and the last third was validation.
Architecture profiles were selected from purged chronological folds inside the
fit partition, and the outer validation segment could select a lag or decide
whether an experimental candidate beat the paper-faithful model. Released test
labels were read only after a candidate was frozen. This is the historical
protocol for this branch, not the current main-branch evaluation.

The current main-branch family choices instead use three per-subject/per-finger
event folds spanning all 400,000 development samples, with a 95-bin purge and
fold-local fitting of target baselines, CSP, and LARS. Consequently, the earlier
S2-thumb end-to-end run
that improved validation PCC from 0.600 to 0.630 but reduced test PCC from
0.599 to 0.579 cannot be rejected because of the test result. Its status is
unresolved pending purged blocked cross-validation across seeds and frontend
learning rates. Test performance is never a model-selection criterion in this
historical branch.

## Paper-faithful comparison

The frozen raw-test baselines are Macro-5 0.574 for S1 and 0.429 for S2. Their
outer-validation Macro-5 values are 0.583 and 0.469, respectively. Experimental
models must be compared on the same raw trajectories and history offset, with
per-finger PCC, amplitude calibration, rest false activity, onset lag, event
recall, cross-finger confusion, and movement/rest plots reported together.

## Initial experiment order

1. Screen CAR, bipolar, and local-Laplacian variants with the same fixed
   ICA/wavelet/ridge diagnostic.
2. Evaluate effective per-finger lag without changing the decoder.
3. Train the blocked-fold hurdle GRU on the weak S1 little and S2 middle paths;
   expand to all fingers only if validation supports it.
4. Restore synthetic filter-recovery tests and audit complete dilated-path
   magnitude responses.
5. Add the overcomplete learnable filter bank and signed LMP branch, then probe
   learned representations with shallow heads before recurrent decoding.

## Completed validation gates

### Output-domain projection

Baseline-corrected glove flexion has a physical lower bound of zero, while the
unconstrained regressors produced small negative excursions. A deterministic
`max(prediction, 0)` projection was therefore evaluated without fitting any
test-dependent parameter. It improved raw-test Macro-5 from 0.5742 to 0.5782
for S1 and from 0.6390 to 0.6460 for S3; S2 was already nonnegative and was
unchanged at 0.4293. Visual diagnostics also improved derivative PCC and
slightly reduced rest RMS for S1 and S3. This parameter-free physical
constraint was subsequently promoted to `main` in commit `bf9e876`; the
filter-bank and hurdle deviations remain isolated on this experimental branch.

### Explicit latency

Validation-selected per-finger shifts were at most one 25 Hz bin (40 ms). S1
raw-test Macro-5 changed only from 0.5742 to 0.5748, while S2 fell from 0.4293
to 0.4267. The effect did not transfer reliably, so no latency correction is
promoted.

### Spatial re-reference

All three alternatives lost substantial performance relative to the frozen
models. S1 raw-test Macro-5 was 0.4474 with CAR, 0.4736 with adjacent bipolar,
and 0.4298 with local Laplacian. S2 was 0.3152, 0.3006, and 0.3118,
respectively. These are negative controls; the original channel representation
is retained.

### Blocked-fold hurdle GRU

The validation gate was run first on S1 little and S2 middle. S1 little reached
validation/test PCC 0.385/0.389 versus the selected model's 0.459/0.420. S2
middle reached 0.357/0.075 versus 0.455/0.208. Movement plots showed elevated
rest activity rather than cleaner state transitions, so the candidate was not
expanded to all 15 subject-finger pairs. This also rejects the current
movement-state/conditional-amplitude implementation, not the broader idea of
finger-specific decoders.

### Synthetic filter recovery

The recovery harness contains single-band, nearby-band, broadband, spatially
mixed, correlated-finger, and drifting 60/120 Hz line-noise cases. It reports
complete dilated path responses by finite-difference linearization around rest;
zero-input subtraction prevents learned biases from masquerading as a DC
transfer response. Prediction recovery and spectral-path identification are
reported separately because an energy nonlinearity and signed spatial mixing
make a single head-weighted peak frequency ambiguous.

The corrected matched recovery results favor one additional wavelet level in
all six scenarios:

| Synthetic case | Depth 3 PCC | Depth 4 PCC | Change |
|---|---:|---:|---:|
| single band | 0.9802 | 0.9845 | +0.0043 |
| nearby bands | 0.9527 | 0.9687 | +0.0160 |
| broadband | 0.8179 | 0.8699 | +0.0520 |
| spatial mixing | 0.9457 | 0.9751 | +0.0293 |
| correlated fingers | 0.9723 | 0.9835 | +0.0112 |
| 60/120 Hz line noise plus drift | 0.9208 | 0.9773 | +0.0565 |

Depth 4 doubles the terminal paths from 8 to 16 and expands the complete-path
support from 113 to 241 input samples. These synthetic gains justify a matched
real-data validation screen but do not by themselves justify replacing the
published three-level tree.

The matched real-data fixed-feature screen rejected the full depth-4 tree.
S1 validation Macro-5 decreased from 0.4816 at depth 3 to 0.4694 at depth 4;
S2 decreased from 0.3255 to 0.3105. Released-test Macro-5 also decreased from
0.4736 to 0.4676 for S1 and from 0.3113 to 0.3090 for S2. Full depth-4
fine-tuning was therefore not launched. A future asymmetric tree should split
only validation-supported paths and retain a separate signed low-frequency
branch, rather than doubling every terminal band.

The asymmetric shallow probe replaces only the initialized `LLH` and `LHH`
depth-3 parents (the paths with more than 96% of their initial response power
inside 60--200 Hz) with four depth-4 children, retains the other six depth-3
paths, and appends a signed 201-tap 0--5 Hz LMP branch. LARS provides the sparse
gate at this diagnostic stage. S1 was essentially unchanged on validation
(0.4816 to 0.4811 Macro-5). S2 improved from 0.3255 to 0.3549, with validation
improvement on all five fingers; released-test Macro-5 similarly increased
from 0.3113 to 0.3382. This passes the representation gate for one S2-middle
trainable LSTM run, but it remains below the selected S2 ensemble.

### S2-middle frontend learning rate and seed ensemble

The paper describes two broad stages: ICA/wavelet/LARS fitting initializes a
network whose initial output matches LARS, and the initialized network is then
trained with Adam to reduce MSE. It does not document an explicit period in
which the convolutional frontend is frozen. The reported global learning rate
is `1e-5`, with `0.005` decay per epoch for 100 epochs. Because the old
implementation details are unavailable, head warmup and per-parameter-group
learning rates are treated here as new validation-controlled experiments, not
as paper-faithful facts.

On S2 middle, the asymmetric initialized network starts at validation PCC
0.3864. Freezing the frontend reached 0.4884, while direct joint fine-tuning
was strongly learning-rate and seed dependent. With model and minibatch seeds
both zero, validation PCC was 0.4665 at frontend LR `1e-6`, 0.4730 at `3e-6`,
0.4525 at `1e-5`, 0.4199 at `3e-5`, 0.4813 at `1e-4`, and 0.5088 at `3e-4`.
A 25-epoch frozen warmup followed by frontend LR `1e-4` reached 0.4985. Updating
only the spatial or only the wavelet filters reached 0.4784 and 0.4508,
respectively. Applying the paper's small learning rate and decay uniformly to
all parameter groups reached only 0.4101. Thus fine-tuning can help, as in the
author's recollection, but the current optimizer is not stable enough for a
single run to be trusted.

Three independent seeds at frontend LR `1e-4` varied from 0.3945 to 0.4813;
three at `3e-4` varied from 0.4049 to 0.5088. Equal-weight averaging is much
more stable and requires no validation-fitted stacking weights. The three
`1e-4` seeds reached validation/test PCC 0.4954/0.3162, the three `3e-4` seeds
reached 0.4943/0.3955, and the validation-selected six-model ensemble reached
0.5216/0.3907. Substituting that ensemble only for S2 middle raises the full
subject validation Macro-5 from 0.4686 to 0.4820 and descriptive test Macro-5
from 0.4293 to 0.4659. The released test score is descriptive only and did not
select the ensemble.

The variation is not attributable to only one random mechanism. At frontend LR
`3e-4`, holding model initialization at seed 0 and changing minibatch order
across seeds 0/1/2 produced validation PCC 0.5088/0.4464/0.4253 (sample SD
0.0434). Holding minibatch order at seed 0 and changing initialization across
seeds 0/1/2 produced 0.5088/0.3880/0.4259 (sample SD 0.0618). Both the head
initialization and the order of highly autocorrelated minibatches materially
affect the selected checkpoint.

The six-model ensemble also improves the validation morphology score from
0.2753 to 0.3591, derivative PCC from 0.1481 to 0.2497, state F1 from 0.4339 to
0.4973, and movement peak ratio from 0.2801 to 0.5079. Rest RMS increases
slightly from 0.0430 to 0.0477. Visual review confirms substantially recovered
movement peaks, but also shows a remaining false-positive rest excursion, so
this is an experimental S2-middle result rather than a general promotion to all
15 subject-finger models.

### All-subject differentiable-stem blocked-CV audit

The planned S3 extension was implemented for all subjects and fingers rather
than only S3. Seven continuous zero-phase Butterworth carrier bands are cached
once. A PyTorch frontend applies trainable band-specific CSP spatial weights and
trainable depthwise FIR corrections initialized as impulses before 40 ms log
energy binning. Thus the initial feature tensor reproduces the fixed CSP/band
representation while the spatial and spectral corrections remain
differentiable.

Four independent-stem candidates were evaluated on three rolling blocked folds
inside the official training partition: wavelet LARS-regime initialization,
wavelet Softplus, CSP LARS-regime initialization, and CSP Softplus. All five
fingers were trained separately for all three subjects. Candidate membership,
ensemble membership, calibration, and median refit epoch were fixed from these
folds; inner jobs did not load released-test labels. Refits then used the full
official training partition with no validation-selected checkpoint.

The one-time chronological validation `Macro-5` was 0.485/0.355/0.551 for
S1/S2/S3. Descriptive released-test `Macro-5` was 0.440/0.356/0.615. These
values do not support promotion for S1 or S2, despite selected inner-fold OOF
PCCs of 0.586--0.849. S3 transfers better and reaches descriptive `Hist-4`
0.650. The main finding is therefore a large temporal generalization gap, not a
new aggregate record.

The first assembled plots accidentally compared raw-coordinate affine outputs
against cleaned nonnegative targets. The implementation now saves both domains
explicitly. Cleaned trajectories use training-fold-only, through-origin gains;
linear outputs use a smooth Softplus boundary. This reduces test rest RMS to
0.068/0.055/0.113 for S1/S2/S3 and reveals the actual morphology: S1 index is
strong, S1 little is mostly missed, S2 contains coupled false bursts, and S3
captures the clearest five-finger trends.

## Methodological references

- [Yao et al. (2022), Riemannian features and gradient-boosted trees](https://doi.org/10.1088/1741-2552/ac4ed1)
- [FingerFlex paper](https://arxiv.org/abs/2211.01960)
- [FingerFlex implementation](https://github.com/Irautak/FingerFlex)

These are diagnostic references, not drop-in pipelines. In particular, this
project retains zero-phase 60/120/180 Hz notches rather than copying a different
line-frequency assumption.
