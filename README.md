# ECoG finger-trajectory decoding

In 2018, we published a CNN-LSTM model for continuous finger-trajectory
decoding from electrocorticography (ECoG):

> Z. Xie, O. Schwartz, and A. Prasad, “Decoding of finger trajectory from
> ECoG using deep learning,” *Journal of Neural Engineering*, 15(3), 036009,
> 2018. [doi:10.1088/1741-2552/aa9dbe](https://doi.org/10.1088/1741-2552/aa9dbe)

Several people have asked me for the source code. Unfortunately, the original
code was lost when my laptop hard drive failed during a move. I began this
repository in 2026 because those requests deserved a better answer than “the
code is gone.”

This is a late reimplementation, not a recovered copy of the old Theano/Keras
project. I rebuilt the method from the paper, the public BCI Competition IV
data, and my recollection of the original experiments. The reconstruction was
developed with substantial help from OpenAI Codex using GPT-5.6 Sol for code,
experiment orchestration, diagnostics, and documentation. I remain responsible
for the scientific decisions and interpretation.

The released implementation deliberately uses one understandable model for all
15 subject/finger pairs. It keeps the original 1 kHz signal, initializes one
trainable wavelet tree to spend its eight outputs below 200 Hz, and uses neither
an auxiliary low-frequency branch nor a mixture of separately trained models.

## The model

One model is trained for each subject and each of the five fingers. Every model
has the same signal path:

1. Remove the documented bad channel(s), notch 60, 120, and 180 Hz line noise,
   and standardize ECoG with statistics fitted on the training split.
2. Initialize a 1x1 spatial convolution with FastICA components and one
   finger-specific CSP component. CSP (common spatial patterns) contrasts
   movement of the decoded finger with common rest.
3. Pass every spatial component through one three-level, undecimated
   `bior6.8` wavelet-packet tree. “Undecimated” means that the convolutions do
   not discard time samples.
4. Convert each of the eight terminal wavelet signals to log energy in
   non-overlapping 40 ms bins, giving a 25 Hz feature sequence aligned with the
   glove trajectory.
5. Use LARS, a sparse linear regression, to select spatial-frequency histories
   and initialize a nonlinear LSTM in a near-linear operating regime.
6. Train the LSTM with the spatial/wavelet stem frozen, then fine-tune the full
   differentiable path at a smaller learning rate. A Softplus output represents
   nonnegative flexion.

The LSTM is initialized from LARS; it is not a residual model with a frozen
linear skip. Weights that should initially be near zero are randomized at about
`1e-3`, preserving nonlinear capacity while starting from a stable linear
decoder.

### Why interpolate the wavelet filters?

The competition ECoG is stored at 1 kHz, so an ordinary three-level wavelet
packet divides the 0–500 Hz Nyquist range. The useful acquisition range is
concentrated below approximately 200 Hz. Allocating five of eight initial
wavelet leaves above that range wastes the initialization.

The released model keeps the 1 kHz samples and changes the filters rather than
resampling the input. The first `bior6.8` low/high pair is stretched by 5/2 with
polyphase Kaiser interpolation. The next two levels use the original taps at
dilations 5 and 10. This preserves the 1 kHz time grid while giving eight
measured response centroids at approximately 13.6, 40.3, 63.7, 91.9, 110.2,
137.4, 159.3, and 186.7 Hz. The path labels are Gray-coded after high-pass
splits, so `H` and `L` strings should not be read as a simple left-to-right
frequency order.

![Measured frequency-compressed wavelet initialization](docs/figures/frequency-compressed-wavelet-initialization.png)

The [machine-readable frequency audit](docs/results/frequency-compressed-wavelet-initialization.json)
comes from the implemented PyTorch cascade, including its nonlinearities—not
from idealized textbook band edges.

### Spatial initialization

FastICA supplies label-free spatial components. The additional CSP component is
fitted separately inside every subject/finger training split. It uses the two
initialized wavelet paths covering roughly 100–150 Hz, aggregated before the
CSP covariance calculation. The resulting single CSP weight vector is then
applied to the same notched broadband ECoG as the ICA rows. It is not a second
temporal branch, and it is not one of seven conventional bandpass inputs.

LARS sees the eight wavelet energies from every retained spatial row and keeps
only the useful histories. Thus the complete decoder remains one spatial
convolution, one wavelet tree, and one LSTM.

## Target and validation

The glove trajectories have slowly changing rest levels. A local lower-envelope
baseline is therefore fitted and subtracted separately inside each split. The
window is subject-specific because the glove drift differs across recordings.
Hard winner-take-all assignment is not used: ring/little and other finger pairs
can genuinely co-move, and forced reassignment can fabricate movement on the
wrong finger.

Random bin splits would leak nearly identical ECoG histories across train and
validation. Model selection instead uses three folds made from complete
movement/rest events. A 95-bin (3.8 s) margin is removed around every held-out
interval. Target baselines, normalization, FastICA, CSP, and LARS are refitted
inside every fold.

The complete 400,000-sample competition training file is the development set.
After cross-fold selection, six seeds are refitted on all development samples.
The separate 200,000-sample released test file is used only for the final score.

## Results: no test peek during selection

The reported metric is Pearson correlation coefficient (PCC) against the
unmodified released test glove trajectory, matching the competition convention.
`Macro-5` is the unweighted mean over five independently decoded fingers.

| Subject | 2018 paper | 2026 single-tree refit |
|---|---:|---:|
| S1 | 0.556 | **0.525** |
| S2 | 0.408 | **0.414** |
| S3 | 0.582 | **0.593** |

| Subject | Finger | 2018 paper | 2026 refit | Difference |
|---|---|---:|---:|---:|
| S1 | Thumb | 0.75 | 0.691 | -0.059 |
| S1 | Index | 0.79 | 0.743 | -0.047 |
| S1 | Middle | 0.17 | 0.222 | +0.052 |
| S1 | Ring | 0.60 | 0.616 | +0.016 |
| S1 | Little | 0.47 | 0.353 | -0.117 |
| S2 | Thumb | 0.62 | 0.586 | -0.034 |
| S2 | Index | 0.38 | 0.377 | -0.003 |
| S2 | Middle | 0.27 | 0.210 | -0.060 |
| S2 | Ring | 0.47 | 0.505 | +0.035 |
| S2 | Little | 0.30 | 0.393 | +0.093 |
| S3 | Thumb | 0.74 | 0.663 | -0.077 |
| S3 | Index | 0.55 | 0.520 | -0.030 |
| S3 | Middle | 0.46 | 0.619 | +0.159 |
| S3 | Ring | 0.41 | 0.557 | +0.147 |
| S3 | Little | 0.75 | 0.608 | -0.142 |

S2 and S3 exceed the paper’s rounded aggregate; S1 does not. Six of the 15
individual finger scores exceed the rounded paper values. The exact scores and
all six member audits are in
[`docs/results/final-single-branch-six-seed.json`](docs/results/final-single-branch-six-seed.json).

All 90 members passed a collapse screen based only on development predictions.
Seed variation is small: the largest standard deviation of member test PCC is
0.0039. The members are also highly correlated with one another, so the
six-seed mean adds stability but little ensemble gain.

The figure below is the visual verdict, not a decorative score plot. Black is a
post-hoc baseline-corrected glove trace used only for diagnosis. Blue is the
exact mean of the six nonnegative Softplus outputs; it is not rescaled with test
labels. The PCC printed above each panel is still computed against the raw
competition target.

![Released-test trajectories for all 15 models](docs/figures/final-single-wavelet-test-trajectories.png)

The model captures clear event timing for S1 thumb/index/ring and for most S3
fingers. It also shows the remaining failure plainly: S1/S2 middle have weak
movement selectivity and excessive low-amplitude activity during rest, while
S2 little misses several large events. S3 little tracks several episodes but
does not reproduce the paper’s unusually high score. PCC alone should therefore
not be read as a complete measure of trajectory quality.

I had inspected released labels during the broader reconstruction before fixing
this protocol, and the original 2018 exploration may also have been influenced
by repeated test feedback. The result above is called “no test peek” because
the architecture, preprocessing, schedules, and seeds for this refit were fixed
from development folds before these test scores were read—not because the
labels were historically unknown to me.

The [experimental archive](experiments/README.md) contains alternative models,
including models that score better for some individual fingers. They are not
substituted into this table because they were developed under different or
test-informed diagnostic protocols.

## Reproduce the release

Download [BCI Competition IV, Data Set 4](https://www.bbci.de/competition/iv/)
and arrange it as follows:

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

The competition data are not redistributed. Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
export PYTHONPATH=scripts:src

python scripts/audit_dataset.py
python scripts/preprocess_dataset.py --subjects 1 2 3
python scripts/build_event_stratified_folds.py \
  --subjects 1 2 3 --fingers thumb index middle ring little \
  --purge-bins 95 --selection-scope full-development \
  --target-map configs/targetsafe_conservative_targets.yaml \
  --output-root outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1

python scripts/audit_wavelet_frequency_response.py \
  --sampling-rate 1000 --tap-resample-up 5 --tap-resample-down 2 \
  --fft-size 32768 \
  --output outputs/single_wavelet_1000hz_tap5over2_v1/frequency_response.json

python scripts/run_single_wavelet_cv.py \
  --subjects 1 2 3 --fingers thumb index middle ring little \
  --gpus 0 1 2 3 4 5 6 7 \
  --fold-root outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1 \
  --output-root outputs/single_wavelet_1000hz_tap5over2_nested_v2 \
  --model-rate 1000 --tap-resample-up 5 --tap-resample-down 2 \
  --head-learning-rate 1e-4 --spatial-learning-rate 3e-6 \
  --wavelet-learning-rate 3e-6 --lars-candidate-scale 1 \
  --sequence-steps 100 --sequence-stride 25 --batch-size 24 \
  --csp-mode movement_1 --output-activation softplus --softplus-beta 10 \
  --sampler-mode dense_sequences --selection-metric raw_pcc \
  --selection-rule best --require-lstm-update --compile

python scripts/run_single_wavelet_refits.py \
  --selection-root outputs/single_wavelet_1000hz_tap5over2_nested_v2 \
  --output-root outputs/single_wavelet_1000hz_tap5over2_full_refit_v1 \
  --model-rate 1000 --tap-resample-up 5 --tap-resample-down 2 \
  --head-learning-rate 1e-4 --spatial-learning-rate 3e-6 \
  --wavelet-learning-rate 3e-6 --lars-candidate-scale 1 \
  --sequence-steps 100 --sequence-stride 25 --batch-size 24 \
  --csp-mode movement_1 --output-activation softplus --softplus-beta 10 \
  --sampler-mode dense_sequences --selection-metric raw_pcc \
  --selection-rule best --gpus 0 1 2 3 4 5 6 7 --compile

python scripts/run_single_wavelet_ensemble.py \
  --selection-root outputs/single_wavelet_1000hz_tap5over2_nested_v2 \
  --initialization-root outputs/single_wavelet_1000hz_tap5over2_full_refit_v1 \
  --output-root outputs/single_wavelet_1000hz_tap5over2_six_seed_refit_v1 \
  --seeds 0 1 2 3 4 5 --gpus 0 1 2 3 4 5 6 7 \
  --model-rate 1000 --tap-resample-up 5 --tap-resample-down 2 \
  --head-learning-rate 1e-4 --spatial-learning-rate 3e-6 \
  --wavelet-learning-rate 3e-6 --lars-candidate-scale 1 \
  --sequence-steps 100 --sequence-stride 25 --batch-size 24 \
  --csp-mode movement_1 --output-activation softplus --softplus-beta 10 \
  --sampler-mode dense_sequences --selection-metric raw_pcc \
  --selection-rule best --compile

python scripts/summarize_single_wavelet_ensemble.py \
  --root outputs/single_wavelet_1000hz_tap5over2_six_seed_refit_v1

python -m pytest -q
```

The code caches reconstructed wavelet leaves in `/dev/shm` and uses
`torch.compile(mode="reduce-overhead")` for repeated training calls. On the H100
server, a complete full-development refit took roughly 10–45 seconds after the
shared cache was prepared.

## Repository layout

```text
configs/                 target and split settings
docs/                    detailed report, figures, and machine-readable results
scripts/                 release preprocessing, validation, training, and rendering
src/ecog_decoding/       reusable model and preprocessing components
tests/                   release-path tests
experiments/             alternatives, negative controls, and historical diagnostics
data/                    local competition files (ignored)
outputs/                 checkpoints and generated outputs (ignored)
```

See the [project report](docs/project-report.md) for the full protocol, matched
400/1000 Hz comparison, visual interpretation, and a concise record of rejected
alternatives. If this reimplementation is useful, please cite the 2018 paper;
[`CITATION.cff`](CITATION.cff) distinguishes the software release from the
original publication and lost historical source.
