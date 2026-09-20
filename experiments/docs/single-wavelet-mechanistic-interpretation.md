# Cross-subject single-wavelet mechanistic interpretation

## Scope

This parameter audit compares one matched model family across all three BCI
Competition IV subjects and all five fingers. It inspects six frozen
single-wavelet checkpoints per pair (90 checkpoints total) without retraining,
reselection, or released-test-driven filtering.

The matched family is intentionally narrower than the model release. It matches
8 of the 15 selected structures; the remaining 7 panels use the corresponding
historical single-wavelet reference so that spatial and spectral quantities are
comparable across every subject/finger pair. These results must not be presented
as an attribution of a different selected architecture.

## Registered electrode coordinates

Competition channels are matched one-to-one to the anatomically registered
`zt`, `bp`, and `cc` recordings by sample identity at the known 5,000-sample
offset. The assignment uses the Hungarian algorithm on absolute channel
correlations. Every retained model input has an exact registered coordinate.
The uncertain inferred channels were excluded during preprocessing: S1 channel
55, S2 channels 21 and 38, and S3 channel 50.

The primary spatial quantity is a covariance-transformed forward pattern. Each
spatial row is transformed with development-set ECoG covariance, normalized by
component variance, and weighted by the final recurrent decoder's structural
use of that row. This avoids interpreting a discriminative spatial-filter
coefficient directly as cortical source amplitude.

Localization uses weighted pairwise distance in registered 3-D millimetres. A
20,000-permutation null randomly reassigns the same importance weights over the
subject's retained coordinates. Reported p-values are one-sided for greater
spatial compactness and are not selected by outcome.

## Results

| Pair | Top five channels | Effective channels | Pair distance | 3-D p | Spectral centroid |
|---|---|---:|---:|---:|---:|
| S1 thumb | 43, 1, 17, 39, 57 | 12.4 | 17.8 mm | 0.0327 | 124.3 Hz |
| S1 index | 1, 43, 39, 7, 17 | 11.0 | 19.5 mm | 0.3342 | 125.8 Hz |
| S1 middle | 17, 1, 39, 15, 43 | 31.9 | 21.5 mm | 0.4680 | 100.7 Hz |
| S1 ring | 39, 17, 4, 27, 50 | 11.3 | 19.4 mm | 0.3067 | 128.8 Hz |
| S1 little | 17, 39, 1, 15, 5 | 22.4 | 20.5 mm | 0.2497 | 117.1 Hz |
| S2 thumb | 24, 8, 3, 18, 22 | 10.2 | 13.5 mm | 0.0016 | 97.0 Hz |
| S2 index | 8, 24, 3, 18, 22 | 10.6 | 12.1 mm | <0.0001 | 119.6 Hz |
| S2 middle | 12, 3, 8, 48, 22 | 7.7 | 10.0 mm | <0.0001 | 112.9 Hz |
| S2 ring | 15, 22, 3, 12, 8 | 10.5 | 11.8 mm | <0.0001 | 100.8 Hz |
| S2 little | 3, 22, 12, 8, 15 | 11.9 | 13.3 mm | 0.0004 | 83.0 Hz |
| S3 thumb | 9, 57, 36, 18, 54 | 17.8 | 16.4 mm | <0.0001 | 121.9 Hz |
| S3 index | 54, 55, 41, 27, 58 | 22.1 | 16.3 mm | <0.0001 | 93.8 Hz |
| S3 middle | 54, 41, 56, 27, 46 | 18.8 | 16.1 mm | <0.0001 | 103.1 Hz |
| S3 ring | 54, 41, 27, 42, 4 | 42.2 | 18.5 mm | 0.0001 | 97.8 Hz |
| S3 little | 41, 46, 54, 27, 21 | 14.8 | 14.4 mm | <0.0001 | 123.7 Hz |

S1 is comparatively distributed: thumb is nominally compact, but none of its
five results survives the 15-pair Bonferroni threshold of 0.0033. Every S2 and
S3 pair does. Compactness is not the same as a single-electrode mechanism; for
example, S3 ring has a significant regional preference while still spreading
weight over about 42 effective channels.

![Registered physical forward patterns](../../docs/figures/all-subjects-single-wavelet-physical-forward-patterns.png)

The decoder-weighted spectral centroids vary by finger and subject, but the
filters themselves barely move. Across all 90 checkpoints, the maximum path
centroid change is 0.120 Hz and the maximum relative wavelet-kernel change is
0.324%. Most learned spectral differentiation therefore comes from the sparse
and recurrent readout choosing among nearly fixed initialized paths.

![Wavelet-path importance](../../docs/figures/all-subjects-single-wavelet-spectral-path-importance.png)

The wavelet paths are squared before pooling, so carrier-phase sign is not
retained. This audit supports neither a phase-specific mechanism nor a causal
claim that stimulating a highlighted contact or frequency would move a finger.

## Reproduce

The competition data and registered reference recordings are not redistributed.
With those inputs and the ignored checkpoint artifacts restored, run from the
repository root:

```bash
export PYTHONPATH=experiments/scripts:scripts:src

python experiments/scripts/infer_competition_channel_map.py \
  --output-root outputs/channel_geometry_inference_v1

for subject in 1 2 3; do
  python experiments/scripts/analyze_single_wavelet_mechanisms.py \
    --subject "$subject" --permutations 20000
done

python experiments/scripts/render_single_wavelet_mechanisms_all_subjects.py
```

The compact, versioned result is
[`../results/single-wavelet-mechanisms-all-subjects.json`](../results/single-wavelet-mechanisms-all-subjects.json).
The full per-checkpoint spectra, signed forward patterns, and generated reports
remain under ignored `outputs/mechanistic_single_wavelet_all_subjects_v1/`.
