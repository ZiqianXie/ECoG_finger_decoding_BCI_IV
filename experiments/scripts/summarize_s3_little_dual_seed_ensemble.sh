#!/usr/bin/env bash
set -euo pipefail

root=outputs/candidate_residual_s3_little_dual_contrast_tails2x2_hiddennorm_seed_ensemble_v1
reference=outputs/candidate_residual_s3_little_dual_contrast_tails2x2_hiddennorm_oof_v2/merged/sub3/little

for model_seed in 2027 2028 2029 2030 2031; do
  PYTHONPATH=scripts:src "$PY" \
    experiments/scripts/merge_single_wavelet_outer_parts.py \
    --subject 3 \
    --fingers little \
    --parts-root "${root}/seed${model_seed}/sub3" \
    --part-prefix fold \
    --output-root "${root}/seed${model_seed}"
done

mkdir -p "${root}/seed2026/sub3/little"
for name in initialized_oof.npy selected_oof.npy target_oof.npy summary.json; do
  cp "${reference}/${name}" "${root}/seed2026/sub3/little/${name}"
done

PYTHONPATH=scripts:src "$PY" \
  experiments/scripts/summarize_single_wavelet_oof_ensemble.py \
  --root "$root" \
  --subject 3 \
  --finger little \
  --seeds 2026 2027 2028 2029 2030 2031 \
  --output "${root}/ensemble"
