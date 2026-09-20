#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MODEL_SEED" >&2
  exit 2
fi

model_seed=$1
cache_root=outputs/candidate_residual_s3_little_dual_contrast_tails2x2_hiddennorm_oof_v2/sub3/little
output_root=outputs/candidate_residual_s3_little_dual_contrast_tails2x2_hiddennorm_seed_ensemble_v1/seed${model_seed}/sub3/little

for outer_fold in 0 1 2; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=scripts:src "$PY" \
    scripts/cross_validate_single_wavelet.py \
    --subject 3 \
    --finger little \
    --prepared-root outputs/preprocessed_v2 \
    --fold-root outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1 \
    --leaf-cache /dev/shm/ecog_wavelet_1000hz_taps5over2/sub3 \
    --model-rate 1000 \
    --tap-resample-up 5 \
    --tap-resample-down 2 \
    --ica-prescreen 512 \
    --component-chunk 16 \
    --csp-mode tails_2x2 \
    --csp-band-mode joint_hhl_hhh \
    --csp-contrast-mode dual_rest_other \
    --ica-cache-root outputs/residual_recurrent_single_wavelet_oof_v1/residual_lstm/lr3e-4 \
    --lasso-backend torch_fista \
    --hidden-size 32 \
    --recurrent-cell residual_lstm \
    --output-activation softplus \
    --softplus-beta 10 \
    --wavelet-interlevel-normalization \
    --no-wavelet-final-normalization \
    --sequence-steps 100 \
    --sequence-stride 25 \
    --sampler-mode dense_sequences \
    --batch-size 24 \
    --head-learning-rate 0.0001 \
    --spatial-learning-rate 0 \
    --wavelet-learning-rate 0 \
    --interlevel-learning-rate 0.001 \
    --frozen-update-grid 50 100 200 \
    --unfreeze-after 100 \
    --unfrozen-update-grid 5 10 20 50 100 \
    --weight-decay 0.0001 \
    --selection-metric raw_pcc \
    --selection-rule best \
    --no-require-lstm-update \
    --no-compile \
    --device cuda \
    --seed "$model_seed" \
    --split-seed 2026 \
    --sampler-seed 2026 \
    --initialization-cache-root "${cache_root}/fold${outer_fold}" \
    --outer-folds "$outer_fold" \
    --output "${output_root}/fold${outer_fold}"
done
