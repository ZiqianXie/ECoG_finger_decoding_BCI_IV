#!/usr/bin/env bash
set -euo pipefail

: "${PY:?Set PY to the project Python interpreter}"
export PYTHONPATH=src

for subject in 1 2; do
    if [[ "$subject" == 1 ]]; then
        target="local_w2_q10"
    else
        target="local_w1_q10"
    fi
    for reference in car bipolar laplacian; do
        "$PY" scripts/prepare_reference_variant.py \
            --subject "$subject" \
            --source-root outputs/preprocessed_v2 \
            --output-root "outputs/preprocessed_ref_${reference}_v1" \
            --reference "$reference"
        "$PY" scripts/benchmark_ridge_target_variants.py \
            --subject "$subject" \
            --prepared-root "outputs/preprocessed_ref_${reference}_v1" \
            --output-root "outputs/reference_ridge_${reference}_v1" \
            --targets "$target" \
            --top-features 512 \
            --device cuda
    done
done
