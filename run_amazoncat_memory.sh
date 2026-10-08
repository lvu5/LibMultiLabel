#!/bin/bash
set -euo pipefail

LML_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LML_REPO_DIR"
LML_PYTHON="${LML_PYTHON:-/home/lvu5/miniconda3/envs/libmultilabel/bin/python}"
LML_THREADS="${LML_THREADS:-8}"

# Keep package imports local and limit native-library and allocator overhead.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS="$LML_THREADS"
export OPENBLAS_NUM_THREADS="$LML_THREADS"
export MKL_NUM_THREADS="$LML_THREADS"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export LML_PROFILE_MEMORY=1
# Use disk-backed storage for staging weights. Override TMPDIR if necessary.
export TMPDIR="${TMPDIR:-$LML_REPO_DIR/runs/tmp}"
mkdir -p "$TMPDIR" "$LML_REPO_DIR/runs"
LML_LOG="${LML_LOG:-$LML_REPO_DIR/runs/amazoncat-memory-$(date -u +%Y%m%dT%H%M%S).log}"

"$LML_PYTHON" -c 'import libmultilabel; from liblinear.liblinearutil import train; print("Package:", libmultilabel.__file__)'
printf 'Label workers: %s; log: %s\n' "$LML_THREADS" "$LML_LOG"
/usr/bin/time -v "$LML_PYTHON" -u main.py \
    --data_name AmazonCat-13K \
    --model_name tree \
    --training_file /home/lvu5/LibMultiLabel/data/AmazonCat-13K/zeroshot/AmazonCat-13K_tfidf_train.svm \
    --test_file /home/lvu5/LibMultiLabel/data/AmazonCat-13K/zeroshot/AmazonCat-13K_tfidf_test.svm \
    --linear \
    --linear_technique tree \
    --data_format svm \
    --seed 42 \
    --liblinear_options "-m $LML_THREADS" \
    "$@" 2>&1 | tee "$LML_LOG"
