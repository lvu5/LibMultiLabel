#!/bin/bash
set -euo pipefail

LML_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LML_REPO_DIR"
LML_PYTHON="${LML_PYTHON:-/home/lvu5/miniconda3/envs/libmultilabel/bin/python}"
LML_THREADS="${LML_THREADS:-8}"
LML_DATA_NAME=EURLEX-57K
LML_TRAIN="${LML_TRAIN:-/home/lvu5/data/EURLEX-57K/zeroshot/eurlex57k_tfidf_train_ext.svm}"
LML_TEST="${LML_TEST:-/home/lvu5/data/EURLEX-57K/zeroshot/eurlex57k_tfidf_test_ext.svm}"
LML_TAG="$(basename "$LML_REPO_DIR")"

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
LML_LOG="${LML_LOG:-$LML_REPO_DIR/runs/${LML_DATA_NAME}-${LML_TAG}-$(date -u +%Y%m%dT%H%M%S).log}"

for f in "$LML_TRAIN" "$LML_TEST"; do
    [[ -f "$f" ]] || { echo "Missing data file: $f" >&2; exit 1; }
done

{
"$LML_PYTHON" -c 'import libmultilabel; from liblinear.liblinearutil import train; print("Package:", libmultilabel.__file__)'
printf 'Repo: %s; dataset: %s; threads: %s; log: %s\n' "$LML_TAG" "$LML_DATA_NAME" "$LML_THREADS" "$LML_LOG"
/usr/bin/time -v "$LML_PYTHON" -u main.py \
    --data_name "$LML_DATA_NAME" \
    --model_name tree \
    --training_file "$LML_TRAIN" \
    --test_file "$LML_TEST" \
    --linear \
    --linear_technique tree \
    --data_format svm \
    --seed 42 \
    --liblinear_options "-m $LML_THREADS" \
    "$@"
} 2>&1 | tee "$LML_LOG"
