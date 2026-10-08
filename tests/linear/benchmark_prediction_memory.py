"""Measure prediction RSS in fresh processes using the same saved model and rows.

Invoke once for each checkout with --repo and the same input file. Cache setup
and the first batch are measured separately from repeated prediction, so a
one-time conversion cannot hide a regression over many batches. The SHA256
covers every timed prediction; --output saves the last batch for inspection.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--test-file', required=True, help='Small LIBSVM file containing the same test rows for both runs.')
parser.add_argument('--output', required=True)
parser.add_argument('--repeats', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--beam-width', type=int, default=10)
args = parser.parse_args()
if min(args.repeats, args.batch_size, args.beam_width) < 1:
    parser.error('repeats, batch-size and beam-width must be positive')
sys.path.insert(0, str(Path(args.repo).resolve()))

import numpy as np
import psutil
from libmultilabel import linear

process = psutil.Process()
start = time.perf_counter()
_, model = linear.load_pipeline(args.checkpoint)
loaded_rss = process.memory_info().rss / 2**20
x = linear.load_dataset('svm', test_path=args.test_file)['test']['x']
if x.shape[0] == 0:
    parser.error('test-file must contain at least one instance')
load_seconds = time.perf_counter() - start

start = time.perf_counter()
if args.beam_width < len(model.root.children):
    model._separate_model_for_pruning_tree()
    model._model_separated = True
cache_setup_seconds = time.perf_counter() - start
cache_rss = process.memory_info().rss / 2**20

start = time.perf_counter()
model.predict_values(x[:args.batch_size], beam_width=args.beam_width)
first_batch_seconds = time.perf_counter() - start

digest = hashlib.sha256()
batch_times = []
start = time.perf_counter()
for _ in range(args.repeats):
    for offset in range(0, x.shape[0], args.batch_size):
        batch = x[offset:offset + args.batch_size]
        batch_start = time.perf_counter()
        predictions = model.predict_values(batch, beam_width=args.beam_width)
        batch_times.append(time.perf_counter() - batch_start)
        digest.update(memoryview(np.ascontiguousarray(predictions)))
benchmark_seconds = time.perf_counter() - start
prediction_seconds = sum(batch_times)
np.save(args.output, predictions)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak /= 2**20 if sys.platform == 'darwin' else 1024
print(json.dumps({
    'repo': args.repo, 'rows': x.shape[0], 'repeats': args.repeats,
    'checkpoint': str(Path(args.checkpoint).resolve()),
    'batch_size': args.batch_size, 'timed_batches': len(batch_times),
    'loaded_rss_mib': loaded_rss, 'final_rss_mib': process.memory_info().rss / 2**20,
    'peak_rss_mib': peak, 'load_seconds': load_seconds,
    'cache_setup_seconds': cache_setup_seconds, 'cache_rss_mib': cache_rss,
    'first_batch_seconds': first_batch_seconds,
    'prediction_seconds': prediction_seconds,
    'seconds_per_batch': prediction_seconds / len(batch_times),
    'median_batch_seconds': float(np.median(batch_times)),
    'benchmark_seconds_including_hash': benchmark_seconds,
    'predictions_sha256': digest.hexdigest(),
}, indent=2))
