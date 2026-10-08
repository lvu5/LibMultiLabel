"""Compare root-clustering memory using one cached label representation.

Run in separate processes for --mode upstream and --mode bounded. --iterations
limits this to a clustering probe, not full training. Inputs remain float64.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--representation', required=True, help='Uncompressed scipy CSR .npz')
parser.add_argument('--mode', choices=('upstream', 'bounded'), required=True)
parser.add_argument('--iterations', type=int, default=6)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--output', required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import psutil
from scipy import sparse
from sparsekmeans import ElkanKmeans as Original
from libmultilabel.linear.cluster import ElkanKmeans as Bounded

x = sparse.load_npz(args.representation)
algorithm = Original if args.mode == 'upstream' else Bounded
model = algorithm(n_clusters=100, n_threads=args.threads, max_iter=args.iterations, random_state=1608637542, verbose=True)
start = time.perf_counter()
labels = model.fit(x)
seconds = time.perf_counter() - start
np.save(args.output, labels)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak /= 2**20 if sys.platform == 'darwin' else 1024
print(json.dumps({'mode':args.mode, 'shape':x.shape, 'nnz':x.nnz, 'iterations':args.iterations,
                  'threads':args.threads, 'peak_rss_mib':peak,
                  'final_rss_mib':psutil.Process().memory_info().rss/2**20, 'seconds':seconds}))
