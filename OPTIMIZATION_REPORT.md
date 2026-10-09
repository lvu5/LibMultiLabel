# Tree-model optimization report: Original → v1 → v2

LibMultiLabel linear tree model (`--linear --linear_technique tree`), evaluated on
EURLEX-4K, EURLEX-57K and AmazonCat-13K (zero-shot splits).

| Version | Directory |
| --- | --- |
| Original | `libmultilabel-original` |
| Optimized v1 | `libmultilabel-mem-benchmark` |
| Optimized v2 | `libmultilabel-mem-working` |

## 1. Results

Times are the CLI's `'main' finished in` values. Peak memory is the process
lifetime peak RSS. All runs use `-m 8`.

| Dataset | Original | v1 | v2 | v2 vs Original |
| --- | ---: | ---: | ---: | ---: |
| EURLEX-4K time | 73.64 s | 55.63 s | **49.83 s** | −32.3% |
| EURLEX-4K peak | 1964.4 MiB | 1848.3 MiB | **1245.5 MiB** | −36.6% |
| EURLEX-57K time | 437.65 s | 327.41 s | **238.39 s** | −45.5% |
| EURLEX-57K peak | 14644.3 MiB | 13181.4 MiB | **7641.6 MiB** | −47.8% |
| AmazonCat-13K time | 3440.90 s | 2727.66 s | **1252.41 s** | −63.6% |
| AmazonCat-13K peak | 20851.4 MiB | 18284.7 MiB | **8146.3 MiB** | −60.9% |

All 12 metrics (P/R/NDCG/PSP@1,3,5) of every version agree within ±0.10
percentage points for v1 and ±0.02 for v2. On AmazonCat they match to two
decimals, except PSP@5 (70.61 vs 70.60).

## 2. Run settings: identical for all versions

- **Scripts:** `run_eurlex4k_zs.sh`, `run_eurlex57k_zs.sh` and
  `run_amazoncat13k_zs.sh` are byte-identical in the three directories.
- **Command line:** every log records the same arguments:
  `main.py --model_name tree --linear --linear_technique tree --data_format svm --seed 42 --liblinear_options "-m 8"`,
  with the same Python environment (`miniconda3/envs/libmultilabel`).
- **Environment:** the scripts set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
  and `MKL_NUM_THREADS` to 8, `MALLOC_ARENA_MAX=2`, `PYTHONNOUSERSITE=1`, and
  `TMPDIR=runs/tmp`.
- **Exception:** the v1 AmazonCat figure (2727.66 s) was recorded with
  `run_amazoncat_memory.sh`. It passes the same arguments and environment and
  reads `/home/lvu5/LibMultiLabel/data/...`, whose files are byte-identical
  (`cmp`) to `/home/lvu5/data/...`. Re-running v1 with `run_amazoncat13k_zs.sh`
  gave 2673 s.
- **Machine:** one 96-thread AMD EPYC 7401 machine, with runs executed one at a
  time.

## 3. Are results the same?

**The original code is not reproducible with `-m 8`.** LIBLINEAR's dual solver
shuffles with an internal random state shared by the 8 training threads. Two
runs of the unmodified original on EURLEX-57K gave P@1 91.52 and 91.55. The
small metric differences in the tables are therefore run-to-run variation, not
effects of the optimizations.

**v2 is verified bit-identical to v1.** With `--liblinear_options "-m 1"`,
training is deterministic. Full CLI runs of v1 and v2 then produced
**byte-identical checkpoint files** (same SHA-256) and identical metrics on all
three datasets:

| Dataset | Checkpoint SHA-256 prefix (v1 = v2) | P@1 |
| --- | --- | ---: |
| EURLEX-4K | `7769296acbfa5df081a02e77` | 81.86 |
| EURLEX-57K | `1701ae018d59c4ac991b9fd5` | 91.55 |
| AmazonCat-13K | `eb83f3a839c89336b973c124` | 94.05 |

Each stage was also checked bitwise on its own:
- parsed data
- label trees and every k-means cluster assignment
- per-batch prediction matrices on all test batches
- metric accumulators
- trained weights under `-m 1`

Multithreaded GraphBLAS operations inside k-means make some intermediate
centroid values differ in the last bits between runs of v1 itself. Cluster
assignments are unaffected, and v2 shows the same noise.

## 4. What changed from Original to v1 (previous pass)

v1 mainly reduced memory, and the time reductions came largely from the same
changes.

- **Clustering:** k-means centroid shifts are computed in row blocks instead of
  full dense difference matrices. Clustering uses at most 8 threads, and the
  centroids are released before recursing into children.
- **Label representation:** the representation is built directly in CSR and
  normalized in place. The model-size estimate uses Boolean feature support
  instead of slicing whole node matrices.
- **Training:**
  - Each node trains only on the feature columns its instances use. The weights
    are mapped back to all features afterwards, which avoids dense
    features-by-labels weight matrices.
  - Labels are copied directly into C buffers.
  - `-m N` is honoured.
- **Assembly:** node weights are staged in one temporary file and assembled
  into the final CSC matrix in bounded chunks, instead of being held twice in
  RAM.
- **Prediction:** a CSR copy of the root and of each subtree is built once and
  reused across batches. This made prediction faster, but evaluation held two
  copies of the model.

## 5. What changed from v1 to v2 (this pass)

The rule throughout was that every floating-point operation keeps the same
operands and the same order, so results stay identical.

### Data loading

LIBSVM files of 64 MiB or more are split into chunks of whole lines and parsed
by 8 worker processes. The workers run the unchanged per-line parser. Error
types, messages and line numbers are preserved, including CR/CRLF line endings.

AmazonCat loading went from 218 s to 32 s.

### Tree building (k-means)

This was the memory peak on the two larger datasets. The GraphBLAS calls are
unchanged; only the copies around them and object lifetimes differ.

- Dense centroids are built directly. GraphBLAS's in-place densify staged about
  three times the dense size in pending tuples.
- Column-major centroid copies are a transpose plus a zero-copy import, instead
  of a duplicate, its transpose and another copy.
- Temporaries that python-graphblas keeps in reference cycles are collected as
  soon as they are released. Old centroids are freed after their last use.
- A large label representation is written to a temporary file while it is
  clustered. It is read back with identical arrays and SciPy flags, which
  GraphBLAS uses when importing.

On AmazonCat, the tree-building peak fell from 17.8 GB to 7.0 GB in isolation,
and time fell from about 595 s to about 380 s.

### Training

- **Queueing:** one pool of `-m` threads trains the labels of all nodes from a
  single queue, in node order and then label order. Previously, most of the 8
  threads idled on AmazonCat's ~3,000 small nodes, and at each node boundary.
  With one thread, LIBLINEAR is called in exactly the original order, which is
  why `-m 1` checkpoints are byte-identical.
- **Node preparation:** each node's instances and meta-labels are found from
  label columns instead of slicing all rows. Features are renumbered with a
  lookup array.

AmazonCat training went from 626 s to 503 s.

### Prediction and evaluation

- **Compiled loops:** compiled loops (Numba, already a dependency) replace the
  per-subtree sparse products and the per-instance Python beam search. They use
  the same operations in the same order:
  - SciPy's `csr_matmat` accumulation order
  - Python's stable `sorted(...)[:beam_width]`
  - `log_expit` and `exp` stay in SciPy and NumPy
- **In-place conversion:** the model is converted to per-subtree CSR blocks in
  place, over its own arrays, so evaluation holds one copy instead of two.
  Accessing or pickling the model restores the identical CSC arrays. A failed
  conversion rolls back.
- **Compact row pointers:** each block keeps row pointers only for the features
  it uses, through a bitmap and rank lookup. This reduced them from 814 MiB to
  100 MiB on EURLEX-57K.
- **Metrics:** NumPy's per-row partitions for top-k and PSP run on row blocks in
  threads, with identical per-row results. Each batch's metric update overlaps
  the next batch's prediction, still in batch order.

On EURLEX-57K, evaluation memory above the loaded model went from +6.4 GB to
+0.6 GB, and AmazonCat evaluation went from about 1400 s to about 390 s.

### Tests

`tests/linear/test_speed_equivalence.py` adds 11 tests that compare each fast
path bitwise with the code it replaces. These cover:
- prediction, including ties and NaN
- CSR conversion and rollback
- pipelined training, run in fresh processes
- parallel parsing and its error messages
- metric selection
- the k-means helpers and the representation spill

All 31 tests pass.

## 6. Caveats

- **First run:** the first run on a new checkout compiles and caches the Numba
  kernels, adding a few seconds.
- **Temporary disk:** large representations and blocks are staged in `TMPDIR`,
  which needs about one model's size of disk.
- **EURLEX-57K slowdown:** on this dataset, two clustering iterations
  occasionally take about 130 s instead of about 8 s. This happened once in
  each version and does not come from this code. A same-day v1 re-run hit it
  (572 s instead of 327 s), so the tables use your original v1 measurement.
