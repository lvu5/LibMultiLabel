"""Sparse k-means adapters with bounded centroid-shift workspaces.

The upstream implementation constructs both a full centroid difference matrix
and its elementwise square. For 100 clusters and 2.66 million features, those
float64 temporaries alone occupy about 4 GiB. Compute the same row reductions
in small blocks; clustering, initialization and stopping criteria stay intact.
"""

import numpy as np
from sparsekmeans import ElkanKmeans as _ElkanKmeans, LloydKmeans as _LloydKmeans


class _BoundedCentroidShift:
    _shift_block_bytes = 16 * 1024**2

    def _cal_centroids_shift(self):
        clusters, features = self.centroids.shape
        rows = max(1, self._shift_block_bytes // max(8 * features, 1))
        shifts = np.empty(clusters, dtype=np.float64)
        for start in range(0, clusters, rows):
            stop = min(start + rows, clusters)
            current = self.centroids[start:stop, :].new()
            previous = self.old_centroids[start:stop, :].new()
            difference = current.ewise_union(
                previous, op="minus", left_default=0, right_default=0
            ).new(nthreads=self.n_threads)
            del current, previous
            squared = difference.ewise_mult(difference, op="times").new(nthreads=self.n_threads)
            del difference
            shifts[start:stop] = squared.reduce_rowwise("plus").new(nthreads=self.n_threads).to_dense(fill_value=0)
            del squared
        np.sqrt(shifts, out=shifts)
        return shifts


class LloydKmeans(_BoundedCentroidShift, _LloydKmeans):
    pass


class ElkanKmeans(_BoundedCentroidShift, _ElkanKmeans):
    pass
