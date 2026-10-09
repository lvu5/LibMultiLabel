"""Sparse k-means adapters with bounded memory.

The upstream implementation constructs both a full centroid difference matrix
and its elementwise square. For 100 clusters and 2.66 million features, those
float64 temporaries alone occupy about 4 GiB. Compute the same row reductions
in small blocks; clustering, initialization and stopping criteria stay intact.

Dense centroids are also copied more often than necessary. Every arithmetic
GraphBLAS call below receives the same operands, in the same storage format, as
upstream; only temporary copies and object lifetimes differ:

* python-graphblas temporaries can live in reference cycles, so large ones
  (such as ``X ** 2``) survive until the cyclic garbage collector runs.
* A column-major copy for ``X @ C.T`` is a transpose plus a zero-copy import,
  instead of a duplicate, its transpose and an import copy.
* Sparse centroids are densified in place with ``C(~C.S) << 0``. When C is
  sparse enough, GraphBLAS stages every new entry as a pending tuple (about
  three times the dense size) and then assembles a full row-major matrix.
  Build that full matrix directly in that regime.
* Elkan's previous centroids are only needed for the centroid shift.
"""

import gc
import sys
import time

import graphblas as gb
import numpy as np
from graphblas import Matrix, dtypes
from sparsekmeans import ElkanKmeans as _ElkanKmeans, LloydKmeans as _LloydKmeans
from sparsekmeans.sparse_kmeans import squared_row_norms

# (minimum rows, maximum density) pairs at which C(~C.S) << 0 takes GraphBLAS's
# pending-tuple path and becomes a full row-major matrix. Denser inputs switch to
# a bitmap. Measured with SuiteSparse:GraphBLAS 9.4: a single row switches at a
# density of 0.05, 10 rows above 0.10 and 50 or more rows above 0.25.
_DIRECT_DENSIFY_LIMITS = ((50, 0.20), (10, 0.05))


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
        gc.collect()
        return shifts


class _BoundedKmeans(_BoundedCentroidShift):
    """Release cyclic GraphBLAS temporaries and avoid staging dense centroids."""

    def _single_kmeans(self, X):
        # sparsekmeans.SparseKmeans._single_kmeans, except for the density check.
        self._setup_internal_state(X)

        self.mean_feature_variance = self._cal_feature_variance(X)
        if self.mean_feature_variance <= sys.float_info.min:
            return

        for iter in range(self.max_iter):
            start_iter = time.time()

            self.centroids, self.is_centroid_dense = _check_centroids_density(self.centroids)

            if self.use_gpu:
                self._assign_cluster(self.gpu_X)
            else:
                self._assign_cluster(X)

            if iter > 0:
                if self._converged():
                    break

            self.old_centroids = self.centroids
            self.centroids = self._update_centroids(X)

            self._update_internal_state()

            end_iter = time.time()
            if self.verbose:
                print(
                    f"Time to conduct iteration: {iter}", end_iter - start_iter, flush=True
                )

        self.is_fitted = True

        return

    def fit(self, X):
        try:
            return super().fit(X)
        finally:
            gc.collect()

    def _initialize_centroids(self, X):
        centroids = super()._initialize_centroids(X)
        gc.collect()
        return centroids

    def _cal_feature_variance(self, X):
        variance = super()._cal_feature_variance(X)
        gc.collect()
        return variance

    def _update_centroids(self, X):
        # The centroids before old_centroids were just released; free them
        # before allocating the new centroids.
        gc.collect()
        return super()._update_centroids(X)


def _column_major_copy(centroids: Matrix) -> Matrix:
    """Equivalent to ``import_fullc(**centroids.ss.export("fullc"))``.

    The transpose stores C.T by rows, which is exactly the column-major layout of
    C. Moving that buffer into a "fullc" matrix keeps one copy instead of three.
    """
    transposed = centroids.T.new()
    if transposed.ss.format != "fullr":
        del transposed
        return Matrix.ss.import_fullc(**centroids.ss.export("fullc"), take_ownership=True)
    exported = transposed.ss.unpack("fullr", raw=True)
    del transposed
    return Matrix.ss.import_fullc(
        values=exported["values"],
        nrows=centroids.nrows,
        ncols=centroids.ncols,
        is_iso=exported["is_iso"],
        take_ownership=True,
    )


def _predict_labels_dense(X, centroids, n_threads):
    """sparsekmeans.predict_labels for dense centroids on the CPU, with fewer copies."""
    if X.dtype != dtypes.FP64:
        X = X.dup(dtype="FP64")
    if centroids.dtype != dtypes.FP64:
        centroids = centroids.dup(dtype="FP64")
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]

    c_squared_norms = squared_row_norms(centroids, n_threads)
    gc.collect()

    XCt = Matrix(dtypes.FP64, nrows=n_samples, ncols=n_clusters)
    centroids = _column_major_copy(centroids)
    XCt << 0
    XCt(accum=gb.binary.plus, nthreads=n_threads) << X.mxm(centroids.T)
    del centroids
    XCt = 2 * XCt.to_dense()

    distances_to_centroids = -XCt + c_squared_norms[np.newaxis, :]
    labels = np.argmin(distances_to_centroids, axis=1)

    return labels, distances_to_centroids


def _densify(centroids: Matrix) -> Matrix:
    """Return ``centroids`` with explicit zeros, as ``C(~C.S) << 0`` produces."""
    nrows, ncols = centroids.shape
    nvals = centroids.nvals
    if (
        centroids.ss.format not in {"csr", "hypercsr"}
        or centroids.dtype != dtypes.FP64
        or centroids.ss.is_iso
        or nvals == 0
        or not any(nrows >= rows and nvals <= density * nrows * ncols for rows, density in _DIRECT_DENSIFY_LIMITS)
    ):
        centroids(~centroids.S) << 0
        return centroids

    sparse = centroids.ss.export("csr")
    row_nnz = np.diff(sparse["indptr"]).astype(np.int64, copy=False)
    positions = np.repeat(np.arange(nrows, dtype=np.int64), row_nnz)
    positions *= ncols
    positions += sparse["col_indices"][:nvals].astype(np.int64, copy=False)
    values = np.zeros(nrows * ncols, dtype=np.float64)
    values[positions] = sparse["values"][:nvals]
    del sparse, row_nnz, positions
    return Matrix.ss.import_fullr(values=values, nrows=nrows, ncols=ncols, take_ownership=True)


def _check_centroids_density(centroids: Matrix):
    """sparsekmeans.check_centroids_density with a direct densification."""
    gamma = 0.05

    if centroids.ss.format == "fullr":
        is_centroid_dense = True
        centroids_density = centroids.V.new().nvals / (centroids.nrows * centroids.ncols)
        if centroids_density <= gamma:
            is_centroid_dense = False
            centroids(mask=centroids.V, replace=True) << centroids
    else:
        is_centroid_dense = False
        centroids_density = centroids.nvals / (centroids.nrows * centroids.ncols)
        if centroids_density > gamma:
            is_centroid_dense = True
            centroids = _densify(centroids)

    return centroids, is_centroid_dense


class LloydKmeans(_BoundedKmeans, _LloydKmeans):
    def _assign_cluster(self, X):
        if self.use_gpu or not self.is_centroid_dense:
            return super()._assign_cluster(X)
        self.labels, distances_to_centroids = _predict_labels_dense(X, self.centroids, self.n_threads)
        np.min(distances_to_centroids, axis=1, out=self.sample_centroids_closest_distance)
        del distances_to_centroids
        gc.collect()


class ElkanKmeans(_BoundedKmeans, _ElkanKmeans):
    def _assign_cluster(self, X):
        # sparsekmeans.ElkanKmeans._assign_cluster, except for the densification.
        n_samples = X.shape[0]
        n_clusters = self.centroids.shape[0]

        samples_idx = np.arange(n_samples)

        c_squared_norms = squared_row_norms(self.centroids, self.n_threads)

        CCt = Matrix(dtypes.FP64, nrows=n_clusters, ncols=n_clusters)
        CCt(nthreads=self.n_threads) << self.centroids.mxm(self.centroids.T)
        CCt = CCt.to_dense(fill_value=0)

        half_centroid_centroid_distances = c_squared_norms[:, np.newaxis] - 2 * CCt + c_squared_norms[np.newaxis, :]
        np.clip(half_centroid_centroid_distances, 0, None, out=half_centroid_centroid_distances)
        half_centroid_centroid_distances = 0.5 * np.sqrt(half_centroid_centroid_distances)

        samples_centroids_product = Matrix(dtypes.FP64, nrows=n_samples, ncols=n_clusters)
        samples_centroids_mask = Matrix.from_coo(
            np.arange(n_samples), self.labels, values=1, nrows=n_samples, ncols=n_clusters
        )

        if self.centroids.ss.format != "fullr":
            self.centroids = _densify(self.centroids)
        gc.collect()

        samples_centroids_product(mask=samples_centroids_mask.S, nthreads=self.n_threads) << X.mxm(self.centroids.T)
        samples_centroids_product = samples_centroids_product.reduce_rowwise(gb.binary.min).to_dense(fill_value=0)

        self.sample_centroids_closest_distance = self.x_squared_norms - 2 * samples_centroids_product + c_squared_norms[self.labels]
        np.clip(self.sample_centroids_closest_distance, 0, None, out=self.sample_centroids_closest_distance)
        self.sample_centroids_closest_distance = np.sqrt(self.sample_centroids_closest_distance)

        for j in range(n_clusters):
            candidate_idx = samples_idx[(self.labels != j) & (self.sample_centroids_closest_distance > self.lower_bounds[j, :])]
            candidate_idx = candidate_idx[self.sample_centroids_closest_distance[candidate_idx] > half_centroid_centroid_distances[j, self.labels[candidate_idx]]]

            if len(candidate_idx) == 0:
                continue

            Xcj = gb.Vector(dtypes.FP64, size=n_samples)
            Xcj_mask = gb.Vector.from_coo(candidate_idx, 1, size=n_samples)

            Xcj(mask=Xcj_mask.S, nthreads=self.n_threads) << X.mxv(self.centroids[j, :])
            Xcj = Xcj.to_dense(fill_value=0)
            distance_to_cj = self.x_squared_norms[candidate_idx] - 2 * Xcj[candidate_idx] + c_squared_norms[j]
            np.clip(distance_to_cj, 0, None, out=distance_to_cj)
            distance_to_cj = np.sqrt(distance_to_cj)
            self.lower_bounds[j, candidate_idx] = distance_to_cj

            reassignment_mask = self.sample_centroids_closest_distance[candidate_idx] > self.lower_bounds[j, candidate_idx]

            samples_to_reassign_idx = candidate_idx[reassignment_mask]

            if len(samples_to_reassign_idx) == 0:
                continue

            self.labels[samples_to_reassign_idx] = j
            update_distances = distance_to_cj[reassignment_mask]
            self.sample_centroids_closest_distance[samples_to_reassign_idx] = update_distances

        gc.collect()

    def _update_internal_state(self):
        super()._update_internal_state()
        # The shift above is the last use of the previous centroids.
        self.old_centroids = None
        gc.collect()
