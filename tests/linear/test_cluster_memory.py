"""Compare bounded centroid reductions with the installed k-means algorithms."""

import unittest

import graphblas as gb
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sparsekmeans import ElkanKmeans, LloydKmeans
from libmultilabel.linear import cluster


class ClusterMemoryTests(unittest.TestCase):
    def test_centroid_shifts_match_for_dense_and_sparse_matrices(self):
        rng = np.random.default_rng(42)
        for density in (0.1, 1):
            current = sparse.random(9, 30, density=density, random_state=rng)
            previous = sparse.random(9, 30, density=density, random_state=rng)
            expected = LloydKmeans(n_clusters=9, n_threads=1)
            expected.centroids = gb.io.from_scipy_sparse(current)
            expected.old_centroids = gb.io.from_scipy_sparse(previous)
            actual = cluster.LloydKmeans(n_clusters=9, n_threads=1)
            actual.centroids = expected.centroids
            actual.old_centroids = expected.old_centroids
            actual._shift_block_bytes = 8 * 30 * 2  # exercise multiple blocks
            assert_allclose(actual._cal_centroids_shift(), expected._cal_centroids_shift(), rtol=1e-14)

    def test_lloyd_and_elkan_match_upstream_assignments(self):
        rng = np.random.default_rng(19)
        x = sparse.random(75, 40, density=0.4, random_state=rng, format="csr")
        for original, bounded in ((LloydKmeans, cluster.LloydKmeans), (ElkanKmeans, cluster.ElkanKmeans)):
            with self.subTest(algorithm=original.__name__):
                kwargs = dict(n_clusters=5, max_iter=15, random_state=42, n_threads=1)
                expected = original(**kwargs)
                actual = bounded(**kwargs)
                actual._shift_block_bytes = 1
                assert_array_equal(actual.fit(x), expected.fit(x))
                assert_allclose(
                    actual.centroids.to_dense(fill_value=0), expected.centroids.to_dense(fill_value=0), rtol=1e-12
                )


if __name__ == '__main__':
    unittest.main()
