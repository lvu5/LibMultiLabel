"""Regression coverage for memory use outside final tree assembly."""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from libmultilabel.linear import linear, tree
from test_tree_memory import make_tree


class TrainingMemoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.x = sparse.csr_matrix(rng.normal(size=(30, 6)))
        labels = rng.integers(0, 2, size=(30, 5))
        labels[:, 0] = 0
        labels[:, 1] = 1
        self.y = sparse.csr_matrix(labels)

    def test_sparse_training_matches_dense_with_constant_labels_and_bias(self):
        for bias in (-1, 1):
            options = f"-s 2 -m 2 -B {bias} -q" if bias > 0 else "-s 2 -m 2 -q"
            dense = linear.train_1vsrest(self.y, self.x, options=options, verbose=False)
            for dtype in (bool, np.uint8, np.float64):
                with self.subTest(bias=bias, dtype=dtype):
                    actual = linear.train_1vsrest(
                        self.y.astype(dtype), self.x, options=options, verbose=False, sparse_output=True
                    )
                    self.assertTrue(sparse.isspmatrix_csc(actual.weights))
                    assert_allclose(actual.weights.toarray(), dense.weights, rtol=1e-12, atol=1e-12)
                    assert_allclose(actual.predict_values(self.x), dense.predict_values(self.x), atol=1e-12)

    def test_node_feature_projection_preserves_weights_and_bias(self):
        x = sparse.hstack([sparse.csr_matrix((30, 17)), self.x, sparse.csr_matrix((30, 9))], format="csr")
        for bias in ("", " -B 1"):
            options = "-s 2 -m 1" + bias
            expected = linear.train_1vsrest(self.y, x, options=options, verbose=False)
            node = tree.Node(np.arange(self.y.shape[1]), [])
            original = linear.train_1vsrest
            dimensions = []

            def train(y, features, *args, **kwargs):
                dimensions.append(features.shape[1])
                return original(y, features, *args, **kwargs)

            with patch.object(linear, "train_1vsrest", side_effect=train):
                tree._train_node(self.y, x, options, node)
            self.assertEqual(dimensions, [6])
            assert_allclose(node.model.weights.toarray(), expected.weights, atol=1e-12, rtol=1e-12)
            assert_allclose(node.model.predict_values(x), expected.predict_values(x), atol=1e-12)

    def test_worker_count_honors_m_and_number_of_labels(self):
        original_start = linear.ParallelOVRTrainer.start
        for requested, expected in ((1, 1), (2, 2), (12, 5)):
            starts = []

            def start(worker):
                starts.append(worker)
                original_start(worker)

            with patch.object(linear.ParallelOVRTrainer, "start", start):
                linear.train_1vsrest(self.y, self.x, options=f"-s 2 -m {requested}", verbose=False)
            self.assertEqual(len(starts), expected)

    def test_default_thread_limit_and_invalid_thread_count(self):
        y = sparse.hstack([self.y] * 3, format="csr")
        started = []
        original_start = linear.ParallelOVRTrainer.start

        def start(worker):
            started.append(worker)
            original_start(worker)

        with patch.object(linear.psutil, "cpu_count", return_value=48), patch.object(
            linear.ParallelOVRTrainer, "start", start
        ):
            linear.train_1vsrest(y, self.x, options="-s 2", verbose=False)
        self.assertEqual(len(started), 8)
        with self.assertRaisesRegex(ValueError, "at least one"):
            linear.train_1vsrest(y, self.x, options="-m 0", verbose=False)

    def test_worker_failure_propagates_and_next_training_succeeds(self):
        with patch.object(linear.ParallelOVRTrainer, "_do_parallel_train", side_effect=RuntimeError("solver failed")):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                linear.train_1vsrest(self.y, self.x, options="-m 2", verbose=False, sparse_output=True)
        self.assertFalse(hasattr(linear.ParallelOVRTrainer, "prob"))
        model = linear.train_1vsrest(self.y, self.x, options="-s 2 -m 1", verbose=False)
        self.assertEqual(model.weights.shape, (6, 5))

    def test_sparse_empty_inputs(self):
        for instances, labels in ((0, 3), (3, 0)):
            model = linear.train_1vsrest(
                sparse.csr_matrix((instances, labels)), sparse.csr_matrix((instances, 6)),
                options="-m 1", verbose=False, sparse_output=True,
            )
            self.assertEqual(model.weights.shape, (6, labels))
            self.assertEqual(model.weights.nnz, 0)

    def test_feature_counts_match_original_with_signed_and_explicit_zero_features(self):
        root, nodes = make_tree()
        x = self.x.copy()
        x.data[::3] = 0
        y = self.y[:, :4]
        expected = (x != 0).T @ y
        count = tree._count_node_features(root, y, x)
        self.assertEqual(count, len(nodes))
        for node in nodes:
            used = np.count_nonzero(expected[:, node.label_map].sum(axis=1))
            self.assertEqual(node.num_features_used, used)

    def test_pruning_csr_cache_is_reused_and_matches_full_predictions(self):
        root, _ = make_tree()
        model = tree.TreeModel(root, *tree._flatten_model(root))
        x = sparse.csr_matrix(np.random.default_rng(8).normal(size=(7, 4)))
        flat_weights = model.flat_model.weights
        arrays = [flat_weights.data.copy(), flat_weights.indices.copy(), flat_weights.indptr.copy()]
        self.assertFalse(model._model_separated)
        batches = (x[:3], x[3:])
        expected = {
            (beam, i): np.vstack([model._beam_search(row, beam, 3) for row in model.flat_model.predict_values(batch)])
            for beam in (1, 4) for i, batch in enumerate(batches)
        }
        with patch.object(model, "_separate_model_for_pruning_tree", wraps=model._separate_model_for_pruning_tree) as build:
            for beam in (1, 4, 1):
                for i, batch in enumerate(batches):
                    assert_array_equal(model.predict_values(batch, beam), expected[beam, i])
            self.assertEqual(build.call_count, 1)
        for block in [model.root_model, *model.subtree_models]:
            self.assertTrue(sparse.isspmatrix_csr(block.weights))
        # The CSR blocks overwrite the flattened weights; accessing them restores the CSC arrays.
        self.assertIs(model.flat_model.weights, flat_weights)
        self.assertFalse(model._model_separated)
        for actual, original in zip((flat_weights.data, flat_weights.indices, flat_weights.indptr), arrays):
            assert_array_equal(actual, original)

    def test_csr_cache_handles_int64_source_indices(self):
        root, _ = make_tree()
        model = tree.TreeModel(root, *tree._flatten_model(root))
        weights = model.flat_model.weights
        weights.indices = weights.indices.astype(np.int64)
        weights.indptr = weights.indptr.astype(np.int64)
        x = sparse.csr_matrix(np.ones((2, 4)))
        full = model.flat_model.predict_values(x)
        expected = np.vstack([model._beam_search(row, 1, 3) for row in full])
        assert_allclose(model.predict_values(x, 1), expected, rtol=1e-12)
        self.assertEqual(weights.indices.dtype, np.int64)
        self.assertEqual(weights.indptr.dtype, np.int64)
        for block in [model.root_model, *model.subtree_models]:
            self.assertTrue(sparse.isspmatrix_csr(block.weights))
            self.assertEqual(block.weights.indices.dtype, block.weights.indptr.dtype)

    def test_checkpoint_after_prediction_drops_derived_caches(self):
        root, _ = make_tree()
        model = tree.TreeModel(root, *tree._flatten_model(root))
        before = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        x = sparse.csr_matrix(np.ones((2, 4)))
        expected = model.predict_values(x, beam_width=1)
        after = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertEqual(len(before), len(after))
        restored = pickle.loads(after)
        self.assertFalse(restored._model_separated)
        self.assertFalse(hasattr(restored, "subtree_models"))
        assert_array_equal(restored.predict_values(x, beam_width=1), expected)
        self.assertTrue(sparse.isspmatrix_csr(restored.root_model.weights))


if __name__ == "__main__":
    unittest.main()
