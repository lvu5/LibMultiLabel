"""Faster code paths must give the same bits as the code they replace.

Run from this directory with the repository on PYTHONPATH:
    python -m unittest test_speed_equivalence
"""

import copy
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse

from libmultilabel.linear import cluster, data_utils, linear, metrics, tree


def assert_bits_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (actual.shape, expected.shape)
    assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def random_tree(rng, num_labels, degree, depth=0):
    """A random label tree in which leaves may be uneven, empty or deep."""
    labels = rng.permutation(num_labels)

    def build(label_map, d):
        if d >= 2 or len(label_map) <= degree or rng.random() < 0.2:
            return tree.Node(np.sort(label_map), [])
        parts = np.array_split(label_map, degree)
        return tree.Node(np.sort(label_map), [build(part, d + 1) for part in parts])

    if num_labels > degree:  # an internal root, so that predictions can prune
        root = tree.Node(np.sort(labels), [build(part, depth + 1) for part in np.array_split(labels, degree)])
    else:
        root = build(labels, depth)
    root.is_root = True
    return root


def trained_model(rng, num_labels=40, degree=4, num_features=30, ties=False):
    root = random_tree(rng, num_labels, degree)
    nodes = []
    root.dfs(nodes.append)
    for node in nodes:
        columns = len(node.label_map) if node.isLeaf() else len(node.children)
        weights = rng.normal(size=(num_features, columns))
        weights[rng.random(weights.shape) < 0.5] = 0
        if ties:
            weights = np.round(weights)
        node.model = linear.FlatModel("node", sparse.csc_matrix(weights), -1, 0, False)
    return tree.TreeModel(root, *tree._flatten_model(root))


def reference_predict(model, x, beam_width, prob_A=3):
    """TreeModel.predict_values without compiled kernels."""
    saved, tree._KERNELS = tree._KERNELS, False
    try:
        return model.predict_values(x, beam_width, prob_A)
    finally:
        tree._KERNELS = saved


class PredictionTests(unittest.TestCase):
    def test_compiled_prediction_matches_python(self):
        rng = np.random.default_rng(0)
        for trial in range(12):
            model = trained_model(rng, ties=trial % 3 == 0)
            reference = copy.deepcopy(model)
            x = sparse.random(37, 30, density=0.3, random_state=trial, format="csr")
            if trial % 4 == 1:
                x = x[:, :25]  # fewer test features than the model
            if trial % 4 == 2:
                x = sparse.hstack([x, sparse.random(37, 5, density=0.3, random_state=trial)], format="csr")
            if trial % 3 == 0:
                x.data = np.round(x.data)  # many tied scores
            for beam_width in (1, 2, 3, 4, 7, 100):
                with self.subTest(trial=trial, beam_width=beam_width):
                    assert_bits_equal(model.predict_values(x, beam_width), reference_predict(reference, x, beam_width))

    def test_nan_falls_back_to_python_order(self):
        rng = np.random.default_rng(1)
        model = trained_model(rng)
        reference = copy.deepcopy(model)
        x = sparse.random(10, 30, density=0.5, random_state=3, format="csr")
        x.data[3] = np.nan
        for beam_width in (1, 2, 100):
            assert_bits_equal(model.predict_values(x, beam_width), reference_predict(reference, x, beam_width))

    def test_blocks_round_trip_and_rollback(self):
        rng = np.random.default_rng(2)
        model = trained_model(rng)
        reference = copy.deepcopy(model)  # pickling a model restores its CSC weights first
        weights = model.flat_model.weights
        original = [weights.data.copy(), weights.indices.copy(), weights.indptr.copy()]
        x = sparse.random(9, 30, density=0.4, random_state=4, format="csr")
        expected = model.predict_values(x, 1)
        self.assertTrue(model._model_separated)
        self.assertFalse(np.array_equal(weights.indices, original[1]))  # blocks overwrote the CSC arrays
        # Unpruned prediction reuses the blocks.
        assert_bits_equal(model.predict_values(x, 100), reference_predict(reference, x, 100))
        self.assertTrue(model._model_separated)
        self.assertIs(model.flat_model.weights, weights)
        for actual, before in zip((weights.data, weights.indices, weights.indptr), original):
            assert_bits_equal(actual, before)

        calls = [0]
        transpose = tree._CompressedTranspose.__call__

        def fail_on_third(self, *args):
            calls[0] += 1
            if calls[0] == 3:
                raise MemoryError("injected")
            return transpose(self, *args)

        with unittest.mock.patch.object(tree._CompressedTranspose, "__call__", fail_on_third):
            with self.assertRaises(MemoryError):
                model.predict_values(x, 1)
        self.assertFalse(model._model_separated)
        for actual, before in zip((weights.data, weights.indices, weights.indptr), original):
            assert_bits_equal(actual, before)
        assert_bits_equal(model.predict_values(x, 1), expected)

    def test_checkpoint_bytes_do_not_depend_on_prediction(self):
        rng = np.random.default_rng(3)
        model = trained_model(rng)
        before = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        model.predict_values(sparse.random(5, 30, density=0.4, random_state=1, format="csr"), 1)
        self.assertEqual(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL), before)

    def test_compressed_transpose_matches_scipy(self):
        rng = np.random.default_rng(4)
        for shape in [(0, 3), (4, 0), (1, 1), (60, 7), (500, 90)]:
            for density in (0.0, 0.05, 0.5):
                matrix = sparse.random(*shape, density=density, random_state=int(rng.integers(1 << 30)), format="csc")
                if matrix.nnz > 2:
                    # Non-canonical: reversed rows within each column, and the first entry repeated.
                    data, indices, indptr = [], [], [0]
                    for c in range(shape[1]):
                        rows = matrix.indices[matrix.indptr[c] : matrix.indptr[c + 1]][::-1]
                        values = matrix.data[matrix.indptr[c] : matrix.indptr[c + 1]][::-1]
                        if rows.size:
                            rows, values = np.r_[rows, rows[:1]], np.r_[values, values[:1]]
                        indices.extend(rows)
                        data.extend(values)
                        indptr.append(len(indices))
                    matrix = sparse.csc_matrix((np.array(data), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)), shape=shape)
                expected = matrix.tocsr()
                for aliased, through_file in ((False, False), (True, False), (True, True)):
                    with self.subTest(shape=shape, density=density, aliased=aliased, through_file=through_file), \
                            unittest.mock.patch.object(tree, "_TRANSPOSE_IN_MEMORY_MAX_NNZ", 0 if through_file else 1 << 24), \
                            unittest.mock.patch.object(tree, "_TRANSPOSE_CHUNK_NNZ", 3):
                        transpose = tree._CompressedTranspose(shape[0], matrix.indptr.dtype, aliased)
                        indptr = np.empty(shape[0] + 1, dtype=matrix.indptr.dtype)
                        indices, data = matrix.indices.copy(), matrix.data.copy()
                        out_indices, out_data = (indices, data) if aliased else (np.empty_like(indices), np.empty_like(data))
                        transpose(shape[0], matrix.indptr, indices, data, indptr, out_indices, out_data)
                        assert_array_equal(indptr, expected.indptr)
                        assert_array_equal(out_indices, expected.indices)
                        assert_bits_equal(out_data, expected.data)


class TrainingTests(unittest.TestCase):
    def test_rows_with_entries_matches_getnnz(self):
        rng = np.random.default_rng(5)
        for shape in [(1, 1), (40, 9), (300, 50)]:
            y = sparse.random(*shape, density=0.1, random_state=int(rng.integers(1 << 30)), format="csr")
            y.data[: y.nnz // 2] = 0  # explicit zeros are entries
            for labels in (np.arange(0), np.arange(shape[1]), np.sort(rng.choice(shape[1], max(1, shape[1] // 3), replace=False))):
                assert_array_equal(tree._rows_with_entries(y.tocsc(), labels), y[:, labels].getnnz(axis=1) > 0)

    def test_pipelined_training_matches_node_by_node_training(self):
        # LIBLINEAR's dual solvers shuffle with a random state that persists
        # across calls and cannot be reset, so each training runs in a fresh
        # process. With one thread both call LIBLINEAR in the same order.
        script = """
import copy, pickle, sys
import numpy as np
from scipy import sparse
from libmultilabel.linear import tree
from test_speed_equivalence import random_tree
mode, options, out = sys.argv[1:4]
rng = np.random.default_rng(6)
x = sparse.random(120, 40, density=0.2, random_state=7, format="csr")
y = sparse.csr_matrix((rng.random((120, 30)) < 0.15).astype(np.float64))
root = random_tree(np.random.default_rng(8), 30, 3)
if mode == "pipelined":
    model = tree.train_tree(y, x, options, root=root, verbose=False)
    weights, node_ptr = model.flat_model.weights, model.node_ptr
else:
    def train_node(node):
        if node.is_root:
            tree._train_node(y, x, options, node)
        else:
            relevant = y[:, node.label_map].getnnz(axis=1) > 0
            tree._train_node(y[relevant], x[relevant], options, node)
    flat, node_ptr = tree._flatten_model(root, train_node)
    weights = flat.weights
with open(out, "wb") as f:
    pickle.dump((weights.data, weights.indices, weights.indptr, node_ptr), f)
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), *sys.path]))
        with tempfile.TemporaryDirectory() as directory:
            for options in ("-s 1 -m 1 -q", "-s 2 -B 1 -m 1 -q", "-s 3 -m 1 -q -c 0.5"):
                results = []
                for mode in ("node_by_node", "pipelined"):
                    out = os.path.join(directory, f"{mode}.pkl")
                    subprocess.run([sys.executable, "-c", script, mode, options, out], check=True, env=env, capture_output=True)
                    with open(out, "rb") as f:
                        results.append(pickle.load(f))
                with self.subTest(options=options):
                    for actual, expected in zip(results[1], results[0]):
                        assert_bits_equal(actual, expected)


class DataTests(unittest.TestCase):
    def test_parallel_reading_matches_sequential(self):
        lines = ["1,2 1:0.5 3:0.25", "3 2:1e-3 4:0 5:-7", " 4:2", "7 1:+0.0 2:-0.0 3:1e400", "1:1 2:2", "2,3"] * 40
        cases = {
            "valid": "\n".join(lines) + "\n",
            "crlf_and_cr": "\r\n".join(lines[:50]) + "\r" + "\r".join(lines[50:]),
            "bad_float_late": "\n".join(lines) + "\n2 1:x 0:1\n",
            "zero_index_late": "\n".join(lines) + "\n2 0:1 1:x\n",
            "empty_line": "\n".join(lines[:100]) + "\n\n" + "\n".join(lines[100:]),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, text in cases.items():
                path = os.path.join(directory, f"{name}.svm")
                with open(path, "w", newline="") as f:
                    f.write(text)

                def read(parallel):
                    with unittest.mock.patch.object(data_utils, "_PARALLEL_READ_MIN_BYTES", 0 if parallel else 1 << 62), \
                            unittest.mock.patch.object(data_utils, "_READ_CHUNK_BYTES", 97):
                        try:
                            result = data_utils._read_libsvm_format(path)
                        except Exception as error:
                            return type(error), str(error)
                    x = result["x"]
                    return x.shape, x.indices.dtype, x.indptr.tolist(), x.indices.tolist(), x.data.view(np.uint64).tolist(), result["y"]

                with self.subTest(case=name):
                    self.assertEqual(read(True), read(False))


class MetricTests(unittest.TestCase):
    def test_parallel_selection_matches_numpy(self):
        rng = np.random.default_rng(9)
        array = np.round(rng.random((700, 2000)) * 4) / 4  # heavy ties
        with unittest.mock.patch.object(metrics, "_PARALLEL_SELECT_MIN_SIZE", 1):
            for select in (np.partition, np.argpartition):
                for k in (1, 3, 5):
                    assert_bits_equal(metrics._select_last_columns(select, array, -k, k), select(array, -k, axis=-1)[:, -k:])


class ClusterTests(unittest.TestCase):
    def test_spilled_representation_builds_the_same_tree(self):
        rng = np.random.default_rng(11)
        representation = sparse.random(300, 80, density=0.2, random_state=12, format="csr")

        def labels_of(root):
            nodes = []
            root.dfs(lambda node: nodes.append((node.label_map.tolist(), len(node.children))))
            return nodes

        np.random.seed(5)
        expected = tree._build_tree(representation.copy(), np.arange(300), 0, 4, 3)
        with unittest.mock.patch.object(tree, "_SPILL_MIN_NNZ", 0):
            np.random.seed(5)
            actual = tree._build_tree([representation.copy()], np.arange(300), 0, 4, 3)
        self.assertEqual(labels_of(actual), labels_of(expected))

    def test_column_major_copy_and_densify_match_graphblas(self):
        import graphblas as gb

        rng = np.random.default_rng(10)
        for shape in [(1, 7), (12, 30), (60, 500)]:
            for density in (0.0, 0.01, 0.04, 0.15, 0.5):
                A = sparse.random(*shape, density=density, random_state=int(rng.integers(1 << 30)), format="csr")
                expected = gb.io.from_scipy_sparse(A)
                expected(~expected.S) << 0
                expected.wait()
                actual = cluster._densify(gb.io.from_scipy_sparse(A))
                actual.wait()
                self.assertTrue(actual.isequal(expected))
                self.assertEqual(actual.ss.format, expected.ss.format)
                reference = gb.Matrix.ss.import_fullc(**expected.ss.export("fullc"))
                copied = cluster._column_major_copy(expected)
                self.assertTrue(copied.isequal(reference))
                self.assertEqual(copied.ss.format, reference.ss.format)


if __name__ == "__main__":
    unittest.main()
