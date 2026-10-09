from __future__ import annotations

import collections
import gc
import queue
import re
import tempfile
import threading
from typing import Callable, Iterable

import graphblas as gb
import numpy as np
import scipy.sparse as sparse
from scipy.special import log_expit
import sklearn.preprocessing
from tqdm import tqdm
import psutil

from . import linear
from .cluster import LloydKmeans, ElkanKmeans
from .memory import log_memory

__all__ = ["train_tree", "TreeModel", "train_ensemble_tree", "EnsembleTreeModel"]

DEFAULT_K = 100
DEFAULT_DMAX = 10
_WEIGHT_CHUNK_SIZE = 1 << 20  # At most 8 MiB per chunk for float64 weights or int64 indices.
_KERNELS = None
# Larger blocks are transposed in place through a temporary file, not a second in-memory copy.
_TRANSPOSE_IN_MEMORY_MAX_NNZ = 1 << 24
_TRANSPOSE_CHUNK_NNZ = 1 << 22
# Larger label representations are kept on disk while clustered.
_SPILL_MIN_NNZ = 1 << 24


def _prediction_kernels():
    """The compiled prediction kernels, or None if Numba is unavailable."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from . import tree_kernels
        except ImportError:
            tree_kernels = False
        _KERNELS = tree_kernels
    return _KERNELS or None


class _CompressedTranspose:
    """Transpose compressed (CSR or CSC) arrays into given output arrays.

    The result is SciPy's conversion between CSR and CSC: each output vector
    lists its entries in increasing input-vector order. If the outputs may alias
    the inputs, each result is first written to a temporary of its own size.
    """

    def __init__(self, max_num_out: int, index_dtype, aliased: bool):
        self.kernels = _prediction_kernels()
        self.aliased = aliased
        if self.kernels is not None:
            self.num_chunks = self.kernels.num_threads()
            # Counts and cursors never exceed the number of entries, which index_dtype holds.
            self.counts = np.empty(self.num_chunks * max_num_out, dtype=index_dtype)

    def __call__(self, num_out, in_ptr, in_idx, in_val, out_ptr, out_idx, out_val):
        nnz = len(in_idx)
        if self.aliased and self.kernels is not None and nnz > _TRANSPOSE_IN_MEMORY_MAX_NNZ:
            self._transpose_through_file(num_out, in_ptr, in_idx, in_val, out_ptr, out_idx, out_val)
            return
        if self.aliased:
            target_idx, target_val = np.empty_like(out_idx), np.empty_like(out_val)
        else:
            target_idx, target_val = out_idx, out_val
        num_in = len(in_ptr) - 1
        if self.kernels is None:
            matrix = sparse.csr_matrix((num_in, num_out), dtype=in_val.dtype)
            matrix.data, matrix.indices, matrix.indptr = in_val, in_idx, in_ptr
            matrix = matrix.tocsc()
            out_ptr[:], target_idx[:], target_val[:] = matrix.indptr, matrix.indices, matrix.data
            del matrix
        else:
            # Consecutive input vectors with about the same number of entries each.
            cuts = np.searchsorted(in_ptr, np.linspace(0, nnz, self.num_chunks + 1)[1:-1])
            chunks = np.unique(np.concatenate(([0], cuts, [num_in]))).astype(np.int64)
            counts = self.counts[: (len(chunks) - 1) * num_out].reshape(len(chunks) - 1, num_out)
            self.kernels.run(
                self.kernels.compressed_transpose, num_out, in_ptr, in_idx, in_val, out_ptr, target_idx, target_val, chunks, counts
            )
        if self.aliased:
            out_idx[:] = target_idx
            out_val[:] = target_val

    def _transpose_through_file(self, num_out, in_ptr, in_idx, in_val, out_ptr, out_idx, out_val):
        """The same counting sort, reading the input back from a temporary file in
        chunks instead of holding a second copy of a large block in memory."""
        out_ptr[0] = 0
        out_ptr[1:] = np.cumsum(np.bincount(in_idx, minlength=num_out))
        cursor = out_ptr[:-1].astype(np.int64)
        base = int(in_ptr[0])
        with tempfile.TemporaryFile(prefix="libmultilabel-blocks-") as staged:
            staged.write(np.ascontiguousarray(in_idx).data.cast("B"))
            staged.write(np.ascontiguousarray(in_val).data.cast("B"))
            values_at = len(in_idx) * in_idx.dtype.itemsize
            num_in = len(in_ptr) - 1
            first = 0
            while first < num_in:
                # Whole input vectors with at most _TRANSPOSE_CHUNK_NNZ entries, or a single larger one.
                last = int(np.searchsorted(in_ptr, in_ptr[first] + _TRANSPOSE_CHUNK_NNZ, side="right")) - 1
                last = min(max(last, first + 1), num_in)
                lo, hi = int(in_ptr[first]) - base, int(in_ptr[last]) - base
                idx = np.empty(hi - lo, dtype=in_idx.dtype)
                val = np.empty(hi - lo, dtype=in_val.dtype)
                staged.seek(lo * in_idx.dtype.itemsize)
                _read_staged(staged, idx, idx.dtype)
                staged.seek(values_at + lo * in_val.dtype.itemsize)
                _read_staged(staged, val, val.dtype)
                self.kernels.scatter_transposed(first, in_ptr[first : last + 1], idx, val, cursor, out_idx, out_val)
                del idx, val
                first = last


# The block models have zero thresholds, as FlatModel.predict_values adds them.
_BLOCK_THRESHOLDS = 0


def _compress_row_pointers(indptr: np.ndarray, bits: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Pointers of the nonempty rows of CSR row pointers ``indptr``, setting the
    bitmap of nonempty rows and the number of nonempty rows before each word."""
    nonempty = indptr[1:] != indptr[:-1]
    padded = np.zeros(bits.size * 64, dtype=np.bool_)
    padded[: nonempty.size] = nonempty
    padded = padded.reshape(bits.size, 64)
    powers = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    bits[:] = np.bitwise_or.reduce(np.where(padded, powers, np.uint64(0)), axis=1)
    per_word = padded.sum(axis=1)
    rank[0] = 0
    np.cumsum(per_word[:-1], out=rank[1:])
    return np.append(indptr[:-1][nonempty], indptr[-1])


def _blocks_dot(kernels, layout: dict, x: sparse.csr_matrix, blocks: np.ndarray, out: np.ndarray):
    """Add x[i] @ W_b to the columns of block b in out, for every block b in blocks[i]."""
    kernels.run(
        kernels.blocks_dot,
        x.indptr,
        x.indices,
        x.data,
        blocks,
        layout["offset"],
        layout["first_column"],
        layout["row_bits"],
        layout["row_rank"],
        layout["row_ptr"],
        layout["row_ptr_offset"],
        layout["indices"],
        layout["data"],
        out,
    )


class Node:
    def __init__(
        self,
        label_map: np.ndarray,
        children: list[Node],
    ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
        """
        self.label_map = label_map
        self.children = children
        self.is_root = False

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: Callable[[Node], None]):
        visit(self)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            child.dfs(visit)


class TreeModel:
    """A model returned from train_tree."""

    def __init__(
        self,
        root: Node,
        flat_model: linear.FlatModel,
        node_ptr: np.ndarray,
    ):
        self.name = "tree"
        self.root = root
        self.flat_model = flat_model
        self.node_ptr = node_ptr
        self.multiclass = False
        self._model_separated = False # Indicates whether the model has been separated for pruning tree.

    # Derived prediction state, rebuilt on demand and never serialized.
    _CACHED_STATE = ("root_model", "subtree_models", "_block_models_cache", "_block_ranges", "_weights_in_blocks", "_tree_arrays")

    @property
    def flat_model(self) -> linear.FlatModel:
        """The flattened model. Pruned prediction may hold its weights as CSR blocks;
        accessing it restores the CSC layout first."""
        if self._model_separated:
            self._merge_model_after_pruning()
        return self._flat_model

    @flat_model.setter
    def flat_model(self, flat_model: linear.FlatModel):
        if getattr(self, "_model_separated", False):
            self._merge_model_after_pruning()
        self._flat_model = flat_model

    def __getstate__(self):
        # The prediction cache can be rebuilt from flat_model. Serializing it
        # as well would unnecessarily store a second copy of the weights.
        # Keep the attribute names of earlier checkpoints.
        flat_model = self.flat_model
        state = {
            "flat_model" if key == "_flat_model" else key: flat_model if key == "_flat_model" else value
            for key, value in self.__dict__.items()
            if key not in self._CACHED_STATE
        }
        state["_model_separated"] = False
        return state

    def __setstate__(self, state):
        # Also discard duplicated caches from older checkpoints.
        for key in self._CACHED_STATE:
            state.pop(key, None)
        state["_flat_model"] = state.pop("flat_model")
        state["_model_separated"] = False
        self.__dict__.update(state)

    def sigmoid_A(self, x: np.ndarray, prob_A: int) -> np.ndarray:
        """
        Calculate log(sigmoid(prob_A * x)), which represents the probability of the positive class in binary classification.

        Args:
            x (np.ndarray): The decision value matrix with dimension number of instances * number of classes.
            prob_A (int):
                The hyperparameter used in the probability estimation function for
                binary classification: sigmoid(prob_A * x).

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        return log_expit(prob_A * x)

    def predict_values(
        self,
        x: sparse.csr_matrix,
        beam_width: int = 10,
        prob_A: int = 3,
    ) -> np.ndarray:
        """Calculate the probability estimates associated with x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search.
            prob_A (int, optional):
                The hyperparameter used in the probability estimation function for
                binary classification: sigmoid(prob_A * decision_value_matrix).

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        kernels = _prediction_kernels() if beam_width >= 1 and x.shape[0] > 0 else None
        if beam_width >= len(self.root.children):
            # Beam_width is sufficiently large; pruning not applied.
            # Calculates decision values for all nodes.
            if kernels is not None and self._model_separated:
                # Reuse the CSR blocks rather than restoring the flattened weights.
                all_preds = self._predict_values_from_blocks(kernels, x)
                if all_preds is not None:
                    scores = self._compiled_beam_search(kernels, self.sigmoid_A(all_preds, prob_A), beam_width)
                    if scores is not None:
                        return scores
            all_preds = linear.predict_values(self.flat_model, x) # number of instances * (number of labels + total number of metalabels)
            if kernels is not None:
                scores = self._compiled_beam_search(kernels, self.sigmoid_A(all_preds, prob_A), beam_width)
                if scores is not None:
                    return scores
        else:
            # Beam_width is small; pruning applied to reduce computation.
            if not self._model_separated:
                self._separate_model_for_pruning_tree()
                self._model_separated = True
            if kernels is not None:
                log_probs = self._prune_tree_and_predict_log_probs(kernels, x, beam_width, prob_A)
                scores = None if log_probs is None else self._compiled_beam_search(kernels, log_probs, beam_width)
                if scores is not None:
                    return scores
            all_preds = self._prune_tree_and_predict_values(x, beam_width, prob_A) # number of instances * (number of labels + total number of metalabels)
        return np.vstack([self._beam_search(all_preds[i], beam_width, prob_A) for i in range(all_preds.shape[0])])

    def _separate_model_for_pruning_tree(self):
        """Build CSR weights of the root and of each root subtree for fast repeated batch prediction.

        Block b is the CSR conversion of a column range of the flattened CSC
        weights. All blocks share one data and one index array, where block b
        starts at ``block_offset[b]``. If the CSC weights are canonical (sorted,
        without duplicates) and the ranges do not overlap, each block overwrites
        its own CSC range, so prediction needs no second copy of the model;
        ``flat_model`` converts the blocks back to the identical CSC arrays when
        accessed. Otherwise the blocks are copies.

        A block usually has entries in few rows (features), so only those rows
        have row pointers, found through a bitmap of nonempty rows and the number
        of nonempty rows before each 64-row word.
        """
        log_memory("prediction: building CSR cache")
        flat_model = self._flat_model
        weights = flat_model.weights
        if not sparse.isspmatrix_csc(weights):
            weights = flat_model.weights = weights.tocsc()
        num_features = weights.shape[0]

        ranges = [(self.node_ptr[self.root.index], self.node_ptr[self.root.index + 1])]
        for i, child in enumerate(self.root.children):
            stop = (
                self.node_ptr[self.root.children[i + 1].index]
                if i + 1 < len(self.root.children) else self.node_ptr[-1]
            )
            ranges.append((self.node_ptr[child.index], stop))
        nnz_ranges = [(int(weights.indptr[start]), int(weights.indptr[stop])) for start, stop in ranges]
        sizes = np.array([hi - lo for lo, hi in nnz_ranges], dtype=np.int64)
        # SciPy caches the check in attributes, which would also be pickled.
        cached_flags = {key: weights.__dict__[key] for key in ("_has_canonical_format", "_has_sorted_indices") if key in weights.__dict__}
        canonical = bool(weights.has_canonical_format)
        for key in ("_has_canonical_format", "_has_sorted_indices"):
            weights.__dict__.pop(key, None)
        weights.__dict__.update(cached_flags)
        in_place = canonical and all(
            prev_hi <= lo for (_, prev_hi), (lo, _) in zip(nnz_ranges, nnz_ranges[1:])
        )
        if in_place:
            data, indices = weights.data, weights.indices
            block_offset = np.array([lo for lo, _ in nnz_ranges], dtype=np.int64)
        else:
            data = np.empty(sizes.sum(), dtype=weights.dtype)
            indices = np.empty(sizes.sum(), dtype=weights.indices.dtype)
            block_offset = np.concatenate(([0], np.cumsum(sizes[:-1]))).astype(np.int64)
        num_words = (num_features + 63) // 64
        self._weights_in_blocks = in_place
        layout = self._block_ranges = {
            "columns": ranges,
            "nnz": nnz_ranges,
            "offset": block_offset,
            "first_column": np.array([start for start, _ in ranges], dtype=np.int64),
            "num_features": num_features,
            "row_bits": np.zeros((len(ranges), num_words), dtype=np.uint64),
            "row_rank": np.zeros((len(ranges), num_words), dtype=np.int64 if num_features > np.iinfo(np.int32).max else np.int32),
            # Row pointers of the nonempty rows of each block, then all of them in one array.
            "row_ptrs": [None] * len(ranges),
            "indices": indices,
            "data": data,
            # Whether each block currently holds CSR arrays in place of its CSC range.
            "is_csr": np.zeros(len(ranges), dtype=np.bool_),
        }

        transpose = _CompressedTranspose(num_features, weights.indptr.dtype, in_place)
        indptr = np.empty(num_features + 1, dtype=weights.indptr.dtype)
        try:
            for b, ((start, stop), (lo, hi)) in enumerate(zip(ranges, nnz_ranges)):
                dest = np.s_[block_offset[b] : block_offset[b] + hi - lo]
                transpose(
                    num_features,
                    weights.indptr[start : stop + 1] - lo,
                    weights.indices[lo:hi],
                    weights.data[lo:hi],
                    indptr,
                    indices[dest],
                    data[dest],
                )
                layout["row_ptrs"][b] = _compress_row_pointers(indptr, layout["row_bits"][b], layout["row_rank"][b])
                layout["is_csr"][b] = in_place
        except BaseException:
            # Restore the CSC ranges already overwritten before giving up.
            self._merge_model_after_pruning()
            raise
        finally:
            del transpose, indptr

        lengths = np.array([row_ptr.size for row_ptr in layout["row_ptrs"]], dtype=np.int64)
        layout["row_ptr_offset"] = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
        layout["row_ptr"] = np.concatenate(layout["row_ptrs"])
        layout["row_ptrs"] = [
            layout["row_ptr"][start : start + length] for start, length in zip(layout["row_ptr_offset"], lengths)
        ]
        log_memory("prediction: CSR cache ready")

    def _block_row_pointers(self, b: int) -> np.ndarray:
        """The full CSR row pointers of block b."""
        layout = self._block_ranges
        num_features = layout["num_features"]
        shifts = np.arange(64, dtype=np.uint64)
        nonempty = ((layout["row_bits"][b][:, np.newaxis] >> shifts) & np.uint64(1)).astype(np.bool_).ravel()[:num_features]
        row_ptr = layout["row_ptrs"][b]
        counts = np.zeros(num_features, dtype=row_ptr.dtype)
        counts[nonempty] = np.diff(row_ptr)
        indptr = np.empty(num_features + 1, dtype=row_ptr.dtype)
        indptr[0] = row_ptr[0]
        np.cumsum(counts, out=indptr[1:])
        indptr[1:] += row_ptr[0]
        return indptr

    def _block_models(self) -> list[linear.FlatModel]:
        """FlatModels of the CSR blocks, for code paths that do not use the kernels."""
        layout = self.__dict__.get("_block_ranges")
        if layout is None:
            raise AttributeError("The model is not separated into blocks.")
        if "_block_models_cache" not in self.__dict__:
            models = []
            for b, ((start, stop), (lo, hi)) in enumerate(zip(layout["columns"], layout["nnz"])):
                dest = np.s_[layout["offset"][b] : layout["offset"][b] + hi - lo]
                # Set the buffers directly: the tuple constructor may downcast
                # int64 indices for small blocks and allocate an unnecessary copy.
                block = sparse.csr_matrix((layout["num_features"], stop - start), dtype=layout["data"].dtype)
                block.data = layout["data"][dest]
                block.indices = layout["indices"][dest]
                block.indptr = self._block_row_pointers(b).astype(block.indices.dtype, copy=False)
                name = "root-flattened-tree" if b == 0 else "subtree-flattened-tree"
                models.append(linear.FlatModel(name, block, self._flat_model.bias, _BLOCK_THRESHOLDS, False))
            self._block_models_cache = models
        return self._block_models_cache

    @property
    def root_model(self) -> linear.FlatModel:
        return self._block_models()[0]

    @property
    def subtree_models(self) -> list[linear.FlatModel]:
        return self._block_models()[1:]

    def _merge_model_after_pruning(self):
        """Release the CSR blocks, first restoring the CSC weights they overwrote."""
        blocks = self.__dict__.get("_block_ranges")
        if blocks is not None and blocks["is_csr"].any():
            weights = self._flat_model.weights
            num_columns = max(stop - start for start, stop in blocks["columns"])
            transpose = _CompressedTranspose(num_columns, weights.indptr.dtype, True)
            for b in np.flatnonzero(blocks["is_csr"]):
                (start, stop), (lo, hi) = blocks["columns"][b], blocks["nnz"][b]
                indptr = np.empty(stop - start + 1, dtype=weights.indptr.dtype)
                transpose(
                    stop - start,
                    self._block_row_pointers(b),
                    weights.indices[lo:hi],
                    weights.data[lo:hi],
                    indptr,
                    weights.indices[lo:hi],
                    weights.data[lo:hi],
                )
                if not np.array_equal(indptr, weights.indptr[start : stop + 1] - lo):
                    raise RuntimeError("CSR blocks do not match the flattened weights.")
                blocks["is_csr"][b] = False
            del transpose
        for key in self._CACHED_STATE:
            if key != "_tree_arrays":
                self.__dict__.pop(key, None)
        self._model_separated = False

    def _prune_tree_and_predict_values(self, x: sparse.csr_matrix, beam_width: int, prob_A: int) -> np.ndarray:
        """Calculates the selective decision values associated with instances x by evaluating only the most relevant subtrees.

        Only subtrees corresponding to the top beam_width candidates from the root are evaluated,
        skipping the rest to avoid unnecessary computation.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int): Number of top candidate branches considered for prediction.
            prob_A (int):
                The hyperparameter used in the probability estimation function for
                binary classification: sigmoid(prob_A * decision_value_matrix).

        Returns:
            np.ndarray: A matrix with dimension number of instances * (number of labels + total number of metalabels).
        """
        # Initialize space for all predictions with negative infinity
        num_instances, num_labels = x.shape[0], self.node_ptr[-1]
        all_preds = np.full((num_instances, num_labels), -np.inf)

        # Calculate root decision values and scores
        root_preds = linear.predict_values(self.root_model, x)
        children_scores = 0.0 + self.sigmoid_A(root_preds, prob_A)

        slice = np.s_[:, self.node_ptr[self.root.index] : self.node_ptr[self.root.index + 1]]
        all_preds[slice] = root_preds

        # Select indices of the top beam_width subtrees for each instance
        top_beam_width_indices = np.argsort(-children_scores, axis=1, kind="stable")[:, :beam_width]

        # Build a mask where mask[i, j] is True if the j-th subtree is among the top beam_width subtrees for the i-th instance
        mask = np.zeros_like(children_scores, dtype=np.bool_)
        np.put_along_axis(mask, top_beam_width_indices, True, axis=1)
        
        # Calculate predictions for each subtree with its corresponding instances
        for subtree_idx in range(len(self.root.children)):
            subtree_model = self.subtree_models[subtree_idx]
            instances_mask = mask[:, subtree_idx]
            if not instances_mask.any():
                continue
            reduced_instances = x[np.s_[instances_mask], :]

            # Locate the position of the subtree root in the weight mapping of all nodes
            subtree_weights_start = self.node_ptr[self.root.children[subtree_idx].index]
            subtree_weights_end = subtree_weights_start + subtree_model.weights.shape[1]

            slice = np.s_[instances_mask, subtree_weights_start:subtree_weights_end]
            all_preds[slice] = linear.predict_values(subtree_model, reduced_instances)

        return all_preds

    def _prune_tree_and_predict_log_probs(self, kernels, x: sparse.csr_matrix, beam_width: int, prob_A: int):
        """``sigmoid_A`` of ``_prune_tree_and_predict_values``, using compiled products.

        Returns None if the inputs are not float64, which the kernels require.
        """
        num_instances, num_columns = x.shape[0], self.node_ptr[-1]
        # FlatModel.predict_values aligns features and appends the bias column per
        # block; all blocks share them, so prepare the instances once.
        x = self._flat_model.prepare_instances(x)
        layout = self._block_ranges
        if x.dtype != np.float64 or layout["data"].dtype != np.float64:
            return None
        values = np.zeros((num_instances, num_columns))

        def evaluate(blocks):
            _blocks_dot(kernels, layout, x, blocks, values)

        log_probs = np.full((num_instances, num_columns), -np.inf)

        # Block 0 is the root.
        evaluate(np.zeros((num_instances, 1), dtype=np.int64))
        root = np.s_[:, self.node_ptr[self.root.index] : self.node_ptr[self.root.index + 1]]
        root_log_probs = self.sigmoid_A(values[root] + _BLOCK_THRESHOLDS, prob_A)
        children_scores = 0.0 + root_log_probs
        log_probs[root] = root_log_probs

        top_beam_width_indices = np.argsort(-children_scores, axis=1, kind="stable")[:, :beam_width]
        mask = np.zeros_like(children_scores, dtype=np.bool_)
        np.put_along_axis(mask, top_beam_width_indices, True, axis=1)

        # Block i + 1 is the subtree of the i-th child of the root.
        evaluate(top_beam_width_indices.astype(np.int64) + 1)
        for subtree_idx, child in enumerate(self.root.children):
            rows = np.flatnonzero(mask[:, subtree_idx])
            if rows.size == 0:
                continue
            start, stop = layout["columns"][subtree_idx + 1]
            block = np.s_[rows, start:stop]
            log_probs[block] = self.sigmoid_A(values[block] + _BLOCK_THRESHOLDS, prob_A)
        return log_probs

    def _predict_values_from_blocks(self, kernels, x: sparse.csr_matrix):
        """``linear.predict_values(self.flat_model, x)`` computed from the CSR blocks.

        The flattened CSC product sums each value over features in increasing
        order, while the blocks follow the stored order of x. Both agree when x
        and the weights are canonical (sorted, without duplicates), and the blocks
        cover every column; otherwise this returns None.
        """
        layout = self._block_ranges
        bounds = [start for start, _ in layout["columns"]] + [layout["columns"][-1][1]]
        if (
            not self._weights_in_blocks
            or bounds[0] != 0
            or bounds[-1] != self.node_ptr[-1]
            or any(stop != start for (_, stop), (start, _) in zip(layout["columns"], layout["columns"][1:]))
        ):
            return None
        x = self._flat_model.prepare_instances(x)
        if x.dtype != np.float64 or layout["data"].dtype != np.float64 or not x.has_canonical_format:
            return None
        values = np.zeros((x.shape[0], self.node_ptr[-1]))
        blocks = np.broadcast_to(np.arange(len(layout["columns"]), dtype=np.int64), (x.shape[0], len(layout["columns"])))
        _blocks_dot(kernels, layout, x, np.ascontiguousarray(blocks), values)
        return values + self._flat_model.thresholds

    def _compiled_beam_search(self, kernels, log_probs: np.ndarray, beam_width: int):
        """``_beam_search`` for every row of ``sigmoid_A(all_preds)`` at once.

        Returns None, so that the caller falls back to ``_beam_search``, if the tree
        cannot be described by index arrays or if a NaN makes sorting ill-defined.
        """
        arrays = self._prediction_tree_arrays()
        if arrays is None or np.isnan(log_probs).any():
            return None
        node_ptr, is_leaf, child_ptr, child_idx, label_ptr, label_idx = arrays
        # A beam never holds more nodes than the tree has.
        beam_width = min(beam_width, len(is_leaf))
        num_instances = log_probs.shape[0]
        beam_nodes = np.empty((num_instances, beam_width), dtype=np.int64)
        beam_scores = np.empty((num_instances, beam_width), dtype=np.float64)
        beam_size = np.empty(num_instances, dtype=np.int64)
        kernels.run(
            kernels.beam_search,
            log_probs,
            beam_width,
            self.root.index,
            node_ptr,
            is_leaf,
            child_ptr,
            child_idx,
            beam_nodes,
            beam_scores,
            beam_size,
        )
        rows, labels, log_scores = kernels.run(
            kernels.beam_leaf_scores, log_probs, beam_nodes, beam_scores, beam_size, node_ptr, label_ptr, label_idx
        )
        scores = np.zeros((num_instances, len(self.root.label_map)))
        scores[rows, labels] = np.exp(log_scores)
        return scores

    def _prediction_tree_arrays(self):
        """Children and labels of every node, indexed by node.index, for the kernels."""
        if "_tree_arrays" not in self.__dict__:
            nodes = [None] * (len(self.node_ptr) - 1)
            valid = True

            def collect(node):
                nonlocal valid
                index = getattr(node, "index", None)
                if index is None or not 0 <= index < len(nodes) or nodes[index] is not None:
                    valid = False
                else:
                    nodes[index] = node

            self.root.dfs(collect)
            arrays = None
            if valid and all(node is not None for node in nodes):
                node_ptr = np.asarray(self.node_ptr, dtype=np.int64)
                is_leaf = np.array([node.isLeaf() for node in nodes], dtype=np.bool_)
                children = [[child.index for child in node.children] for node in nodes]
                labels = [np.asarray(node.label_map, dtype=np.int64) if node.isLeaf() else np.empty(0, np.int64) for node in nodes]
                widths = np.diff(node_ptr)
                num_labels = len(self.root.label_map)
                if all(
                    widths[i] == (len(labels[i]) if is_leaf[i] else len(children[i]))
                    and (labels[i].size == 0 or (labels[i].min() >= 0 and labels[i].max() < num_labels))
                    for i in range(len(nodes))
                ):
                    child_ptr = np.zeros(len(nodes) + 1, dtype=np.int64)
                    child_ptr[1:] = np.cumsum([len(c) for c in children])
                    child_idx = np.fromiter((c for cs in children for c in cs), dtype=np.int64, count=child_ptr[-1])
                    label_ptr = np.zeros(len(nodes) + 1, dtype=np.int64)
                    label_ptr[1:] = np.cumsum([len(l) for l in labels])
                    label_idx = np.concatenate(labels) if labels else np.empty(0, np.int64)
                    arrays = (node_ptr, is_leaf, child_ptr, child_idx, label_ptr, label_idx)
            self._tree_arrays = arrays
        return self._tree_arrays

    def _beam_search(self, instance_preds: np.ndarray, beam_width: int, prob_A: int) -> np.ndarray:
        """Predict with beam search using cached probability estimates for a single instance.

        Args:
            instance_preds (np.ndarray): A vector of cached probability estimates of each node, has dimension number of labels + total number of metalabels.
            beam_width (int): Number of candidates considered.
            prob_A (int, optional):
                The hyperparameter used in the probability estimation function for
                binary classification: sigmoid(prob_A * decision_value_matrix).

        Returns:
            np.ndarray: A vector with dimension number of classes.
        """
        cur_level = [(self.root, 0.0)]  # pairs of (node, score)
        next_level = []
        while True:
            num_internal = sum(map(lambda pair: not pair[0].isLeaf(), cur_level))
            if num_internal == 0:
                break

            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.node_ptr[node.index] : self.node_ptr[node.index + 1]]
                pred = instance_preds[slice]
                children_score = score + self.sigmoid_A(pred, prob_A)
                next_level.extend(zip(node.children, children_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -pair[1])[:beam_width]
            next_level = []

        num_labels = len(self.root.label_map)
        scores = np.zeros(num_labels)
        for node, score in cur_level:
            slice = np.s_[self.node_ptr[node.index] : self.node_ptr[node.index + 1]]
            pred = instance_preds[slice]
            scores[node.label_map] = np.exp(score + self.sigmoid_A(pred, prob_A))
        return scores


def train_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=DEFAULT_K,
    dmax=DEFAULT_DMAX,
    verbose: bool = True,
    root: Node = None,
) -> TreeModel:
    """Train a linear model for multi-label data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Node weights are staged in a temporary file during training to avoid retaining
    a second copy of the model while assembling the final CSC weight matrix.
    The temporary directory (configurable with TMPDIR) needs space for the sparse
    node weights. The returned model is held in memory and does not depend on this file.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.
        root (Node, optional): Pre-built tree root. Defaults to None.

    Returns:
        TreeModel: A model which can be used in predict_values.
    """
    log_memory("tree: start")
    if root is None:
        # CSR operands produce CSR directly: avoid a full CSC-to-CSR copy of
        # the large label-by-feature matrix, then normalize it in place.
        label_representation = y.T.tocsr() @ x
        label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1, copy=False)
        log_memory("tree: label representation ready")
        # Hand over the only reference, so that clustering can release it.
        owned_representation = [label_representation]
        del label_representation
        root = _build_tree(owned_representation, np.arange(y.shape[1]), 0, K, dmax)
        root.is_root = True
    log_memory("tree: built")

    num_nodes = _count_node_features(root, y, x)
    log_memory("tree: feature counts ready")

    model_size = get_estimated_model_size(root)
    print(f"The estimated tree model size is: {model_size / (1024**3):.3f} GB")

    # Calculate the total memory (excluding swap) on the local machine
    total_memory = psutil.virtual_memory().total
    print(f"Your system memory is: {total_memory / (1024**3):.3f} GB")

    if total_memory <= model_size:
        raise MemoryError(f"Not enough memory to train the model.")

    pbar = tqdm(total=num_nodes, disable=not verbose)
    trained_nodes = 0

    def on_trained(node):
        nonlocal trained_nodes
        pbar.update()
        trained_nodes += 1
        if node is root or trained_nodes % 100 == 0:
            log_memory(f"training: node {trained_nodes}/{num_nodes}")

    trained = _train_nodes(root, y, x, options, on_trained)
    try:
        flat_model, node_ptr = _flatten_model(root, trained_nodes=trained)
    finally:
        trained.close()  # stops and joins the training threads if assembly failed
        pbar.close()
    return TreeModel(root, flat_model, node_ptr)


def _build_tree(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Build the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
            It may be wrapped in a one-element list that holds its only reference. Then the list is emptied and
            a large matrix is kept on disk while it is clustered, which only needs the clustering's own copy.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: Root of the (sub)tree built from label_representation.
    """
    owned = isinstance(label_representation, list)
    if owned:
        label_representation = label_representation.pop()
    children = []
    if d < dmax and label_representation.shape[0] > K:
        if label_representation.shape[0] > 10000:
            kmeans_algo = ElkanKmeans
        else:
            kmeans_algo = LloydKmeans

        kmeans = kmeans_algo(
            n_clusters=K,
            max_iter=300,
            tol=0.0001,
            random_state=np.random.randint(2**31 - 1),
            verbose=True,
            n_threads=min(8, psutil.cpu_count(logical=False) or 1),
        )
        if owned and label_representation.nnz >= _SPILL_MIN_NNZ:
            # fit converts its input with the same call; give it the converted matrix.
            points = gb.io.from_scipy_sparse(label_representation)
            spilled = _SpilledCSR(label_representation)
            del label_representation
            # fit retains centroids, but tree construction only needs the labels.
            # Release them before reading the representation back.
            try:
                metalabels = kmeans.fit(points)
            finally:
                del points, kmeans
                gc.collect()
                label_representation = spilled.load()
        else:
            metalabels = kmeans.fit(label_representation)
            # fit retains centroids, but tree construction only needs the labels.
            # Release them before descending into another clustering problem.
            del kmeans

        unique_labels = np.unique(metalabels)
        if len(unique_labels) == K:
            create_child_node = lambda i: _build_tree(
                [label_representation[metalabels == i]], label_map[metalabels == i], d + 1, K, dmax
            )
        else:
            create_child_node = lambda i: Node(label_map=label_map[metalabels == i], children=[])

        for i in range(K):
            child = create_child_node(i)
            children.append(child)

    return Node(label_map=label_map, children=children)


class _SpilledCSR:
    """A CSR matrix kept in a temporary file, to be read back with the same arrays and flags."""

    _FLAGS = ("_has_sorted_indices", "_has_canonical_format")

    def __init__(self, matrix: sparse.csr_matrix):
        self.type, self.shape, self.dtype = type(matrix), matrix.shape, matrix.dtype
        arrays = (matrix.data, matrix.indices, matrix.indptr)
        self.layout = [(array.dtype, array.size) for array in arrays]
        # SciPy caches whether indices are sorted; graphblas.io trusts the cached value.
        self.flags = {key: matrix.__dict__[key] for key in self._FLAGS if key in matrix.__dict__}
        self.file = tempfile.TemporaryFile(prefix="libmultilabel-representation-")
        try:
            for array in arrays:
                self.file.write(np.ascontiguousarray(array).data.cast("B"))
        except BaseException:
            self.file.close()
            raise

    def load(self) -> sparse.csr_matrix:
        with self.file:
            self.file.seek(0)
            arrays = []
            for dtype, size in self.layout:
                array = np.empty(size, dtype=dtype)
                _read_staged(self.file, array, dtype)
                arrays.append(array)
        matrix = self.type(self.shape, dtype=self.dtype)
        matrix.data, matrix.indices, matrix.indptr = arrays
        matrix.__dict__.update(self.flags)
        return matrix


def _count_node_features(root: Node, y: sparse.csr_matrix, x: sparse.csr_matrix) -> int:
    """Count feature unions without floating-point counts or whole-node slices.

    Labels are binary indicators. Boolean multiplication tracks the existence
    of a feature for each label, including signed/cancelling input features.
    """
    features = (y.T.tocsr().astype(bool) @ x.astype(bool)).tocsr()
    features.eliminate_zeros()
    used = np.zeros(x.shape[1], dtype=bool)
    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1
        used.fill(False)
        for label in node.label_map:
            used[features.indices[features.indptr[label] : features.indptr[label + 1]]] = True
        node.num_features_used = np.count_nonzero(used)

    root.dfs(count)
    return num_nodes


def get_estimated_model_size(root):
    total_num_weights = 0

    def collect_stat(node: Node):
        nonlocal total_num_weights

        if node.isLeaf():
            total_num_weights += len(node.label_map) * node.num_features_used
        else:
            total_num_weights += len(node.children) * node.num_features_used

    root.dfs(collect_stat)

    # 16 is because when storing sparse matrices, indices (int64) require 8 bytes and floats require 8 bytes
    # Our study showed that among the used features of every binary classification problem, on average no more than 2/3 of weights obtained by the dual coordinate descent method are non-zeros.
    return total_num_weights * 16 * 2 / 3


def _rows_with_entries(columns: sparse.csc_matrix, labels: np.ndarray) -> np.ndarray:
    """Equals ``columns[:, labels].getnnz(axis=1) > 0`` for a CSC matrix, in
    time proportional to the entries of those columns rather than of all rows."""
    labels = np.asarray(labels, dtype=np.int64)
    starts = columns.indptr[labels].astype(np.int64)
    lengths = columns.indptr[labels + 1] - starts
    total = int(lengths.sum())
    rows = np.zeros(columns.shape[0], dtype=np.bool_)
    if total:
        offsets = np.repeat(starts + lengths - np.cumsum(lengths), lengths) + np.arange(total)
        rows[columns.indices[offsets]] = True
    return rows


def _train_node(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node):
    """If node is internal, compute the metalabels representing each child and train
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    node_y, node_x, projection = _node_training_data(y, x, node)
    model = linear.train_1vsrest(node_y, node_x, False, options, False, sparse_output=True)
    _set_node_model(node, model, projection)


def _node_training_data(y: sparse.csr_matrix, x: sparse.csr_matrix, node: Node):
    """The labels and features that train a node, and the projection back to all features.

    Returns:
        tuple: The 0/1 labels of the node's classifiers, x restricted to the
        features of its instances, and the ``(feature_map, num_features)`` that
        ``_set_node_model`` needs (``feature_map`` is None if nothing was removed).
    """
    # LIBLINEAR allocates dense working vectors even for sparse inputs. A node
    # only needs columns occurring in its instances; absent feature weights are
    # exactly zero. Restore the original feature coordinates in sparse output.
    num_features = x.shape[1]
    used = np.zeros(num_features, dtype=bool)
    used[x.indices] = True
    feature_map = np.flatnonzero(used)
    del used
    if 0 < feature_map.size < num_features:
        # Equals x[:, feature_map]: every entry is kept, in order, and its column
        # renumbered. The data is copied too, as LIBLINEAR sorts indices in place.
        new_columns = np.empty(num_features, dtype=x.indices.dtype)
        new_columns[feature_map] = np.arange(feature_map.size, dtype=x.indices.dtype)
        x = sparse.csr_matrix(
            (x.data.copy(), new_columns[x.indices], x.indptr.copy()), shape=(x.shape[0], feature_map.size)
        )
        del new_columns
    else:
        feature_map = None

    if node.isLeaf():
        node_y = y[:, node.label_map]
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        y_columns = y.tocsc()
        meta_y = np.empty((y.shape[0], len(node.children)), dtype=np.bool_)
        for j, child in enumerate(node.children):
            meta_y[:, j] = _rows_with_entries(y_columns, child.label_map)
        del y_columns
        node_y = sparse.csr_matrix(meta_y)
    return node_y, x, (feature_map, num_features)


def _set_node_model(node: Node, model: linear.FlatModel, projection):
    """Store a node model with weights for all features, given by ``_node_training_data``."""
    feature_map, num_features = projection
    node.model = model
    weights = sparse.csc_matrix(node.model.weights)
    if feature_map is not None:
        extra_features = weights.shape[0] - feature_map.size  # optional bias column
        if extra_features:
            feature_map = np.append(feature_map, np.arange(num_features, num_features + extra_features))
        weights = sparse.csc_matrix(
            (weights.data, feature_map[weights.indices], weights.indptr),
            shape=(num_features + extra_features, weights.shape[1]),
            copy=False,
        )
    node.model.weights = weights


class _NodeProblem:
    """The one-vs-rest problem of a node, as train_1vsrest sets it up, trained one label at a time."""

    def __init__(self, node: Node, y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, projection):
        self.node = node
        self.projection = projection
        x, options, self.bias = linear._prepare_options(x, options)
        self.num_features = x.shape[1]
        self.y = y.tocsc()
        self.prob = linear.problem(np.ones((self.y.shape[0],)), x)
        self.param = linear.parameter(re.sub(r"-m\s+\d+", "", options))
        if self.param.solver_type in [linear.solver_names.L2R_L1LOSS_SVC_DUAL, linear.solver_names.L2R_L2LOSS_SVC_DUAL]:
            self.param.w_recalc = True  # only works for solving L1/L2-SVM dual
        # The problem copies x; this is what the node keeps while it waits for threads.
        self.nbytes = self.prob.x_space.nbytes + self.prob.rowptr.nbytes + self.y.data.nbytes + self.y.indices.nbytes
        # Rows of the trained weights in the coordinates of all features, as
        # _set_node_model maps them: kept features, then any bias row.
        feature_map, num_features = projection
        if feature_map is None:
            self.row_map = None
            self.num_rows = self.num_features
        else:
            extra_features = self.num_features - feature_map.size
            self.row_map = np.concatenate((feature_map, np.arange(num_features, num_features + extra_features)))
            self.num_rows = num_features + extra_features
        self.weights = [None] * self.y.shape[1]
        self.remaining = len(self.weights)
        self.error = None
        self.cancelled = False
        self.lock = threading.Lock()
        self.done = threading.Event()
        if self.remaining == 0:
            self.done.set()

    def train(self, label_idx: int):
        # As ParallelOVRTrainer.run: stop starting labels after an error.
        if self.error is None and not self.cancelled:
            try:
                weights = np.asarray(linear._train_label(self.prob, self.param, linear._signed_label(self.y, label_idx)))
                # The entries of sparse.csc_matrix(weights), already mapped to all features.
                rows = np.flatnonzero(weights)
                self.weights[label_idx] = (rows if self.row_map is None else self.row_map[rows], weights.ravel()[rows])
                del weights, rows
            except Exception as exc:
                with self.lock:
                    if self.error is None:
                        self.error = exc
        with self.lock:
            self.remaining -= 1
            if self.remaining == 0:
                self.done.set()

    def set_node_model(self):
        """Set the node model, as train_1vsrest and _set_node_model would."""
        # The problem is no longer needed once every label is trained.
        self.prob = self.y = None
        if self.error is not None:
            raise self.error
        nnz = sum(rows.size for rows, _ in self.weights)
        index_dtype = np.int32 if max(self.num_rows, nnz) <= np.iinfo(np.int32).max else np.int64
        indptr = np.zeros(len(self.weights) + 1, dtype=index_dtype)
        np.cumsum([rows.size for rows, _ in self.weights], out=indptr[1:])
        indices = np.empty(nnz, dtype=index_dtype)
        data = np.empty(nnz, dtype=np.float64)
        for (rows, values), start, stop in zip(self.weights, indptr[:-1], indptr[1:]):
            indices[start:stop] = rows
            data[start:stop] = values
        self.weights = None
        # Canonical by construction: each column lists increasing rows once.
        weights = sparse.csc_matrix((self.num_rows, len(indptr) - 1), dtype=np.float64)
        weights.data, weights.indices, weights.indptr = data, indices, indptr
        self.node.model = linear.FlatModel(name="1vsrest", weights=weights, bias=self.bias, thresholds=0, multiclass=False)


class _LabelTrainingPool:
    """Threads that train the labels of queued node problems, first in, first out.

    ParallelOVRTrainer trains one node at a time, so threads wait while the last
    labels of a node finish and while the next node is prepared, and a node
    with few labels uses few threads. Here the labels of later nodes start as
    soon as threads are free. With one thread, LIBLINEAR is called in the same
    order as before and returns the same weights.
    """

    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.tasks = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.queued = 0
        self.threads = [threading.Thread(target=self._work, daemon=True) for _ in range(num_threads)]
        for thread in self.threads:
            thread.start()

    def submit(self, problem: _NodeProblem):
        with self.lock:
            self.queued += len(problem.weights)
        for label_idx in range(len(problem.weights)):
            self.tasks.put((problem, label_idx))

    def _work(self):
        while True:
            task = self.tasks.get()
            if task is None:
                return
            with self.lock:
                self.queued -= 1
            task[0].train(task[1])
            del task  # do not keep the problem alive while waiting

    def close(self):
        for _ in self.threads:
            self.tasks.put(None)
        for thread in self.threads:
            thread.join()


def _train_nodes(
    root: Node, y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, on_trained: Callable[[Node], None]
):
    """Train every node and yield it, in depth-first order.

    Later nodes are prepared and queued while earlier ones train, as long as
    the threads have little queued work and the queued problems are not larger
    than the root problem, which is the largest one since every node trains on
    a subset of its instances.
    """
    if "-m" not in (options or "").split():
        options = f"{options or ''} -m {min(8, psutil.cpu_count(logical=False) or 1)}"
    options_split = options.split()
    num_threads = int(options_split[options_split.index("-m") + 1])
    if num_threads < 1:
        raise ValueError("-m must specify at least one training thread.")

    y_columns = y.tocsc()  # to find the instances of each node without slicing all rows of y
    nodes = []
    root.dfs(nodes.append)

    def prepare(node):
        if node.is_root:
            node_y, node_x, projection = _node_training_data(y, x, node)
        else:
            relevant_instances = _rows_with_entries(y_columns, node.label_map)
            node_y, node_x, projection = _node_training_data(y[relevant_instances], x[relevant_instances], node)
        return _NodeProblem(node, node_y, node_x, options, projection)

    def finish(problem):
        problem.done.wait()
        problem.set_node_model()
        on_trained(problem.node)
        return problem.node

    pool = _LabelTrainingPool(num_threads)
    pending = collections.deque()
    pending_bytes = 0
    max_pending_bytes = None
    try:
        for node in nodes:
            while pending:
                if pending[0].done.is_set():
                    problem = pending.popleft()
                    pending_bytes -= problem.nbytes
                    yield finish(problem)
                elif pool.queued >= 2 * num_threads or pending_bytes >= max_pending_bytes:
                    pending[0].done.wait(0.005)
                else:
                    break
            problem = prepare(node)
            if max_pending_bytes is None:
                max_pending_bytes = max(problem.nbytes, 256 * 1024**2)
            pool.submit(problem)
            pending.append(problem)
            pending_bytes += problem.nbytes
        while pending:
            yield finish(pending.popleft())
    finally:
        for problem in pending:
            problem.cancelled = True
        pool.close()


def _read_staged(file, out: np.ndarray, dtype):
    """Read ``out.size`` raw values of dtype into out, converting in bounded chunks if needed."""
    if out.dtype == dtype:
        buffer = out.data.cast("B")
        while buffer:
            read = file.readinto(buffer)
            if not read:
                raise EOFError("Staged weights are truncated.")
            buffer = buffer[read:]
        return
    chunk = np.empty(min(out.size, _WEIGHT_CHUNK_SIZE), dtype=dtype)
    for start in range(0, out.size, chunk.size):
        part = chunk[: min(chunk.size, out.size - start)]
        _read_staged(file, part, dtype)
        out[start : start + part.size] = part


def _flatten_model(
    root: Node,
    train_node: Callable[[Node], None] | None = None,
    trained_nodes: Iterable[Node] | None = None,
) -> tuple[linear.FlatModel, np.ndarray]:
    """Flatten tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.
    Consecutive values of the returned array denote the start and end indices of each node in the tree.
    To extract a node's classifiers:
        slice = np.s_[node_ptr[node.index]:
                      node_ptr[node.index+1]]
        node.model.weights == flat_model.weights[:, slice]

    Args:
        root (Node): Root of the tree.
        train_node (Callable, optional): Train each node immediately before staging
            its weights. If omitted, all nodes must already have trained models.
        trained_nodes (Iterable, optional): The nodes of root in depth-first order,
            each yielded once trained. Overrides train_node.

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the ranges of each node.
    """
    node_ptr = [0]
    node_nnz = []
    node_dtypes = []
    bias = None
    num_features = None
    data_dtype = None

    # Staging before allocation avoids keeping all node weights and the flattened
    # matrix in RAM together. A single file also avoids one open file per node.
    with tempfile.TemporaryFile(prefix="libmultilabel-weights-") as weights_file:

        def visit(node):
            nonlocal bias, num_features, data_dtype
            if train_node is not None:
                train_node(node)
            weights = sparse.csc_matrix(node.model.weights, copy=False)
            if node is root:
                bias = node.model.bias
                num_features = weights.shape[0]
                data_dtype = weights.dtype
            assert bias == node.model.bias
            if weights.shape[0] != num_features:
                raise ValueError("Node weight matrices must have the same number of features.")
            data_dtype = np.result_type(data_dtype, weights.dtype)
            node.index = len(node_nnz)
            node_ptr.append(node_ptr[-1] + weights.shape[1])
            node_nnz.append(weights.nnz)

            arrays = (weights.data[: weights.nnz], weights.indices[: weights.nnz], weights.indptr[:-1])
            for array in arrays:
                weights_file.write(np.ascontiguousarray(array).data.cast("B"))
            node_dtypes.append(tuple(array.dtype for array in arrays))
            del node.model.weights

        if trained_nodes is None:
            root.dfs(visit)
        else:
            train_node = None
            expected = []
            root.dfs(expected.append)
            count = 0
            for node in trained_nodes:
                if count >= len(expected) or node is not expected[count]:
                    raise ValueError("Nodes must be trained in depth-first order.")
                visit(node)
                count += 1
            if count != len(expected):
                raise ValueError("Not every node was trained.")
        log_memory("assembly: nodes staged")

        node_ptr = np.asarray(node_ptr, dtype=np.int64)
        total_nnz = sum(node_nnz)
        num_classifiers = int(node_ptr[-1])
        # Both indices and indptr must use int64 once any dimension or the
        # cumulative NNZ exceeds int32, even if every node individually fits.
        index_dtype = np.int64 if max(num_features, num_classifiers, total_nnz) > np.iinfo(np.int32).max else np.int32
        data = np.empty(total_nnz, dtype=data_dtype)
        indices = np.empty(total_nnz, dtype=index_dtype)
        indptr = np.empty(num_classifiers + 1, dtype=index_dtype)

        weights_file.seek(0)
        offset = 0
        for i, nnz in enumerate(node_nnz):
            end = offset + nnz
            columns = slice(node_ptr[i], node_ptr[i + 1])
            for array, dtype in zip((data[offset:end], indices[offset:end], indptr[columns]), node_dtypes[i]):
                _read_staged(weights_file, array, dtype)
            # Offset in the destination dtype to avoid overflowing int32 node pointers.
            indptr[columns] += offset
            offset = end
        indptr[-1] = total_nnz

    # Matching index dtypes let SciPy reuse these buffers without a full-model copy.
    weights = sparse.csc_matrix((data, indices, indptr), shape=(num_features, num_classifiers), copy=False)

    model = linear.FlatModel(
        name="flattened-tree",
        weights=weights,
        bias=bias,
        thresholds=0,
        multiclass=False,
    )

    log_memory("assembly: flattened weights ready")
    return model, node_ptr


class EnsembleTreeModel:
    """An ensemble of tree models.
    The ensemble aggregates predictions from multiple trees to improve accuracy and robustness.
    """

    def __init__(self, tree_models: list[TreeModel]):
        """
        Args:
            tree_models (list[TreeModel]): A list of trained tree models.
        """
        self.name = "ensemble-tree"
        self.tree_models = tree_models
        self.multiclass = False

    def predict_values(self, x: sparse.csr_matrix, beam_width: int = 10) -> np.ndarray:
        """Calculates the averaged probability estimates from all trees in the ensemble.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search for each tree. Defaults to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes, containing averaged scores.
        """
        all_predictions = [model.predict_values(x, beam_width) for model in self.tree_models]
        return np.mean(all_predictions, axis=0)


def train_ensemble_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K: int = DEFAULT_K,
    dmax: int = DEFAULT_DMAX,
    n_trees: int = 3,
    verbose: bool = True,
    seed: int = None,
) -> EnsembleTreeModel:
    """Trains an ensemble of tree models (Parabel/Bonsai-style).
    
    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str, optional): The option string passed to liblinear. Defaults to ''.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        n_trees (int, optional): Number of trees in the ensemble. Defaults to 3.
        verbose (bool, optional): Output extra progress information. Defaults to True.
        seed (int, optional): The base random seed for the ensemble. Defaults to None, which will use 42.

    Returns:
        EnsembleTreeModel: An ensemble model which can be used for prediction.
    """
    if seed is None:
        seed = 42
        
    tree_models = []
    for i in range(n_trees):
        np.random.seed(seed + i)

        tree_model = train_tree(y, x, options, K, dmax, verbose)
        tree_models.append(tree_model)

    print("Ensemble training completed.")

    return EnsembleTreeModel(tree_models)
