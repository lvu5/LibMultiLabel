from __future__ import annotations

import tempfile
from typing import Callable

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

    def __getstate__(self):
        # The prediction cache can be rebuilt from flat_model. Serializing it
        # as well would unnecessarily store a second copy of the weights.
        state = self.__dict__.copy()
        state.pop("root_model", None)
        state.pop("subtree_models", None)
        state["_model_separated"] = False
        return state

    def __setstate__(self, state):
        # Also discard duplicated caches from older checkpoints.
        state.pop("root_model", None)
        state.pop("subtree_models", None)
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
        if beam_width >= len(self.root.children):
            # Beam_width is sufficiently large; pruning not applied.
            # Calculates decision values for all nodes.
            all_preds = linear.predict_values(self.flat_model, x) # number of instances * (number of labels + total number of metalabels)
        else:
            # Beam_width is small; pruning applied to reduce computation.
            if not self._model_separated:
                self._separate_model_for_pruning_tree()
                self._model_separated = True
            all_preds = self._prune_tree_and_predict_values(x, beam_width, prob_A) # number of instances * (number of labels + total number of metalabels)
        return np.vstack([self._beam_search(all_preds[i], beam_width, prob_A) for i in range(all_preds.shape[0])])

    def _separate_model_for_pruning_tree(self):
        """Build CSR subtree weights once for fast repeated batch prediction.

        Construct each block from a CSC view to avoid a temporary column-slice
        copy, then cache its CSR conversion. Keep the flattened CSC weights for
        unpruned prediction and checkpoints; omit the derived cache when saving.
        """
        log_memory("prediction: building CSR cache")
        weights = self.flat_model.weights
        if not sparse.isspmatrix_csc(weights):
            weights = self.flat_model.weights = weights.tocsc()

        def subtree_model(start, stop, name):
            lo, hi = weights.indptr[start], weights.indptr[stop]
            # The temporary CSC block shares data and indices with flat_model;
            # only the final CSR cache owns another copy of those buffers.
            # Set the buffers directly: the tuple constructor may downcast
            # int64 indices for small blocks and allocate an unnecessary copy.
            block = sparse.csc_matrix((weights.shape[0], stop - start), dtype=weights.dtype)
            block.data = weights.data[lo:hi]
            block.indices = weights.indices[lo:hi]
            block.indptr = weights.indptr[start : stop + 1] - lo
            return linear.FlatModel(name, block.tocsr(), self.flat_model.bias, 0, False)

        self.root_model = subtree_model(
            self.node_ptr[self.root.index], self.node_ptr[self.root.index + 1], "root-flattened-tree"
        )
        self.subtree_models = []
        for i, child in enumerate(self.root.children):
            start = self.node_ptr[child.index]
            stop = (
                self.node_ptr[self.root.children[i + 1].index]
                if i + 1 < len(self.root.children) else self.node_ptr[-1]
            )
            self.subtree_models.append(subtree_model(start, stop, "subtree-flattened-tree"))
        log_memory("prediction: CSR cache ready")

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
        root = _build_tree(label_representation, np.arange(y.shape[1]), 0, K, dmax)
        root.is_root = True
        del label_representation
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

    def visit(node):
        nonlocal trained_nodes
        if node.is_root:
            _train_node(y, x, options, node)
        else:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
            _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()
        trained_nodes += 1
        if node is root or trained_nodes % 100 == 0:
            log_memory(f"training: node {trained_nodes}/{num_nodes}")

    try:
        flat_model, node_ptr = _flatten_model(root, train_node=visit)
    finally:
        pbar.close()
    return TreeModel(root, flat_model, node_ptr)


def _build_tree(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Build the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: Root of the (sub)tree built from label_representation.
    """
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
        metalabels = kmeans.fit(label_representation)
        # fit retains centroids, but tree construction only needs the labels.
        # Release them before descending into another clustering problem.
        del kmeans

        unique_labels = np.unique(metalabels)
        if len(unique_labels) == K:
            create_child_node = lambda i: _build_tree(
                label_representation[metalabels == i], label_map[metalabels == i], d + 1, K, dmax
            )
        else:
            create_child_node = lambda i: Node(label_map=label_map[metalabels == i], children=[])

        for i in range(K):
            child = create_child_node(i)
            children.append(child)

    return Node(label_map=label_map, children=children)


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


def _train_node(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node):
    """If node is internal, compute the metalabels representing each child and train
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    # LIBLINEAR allocates dense working vectors even for sparse inputs. A node
    # only needs columns occurring in its instances; absent feature weights are
    # exactly zero. Restore the original feature coordinates in sparse output.
    num_features = x.shape[1]
    used = np.zeros(num_features, dtype=bool)
    used[x.indices] = True
    feature_map = np.flatnonzero(used)
    del used
    reduced = 0 < feature_map.size < num_features
    if reduced:
        x = x[:, feature_map]

    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False, sparse_output=True)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False, sparse_output=True)

    weights = sparse.csc_matrix(node.model.weights)
    if reduced:
        extra_features = weights.shape[0] - feature_map.size  # optional bias column
        if extra_features:
            feature_map = np.append(feature_map, np.arange(num_features, num_features + extra_features))
        weights = sparse.csc_matrix(
            (weights.data, feature_map[weights.indices], weights.indptr),
            shape=(num_features + extra_features, weights.shape[1]),
            copy=False,
        )
    node.model.weights = weights


def _flatten_model(root: Node, train_node: Callable[[Node], None] | None = None) -> tuple[linear.FlatModel, np.ndarray]:
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

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the ranges of each node.
    """
    node_ptr = [0]
    node_nnz = []
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

            for array in (weights.data[: weights.nnz], weights.indices[: weights.nnz], weights.indptr[:-1]):
                for start in range(0, array.size, _WEIGHT_CHUNK_SIZE):
                    np.save(weights_file, array[start : start + _WEIGHT_CHUNK_SIZE], allow_pickle=False)
            del node.model.weights

        root.dfs(visit)
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
            for array in (data[offset:end], indices[offset:end], indptr[columns]):
                for start in range(0, array.size, _WEIGHT_CHUNK_SIZE):
                    array[start : start + _WEIGHT_CHUNK_SIZE] = np.load(weights_file, allow_pickle=False)
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
