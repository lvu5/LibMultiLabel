"""Compiled loops for tree prediction.

The kernels perform the same floating-point operations, in the same order, as
the NumPy/SciPy code they replace, so they return bitwise-identical results.
Parallel loops only split independent outputs (rows or instances) across threads.

* ``compressed_transpose`` is SciPy's ``csr_tocsc`` (also used by ``tocsr``): a
  stable counting sort. Splitting the input into consecutive chunks keeps the
  order, since each output row lists earlier chunks first.
* ``blocks_dot`` accumulates ``x[i] @ W`` over the stored entries of ``x[i]`` in
  order, as SciPy's ``csr_matmat`` does, but writes into a dense block. This
  skips SciPy's symbolic pass and the sparse result that is immediately densified.
* ``beam_search`` keeps, at every level, the first ``beam_width`` candidates of
  a stable sort by decreasing score, which is what
  ``sorted(candidates, key=lambda pair: -pair[1])[:beam_width]`` selects.

Probability estimates (``log_expit`` and ``exp``) stay in SciPy and NumPy.
Unsigned loop indices avoid Numba's negative-index wraparound checks.
"""

import numba
import numpy as np
import psutil
from numba import njit, prange


def num_threads() -> int:
    """Threads for parallel kernels: at most 8 physical cores, as for training by default."""
    return max(1, min(numba.config.NUMBA_NUM_THREADS, 8, psutil.cpu_count(logical=False) or 1))


def run(kernel, *args):
    """Call a parallel kernel with num_threads() threads."""
    previous = numba.get_num_threads()
    numba.set_num_threads(num_threads())
    try:
        return kernel(*args)
    finally:
        numba.set_num_threads(previous)


@njit(cache=True, nogil=True, parallel=True)
def compressed_transpose(num_out, in_ptr, in_idx, in_val, out_ptr, out_idx, out_val, chunks, counts):
    """Transpose a compressed matrix: ``in_ptr`` indexes its ``len(in_ptr) - 1``
    major vectors, whose entries ``in_idx`` lie in ``range(num_out)``.

    ``chunks`` holds increasing boundaries of major vectors, one chunk per row
    of the workspace ``counts`` (shape ``(len(chunks) - 1, num_out)``).
    """
    num_chunks = len(chunks) - 1
    for t in prange(num_chunks):
        row = counts[t]
        for j in range(num_out):
            row[j] = 0
        for jj in range(np.uintp(in_ptr[chunks[t]]), np.uintp(in_ptr[chunks[t + 1]])):
            row[np.uintp(in_idx[jj])] += 1
    for j in prange(num_out):
        total = 0
        for t in range(num_chunks):
            total += counts[t, j]
        out_ptr[j + 1] = total
    out_ptr[0] = 0
    for j in range(num_out):
        out_ptr[j + 1] += out_ptr[j]
    for j in prange(num_out):
        cursor = out_ptr[j]
        for t in range(num_chunks):
            count = counts[t, j]
            counts[t, j] = cursor
            cursor += count
    for t in prange(num_chunks):
        cursor = counts[t]
        for major in range(chunks[t], chunks[t + 1]):
            for jj in range(np.uintp(in_ptr[major]), np.uintp(in_ptr[major + 1])):
                j = np.uintp(in_idx[jj])
                dest = np.uintp(cursor[j])
                out_idx[dest] = major
                out_val[dest] = in_val[jj]
                cursor[j] += 1


@njit(cache=True, nogil=True)
def scatter_transposed(first_major, in_ptr, in_idx, in_val, cursor, out_idx, out_val):
    """The scatter pass of ``compressed_transpose`` for consecutive major vectors
    starting at ``first_major``, whose entries are ``in_idx``/``in_val`` at
    positions ``in_ptr - in_ptr[0]``. ``cursor`` holds the next output position
    of every output vector and advances."""
    base = np.uintp(in_ptr[0])
    for m in range(len(in_ptr) - 1):
        for jj in range(np.uintp(in_ptr[m]) - base, np.uintp(in_ptr[m + 1]) - base):
            j = np.uintp(in_idx[jj])
            dest = np.uintp(cursor[j])
            out_idx[dest] = first_major + m
            out_val[dest] = in_val[jj]
            cursor[j] += 1


@njit(cache=True, nogil=True)
def _popcount(word):
    word = word - ((word >> np.uint64(1)) & np.uint64(0x5555555555555555))
    word = (word & np.uint64(0x3333333333333333)) + ((word >> np.uint64(2)) & np.uint64(0x3333333333333333))
    word = (word + (word >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (word * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True, nogil=True, parallel=True)
def blocks_dot(
    x_indptr, x_indices, x_data, blocks, block_offset, block_col, row_bits, row_rank, row_ptr, row_ptr_offset,
    w_indices, w_data, out,
):
    """out[i, block_col[b] + j] += (x[i] @ W_b)[j] for every block b in blocks[i].

    Block b is a CSR matrix whose entries are at ``block_offset[b]`` onwards in
    ``w_indices``, ``w_data``. Only its nonempty rows have row pointers: row k is
    nonempty if bit k of ``row_bits[b]`` is set, and its entries are
    ``row_ptr[row_ptr_offset[b] + r]`` to the next pointer, where r counts the
    nonempty rows before k (``row_rank[b, k // 64]`` plus the bits below k in
    its word). Callers zero ``out`` and pass x with exactly
    ``64 * row_bits.shape[1]`` columns or fewer.
    """
    num_features = np.uintp(row_bits.shape[1] * 64)
    for i in prange(blocks.shape[0]):
        row = out[i]
        for t in range(blocks.shape[1]):
            b = np.uintp(blocks[i, t])
            offset = np.uintp(block_offset[b])
            col = np.uintp(block_col[b])
            bits = row_bits[b]
            rank = row_rank[b]
            ptr_offset = np.uintp(row_ptr_offset[b])
            for jj in range(np.uintp(x_indptr[i]), np.uintp(x_indptr[i + 1])):
                k = np.uintp(x_indices[jj])
                if k >= num_features:  # unreachable for valid inputs; avoids out-of-bounds reads
                    continue
                word = bits[k >> np.uintp(6)]
                shift = np.uint64(k & np.uintp(63))
                if (word >> shift) & np.uint64(1) == np.uint64(0):
                    continue
                r = ptr_offset + np.uintp(rank[k >> np.uintp(6)]) + np.uintp(_popcount(word & ((np.uint64(1) << shift) - np.uint64(1))))
                v = x_data[jj]
                for kk in range(offset + np.uintp(row_ptr[r]), offset + np.uintp(row_ptr[r + 1])):
                    row[col + np.uintp(w_indices[kk])] += v * w_data[kk]


@njit(cache=True, nogil=True)
def _insert(nodes, scores, length, capacity, node, score):
    """Insert into the first ``capacity`` entries of a stable decreasing order."""
    pos = length
    while pos > 0 and scores[pos - 1] < score:
        pos -= 1
    if pos >= capacity:
        return length
    end = length if length < capacity else capacity - 1
    for q in range(end, pos, -1):
        nodes[q] = nodes[q - 1]
        scores[q] = scores[q - 1]
    nodes[pos] = node
    scores[pos] = score
    return length + 1 if length < capacity else capacity


@njit(cache=True, nogil=True, parallel=True)
def beam_search(log_probs, beam_width, root, node_ptr, is_leaf, child_ptr, child_idx, beam_nodes, beam_scores, beam_size):
    """Beam search over every row of log_probs; writes each final beam and its size."""
    for i in prange(log_probs.shape[0]):
        cur_nodes = beam_nodes[i]
        cur_scores = beam_scores[i]
        next_nodes = np.empty(beam_width, dtype=np.int64)
        next_scores = np.empty(beam_width, dtype=np.float64)
        cur_nodes[0] = root
        cur_scores[0] = 0.0
        cur_len = 1
        while True:
            num_internal = 0
            for t in range(cur_len):
                if not is_leaf[cur_nodes[t]]:
                    num_internal += 1
            if num_internal == 0:
                break
            next_len = 0
            for t in range(cur_len):
                node = cur_nodes[t]
                score = cur_scores[t]
                if is_leaf[node]:
                    next_len = _insert(next_nodes, next_scores, next_len, beam_width, node, score)
                    continue
                first = child_ptr[node]
                offset = node_ptr[node] - first
                for c in range(first, child_ptr[node + 1]):
                    next_len = _insert(
                        next_nodes, next_scores, next_len, beam_width, child_idx[c], score + log_probs[i, offset + c]
                    )
            for t in range(next_len):
                cur_nodes[t] = next_nodes[t]
                cur_scores[t] = next_scores[t]
            cur_len = next_len
        beam_size[i] = cur_len


@njit(cache=True, nogil=True, parallel=True)
def beam_leaf_scores(log_probs, beam_nodes, beam_scores, beam_size, node_ptr, label_ptr, label_idx):
    """Rows, labels and log scores ``score + log_prob`` of the labels in each final beam."""
    num_rows = len(beam_size)
    start = np.zeros(num_rows + 1, dtype=np.int64)
    for i in prange(num_rows):
        total = 0
        for t in range(beam_size[i]):
            leaf = beam_nodes[i, t]
            total += label_ptr[leaf + 1] - label_ptr[leaf]
        start[i + 1] = total
    for i in range(num_rows):
        start[i + 1] += start[i]
    rows = np.empty(start[num_rows], dtype=np.int64)
    labels = np.empty(start[num_rows], dtype=np.int64)
    log_scores = np.empty(start[num_rows], dtype=np.float64)
    for i in prange(num_rows):
        n = start[i]
        for t in range(beam_size[i]):
            leaf = beam_nodes[i, t]
            score = beam_scores[i, t]
            first = node_ptr[leaf]
            for j in range(label_ptr[leaf + 1] - label_ptr[leaf]):
                rows[n] = i
                labels[n] = label_idx[label_ptr[leaf] + j]
                log_scores[n] = score + log_probs[i, first + j]
                n += 1
    return rows, labels, log_scores
