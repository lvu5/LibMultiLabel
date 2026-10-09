from __future__ import annotations

import codecs
import csv
import io
import locale
import logging
import multiprocessing
import os
import re
import sys
from array import array
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sparse

__all__ = ["load_dataset"]


def _read_libmultilabel_format(data: str | pd.Dataframe) -> dict[str, list[str]]:
    """Read multi-label text data from file or pandas dataframe.

    Args:
        data ('str | pd.Dataframe'): A file path to data in `LibMultiLabel format <https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/cli/ov_data_format.html#libmultilabel-format>`_
            or a pandas dataframe contains index (optional), label, and text.

    Returns:
        dict[str,list[str]]: A dictionary with a list of index (optional), label, and text.
    """
    assert isinstance(data, str) or isinstance(data, pd.DataFrame), "Data must be from a file or pandas dataframe."
    if isinstance(data, str):
        data = pd.read_csv(data, sep="\t", header=None, on_bad_lines="warn", quoting=csv.QUOTE_NONE).fillna("")
    data = data.astype(str)
    if data.shape[1] == 2:
        data.columns = ["y", "x"]
        data = data.reset_index()
    elif data.shape[1] == 3:
        data.columns = ["idx", "y", "x"]
    else:
        raise ValueError(f"Expected 2 or 3 columns, got {data.shape[1]}.")
    data["y"] = data["y"].map(lambda s: s.split())
    return data.to_dict("list")


class _SvmFormatError(Exception):
    """An invalid line, identified by its index among the parsed lines."""

    def __init__(self, line_index: int, invalid_index: bool):
        super().__init__(line_index)
        self.line_index = line_index
        self.invalid_index = invalid_index

    def to_public(self, first_line: int, file_path: str) -> Exception:
        line = first_line + self.line_index + 1
        if self.invalid_index:
            return IndexError(
                f"invalid svm format at line {line} of the file '{file_path}' --> Indices should start from one."
            )
        return ValueError(f"invalid svm format at line {line} of the file '{file_path}'")


def _parse_libsvm_lines(lines) -> tuple[list[list[int]], array, array, array]:
    """Parse lines of LIBSVM-format data into labels, values, column indices and row sizes."""
    prob_y = []
    prob_x = array("d")
    row_nnz = array("l")
    col_idx = array("l")

    pattern = re.compile(r"(?!^$)([+\-0-9,]+\s+)?(.*\n?)")
    for i, line in enumerate(lines):
        m = pattern.fullmatch(line)
        try:
            labels = m[1]
            int_labels = [int(s) for s in labels.split(",")] if labels else []
            prob_y.append(int_labels)
            features = m[2] or ""
            nz = 0
            for e in features.split():
                idx, val = e.split(":")
                idx, val = int(idx), float(val)
                if idx < 1:
                    raise _SvmFormatError(i, invalid_index=True)
                if val != 0:
                    col_idx.append(idx - 1)
                    prob_x.append(val)
                    nz += 1
            row_nnz.append(nz)
        except _SvmFormatError:
            raise
        except:
            raise _SvmFormatError(i, invalid_index=False)
    return prob_y, prob_x, col_idx, row_nnz


# Files at least this large are parsed by worker processes, in chunks of whole lines.
_PARALLEL_READ_MIN_BYTES = 64 * 1024**2
_READ_CHUNK_BYTES = 16 * 1024**2


def _parse_libsvm_chunk(task):
    """Parse the lines in a byte range of a file as open(file_path) would read them."""
    file_path, start, stop, encoding = task
    with open(file_path, "rb") as f:
        f.seek(start)
        lines = io.TextIOWrapper(io.BytesIO(f.read(stop - start)), encoding=encoding)
    try:
        prob_y, prob_x, col_idx, row_nnz = _parse_libsvm_lines(lines)
    except _SvmFormatError as error:
        # A plain tuple: exceptions with extra constructor arguments do not unpickle.
        return error.line_index, error.invalid_index
    col_idx = np.frombuffer(col_idx, dtype="l")
    if col_idx.size == 0 or col_idx.max() <= np.iinfo(np.int32).max:
        col_idx = col_idx.astype(np.int32)  # halves the transfer; csr_matrix stores int32 indices anyway
    return prob_y, np.frombuffer(prob_x, dtype="d"), col_idx, np.frombuffer(row_nnz, dtype="l")


def _read_libsvm_chunks(file_path: str):
    """Parse a large file with worker processes. Returns None if the file should be read sequentially."""
    size = os.path.getsize(file_path)
    num_workers = min(8, psutil.cpu_count(logical=False) or 1)
    # Text mode reads with the locale encoding. Chunks split after b"\n", which
    # is never part of a multi-byte character in UTF-8.
    encoding = "utf-8" if sys.flags.utf8_mode else locale.getpreferredencoding(False)
    if (
        size < _PARALLEL_READ_MIN_BYTES
        or num_workers < 2
        or "fork" not in multiprocessing.get_all_start_methods()
        or codecs.lookup(encoding).name not in {"utf-8", "ascii"}
    ):
        return None

    bounds = [0]
    with open(file_path, "rb") as f:
        while bounds[-1] < size:
            f.seek(min(bounds[-1] + _READ_CHUNK_BYTES, size))
            f.readline()
            bounds.append(min(f.tell(), size))
    tasks = [(file_path, start, stop, encoding) for start, stop in zip(bounds[:-1], bounds[1:])]

    prob_y, prob_x, col_idx, row_nnz = [], [], [], []
    with multiprocessing.get_context("fork").Pool(min(num_workers, len(tasks))) as pool:
        for result in pool.imap(_parse_libsvm_chunk, tasks):
            if len(result) == 2:
                # Lines before this chunk are all valid, so their count numbers the error.
                raise _SvmFormatError(*result).to_public(sum(len(n) for n in row_nnz), file_path)
            prob_y.extend(result[0])
            prob_x.append(result[1])
            col_idx.append(result[2])
            row_nnz.append(result[3])
    return prob_y, np.concatenate(prob_x), np.concatenate(col_idx), np.concatenate(row_nnz)


def _read_libsvm_format(file_path: str) -> dict[str, list[list[int]] | sparse.csr_matrix]:
    """Read multi-label LIBSVM-format data.

    Large files are parsed in parallel; the result is the same as a sequential read.

    Args:
        file_path (str): Path to file.

    Returns:
        tuple[list[list[int]], sparse.csr_matrix]: A tuple of labels and features.
    """
    parsed = _read_libsvm_chunks(file_path)
    if parsed is None:
        try:
            prob_y, prob_x, col_idx, row_nnz = _parse_libsvm_lines(open(file_path))
        except _SvmFormatError as error:
            raise error.to_public(0, file_path)
        prob_x = np.frombuffer(prob_x, dtype="d")
        col_idx = np.frombuffer(col_idx, dtype="l")
        row_nnz = np.frombuffer(row_nnz, dtype="l")
    else:
        prob_y, prob_x, col_idx, row_nnz = parsed

    row_ptr = np.zeros(len(row_nnz) + 1, dtype="l")
    np.cumsum(row_nnz, out=row_ptr[1:])
    prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))

    return {"x": prob_x, "y": prob_y}


def load_dataset(
    data_format: str,
    train_path: str | pd.DataFrame | None = None,
    test_path: str | pd.DataFrame | None = None,
    label_path: str | None = None,
) -> dict[str, dict[str, sparse.csr_matrix | list[list[int]] | list[str]]]:
    """Load dataset in LibSVM or LibMultiLabel formats.

    Args:
        data_format (str): The data format used. 'svm' for LibSVM format, 'txt' for LibMultiLabel format in file and 'dataframe' for LibMultiLabel format in dataframe .
        train_path (str | pd.DataFrame, optional): Training data file or dataframe in LibMultiLabel format. Ignored if eval is True. Defaults to None.
        test_path (str | pd.DataFrame, optional): Test data file or dataframe in LibMultiLabel format. Ignored if test_data doesn't exist. Defaults to None.
        label_path (str, optional): Path to a file holding all labels. Defaults to None.

    Returns:
        dict[str, dict[str, sparse.csr_matrix | str]]: The training and/or test data, with keys 'train' and 'test' respectively.
        The data has keys 'x' for input features and 'y' for labels.
    """
    if data_format not in {"txt", "svm", "dataframe"}:
        raise ValueError(f"unsupported data format {data_format}")
    if train_path is None and test_path is None:
        raise ValueError("train_path and test_path cannot be both None.")

    dataset = defaultdict(dict)
    dataset["data_format"] = data_format

    # load training and test datasets
    if data_format in {"txt", "dataframe"}:
        if train_path is not None:
            train = _read_libmultilabel_format(train_path)
        if test_path is not None:
            test = _read_libmultilabel_format(test_path)
    if data_format in {"svm"}:
        if train_path is not None:
            train = _read_libsvm_format(train_path)
        if test_path is not None:
            test = _read_libsvm_format(test_path)
    if train_path is not None:
        dataset["train"]["x"] = train["x"]
        dataset["train"]["y"] = train["y"]
    if test_path is not None:
        dataset["test"]["x"] = test["x"]
        dataset["test"]["y"] = test["y"]

    # load labels
    if label_path is not None:
        logging.info(f"Load labels from {label_path}.")
        with open(label_path) as fp:
            dataset["classes"] = sorted([c.strip() for c in fp.readlines()])

    return dict(dataset)
