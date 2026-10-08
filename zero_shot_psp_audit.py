"""Compare supervised and zero-shot rankings on the same labels and checkpoint.

This diagnostic imports an explicitly selected checkout without modifying it.
It reports full-label PSP with one propensity model, and separately reports the
seen-only baseline. Candidate fusion policies are exploratory comparisons on a
fixed random test sample, not validation-selected hyperparameters.

The supervised reference uses the same ordering of seen labels as the RRF
predictor, including its tie handling, to isolate insertion of unseen labels.
The historic direct linear evaluator can resolve tied scores differently.
"""

import argparse
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer


def protect_top_k(supervised_scores, fused_scores, k):
    """Keep the supervised top k in order; retain fusion order below them.

    Output scores are for ranking only, not calibrated probabilities. The input
    supervised scores should use the same tie order as the supervised baseline.
    """
    if not 0 <= k <= supervised_scores.shape[1]:
        raise ValueError("protected k must be within the label dimension")
    result = fused_scores.copy()
    if k:
        selected = np.argpartition(supervised_scores, -k, axis=1)[:, -k:]
        order = np.argsort(-np.take_along_axis(supervised_scores, selected, axis=1), axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        promoted = result.max(axis=1, keepdims=True) + np.arange(k, 0, -1)[None, :]
        np.put_along_axis(result, selected, promoted, axis=1)
    return result


def label_ids(line):
    token = line.split(" ", 1)[0].strip()
    return sorted(set(map(int, token.split(",")))) if token else []


def read_labels(path):
    with path.open() as source:
        return [label_ids(line) for line in source]


def denominator(target, weights, k):
    weighted = np.where(target, weights, 0.0)
    return np.partition(weighted, -k, axis=1)[:, -k:].sum()


class RankingMetrics:
    """Accumulate sufficient statistics from descending top-100 label indices."""

    def __init__(self):
        self.rows = 0
        self.unseen_rows = 0
        self.sums = {}

    def add(self, name, value):
        self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def update(self, ranked, target, weights, unseen, denominators):
        hits = np.take_along_axis(target, ranked, axis=1)
        true_count = target.sum(axis=1)
        unseen_count = target[:, unseen].sum(axis=1)
        self.rows += target.shape[0]
        self.unseen_rows += int(np.count_nonzero(unseen_count))
        for k in (1, 3, 5):
            relevant = hits[:, :k]
            ranked_weights = weights[ranked[:, :k]]
            is_unseen = np.isin(ranked[:, :k], unseen)
            self.add(f"P@{k}", relevant.sum() / k)
            self.add(f"R@{k}", (relevant.sum(axis=1) / np.maximum(true_count, 1)).sum())
            discount = 1 / np.log2(np.arange(k) + 2)
            ideal_dcg = np.cumsum(discount)[np.maximum(np.minimum(true_count, k), 1) - 1]
            self.add(f"NDCG@{k}", ((relevant @ discount) / ideal_dcg).sum())
            self.add(f"weighted_hits@{k}", (relevant * ranked_weights).sum())
            self.add(f"unseen_weighted_hits@{k}", (relevant * ranked_weights * is_unseen).sum())
            self.add(f"denominator@{k}", denominators[k])
            self.add(f"unseen_predictions@{k}", is_unseen.sum())
        for k in (10, 50, 100):
            unseen_hits = (hits[:, :k] * np.isin(ranked[:, :k], unseen)).sum(axis=1)
            self.add(f"ZSR@{k}", (unseen_hits / np.maximum(unseen_count, 1)).sum())

    def compute(self):
        result = {name: value / self.rows for name, value in self.sums.items()
                  if name.startswith(("P@", "R@", "NDCG@"))}
        for k in (1, 3, 5):
            result[f"PSP@{k}"] = self.sums[f"weighted_hits@{k}"] / self.sums[f"denominator@{k}"]
        for k in (10, 50, 100):
            result[f"ZSR@{k}"] = self.sums[f"ZSR@{k}"] / max(self.unseen_rows, 1)
        return {"metrics": result, "raw_sums": self.sums}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16384, help="Random sample size; 0 uses the full test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--propensity-a", type=float, default=0.55)
    parser.add_argument("--propensity-b", type=float, default=1.5)
    args = parser.parse_args()
    if args.rows < 0 or args.batch_size < 1:
        parser.error("rows must be nonnegative and batch-size must be positive")
    start = time.perf_counter()
    sys.path.insert(0, str(args.repo.resolve()))
    import libmultilabel.linear as linear
    from libmultilabel.linear.metrics import PropensityScoredPrecisionAtK, _argsort_top_k

    spec = importlib.util.spec_from_file_location("audited_model_predict", args.repo / "zero_shot/model_predict.py")
    fusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)

    print("Loading checkpoint and label statistics", flush=True)
    preprocessor, model = linear.load_pipeline(str(args.checkpoint))
    n_features = model.flat_model.weights.shape[0] if hasattr(model, "flat_model") else model.weights.shape[0]
    with args.label_file.open() as source:
        n_labels = sum(1 for _ in source)
    counts = np.zeros(n_labels, dtype=np.int64)
    n_train = 0
    with args.train_file.open() as source:
        for line in source:
            counts[label_ids(line)] += 1
            n_train += 1
    seen = np.flatnonzero(counts)
    unseen = np.flatnonzero(counts == 0)
    np.testing.assert_array_equal(preprocessor.label_mapping, seen)
    propensity = PropensityScoredPrecisionAtK(1, counts, n_train, args.propensity_a, args.propensity_b)
    weights = propensity.inv_propensity

    y_test = read_labels(args.test_file)
    n_test = len(y_test)
    sample = np.sort(np.random.default_rng(args.seed).choice(n_test, min(args.rows or n_test, n_test), replace=False))
    wanted = set(sample)
    with args.test_file.open("rb") as source:
        sample_bytes = b"".join(line for i, line in enumerate(source) if i in wanted)
    x, _ = load_svmlight_file(io.BytesIO(sample_bytes), multilabel=True, zero_based=False, n_features=n_features)
    del sample_bytes
    label_features, _ = load_svmlight_file(str(args.label_file), multilabel=True, zero_based=False, n_features=n_features)
    if label_features.shape[0] != n_labels:
        raise ValueError("Blank label-feature rows must be preserved before evaluating label alignment")
    binarizer = MultiLabelBinarizer(classes=np.arange(n_labels), sparse_output=True)
    target_sparse = binarizer.fit_transform([y_test[i] for i in sample])

    names = ["supervised_only", "rrf_beta_0.01", "rrf_beta_0.03", "rrf_beta_0.07", "rrf_beta_0.10", "rrf_protect_top5"]
    accumulators = {name: RankingMetrics() for name in names}
    seen_reference = {k: {"numerator": 0.0, "denominator": 0.0} for k in (1, 3, 5)}
    tie_counts = {"raw_top1_tie_rows": 0, "rrf_vs_direct_top1_difference_rows": 0}
    print(f"Predicting {len(sample)} random rows with {len(seen)} seen and {len(unseen)} unseen labels", flush=True)
    for offset in range(0, len(sample), args.batch_size):
        batch = x[offset:offset + args.batch_size]
        target = target_sparse[offset:offset + args.batch_size].toarray().astype(bool)
        raw = model.predict_values(batch)
        supervised = np.full(target.shape, -np.inf)
        supervised[:, seen] = 1 / (1 + np.exp(-raw))
        supervised_ranks = fusion.MixedPredictor.ranks_of(supervised)
        doc_scores = (batch @ label_features.T).toarray()
        doc_ranks = fusion.MixedPredictor.ranks_of(doc_scores)
        base = np.full(target.shape, -np.inf)
        base[:, seen] = 1.0 / (supervised_ranks[:, seen] + fusion.RRF_K)
        variants = {"supervised_only": base}
        for beta in (0.01, 0.03, 0.07, 0.10):
            fused = base.copy()
            fused[:, unseen] = (1 - beta) * (1.0 / (doc_ranks[:, unseen] + fusion.RRF_K))
            variants[f"rrf_beta_{beta:.2f}"] = fused
        if offset == 0:
            reference = fusion.MixedPredictor.__new__(fusion.MixedPredictor)
            reference.strategy = "rank_rrf"
            reference.seen_mask = counts > 0
            reference.unseen_mask = ~reference.seen_mask
            expected = reference._combine_scores(np.zeros_like(base), "zero", 1, 0.01,
                None, None, None, None, supervised_ranks, doc_ranks)
            np.testing.assert_array_equal(variants["rrf_beta_0.01"], expected)
        variants["rrf_protect_top5"] = protect_top_k(base, variants["rrf_beta_0.01"], 5)
        full_denoms = {k: denominator(target, weights, k) for k in (1, 3, 5)}
        ranked_baseline = _argsort_top_k(base, 100)[:, ::-1]
        tie_counts["raw_top1_tie_rows"] += int(np.sum((raw == raw.max(axis=1, keepdims=True)).sum(axis=1) > 1))
        tie_counts["rrf_vs_direct_top1_difference_rows"] += int(np.sum(seen[_argsort_top_k(raw, 5)[:, -1]] != ranked_baseline[:, 0]))
        for k in (1, 3, 5):
            indices = ranked_baseline[:, :k]
            seen_reference[k]["numerator"] += float((np.take_along_axis(target, indices, axis=1) * weights[indices]).sum())
            seen_reference[k]["denominator"] += float(denominator(target[:, seen], weights[seen], k))
        for name, scores in variants.items():
            ranked = _argsort_top_k(scores, 100)[:, ::-1]
            if name in ("rrf_protect_top5", "rrf_beta_0.07", "rrf_beta_0.10"):
                np.testing.assert_array_equal(ranked[:, :5], ranked_baseline[:, :5])
            accumulators[name].update(ranked, target, weights, unseen, full_denoms)
            if offset == 0:
                # Cross-check the independent diagnostic against the repository's metrics.
                check = linear.get_metrics(fusion.METRIC_LIST, n_labels, unseen_labels=unseen,
                    label_pos_counts=counts, num_instances=n_train,
                    propensity_a=args.propensity_a, propensity_b=args.propensity_b)
                check.update(scores, target)
                actual = accumulators[name].compute()["metrics"]
                for metric, value in check.compute().items():
                    np.testing.assert_allclose(actual[metric], value, rtol=1e-12, atol=1e-12)
        if offset % (args.batch_size * 16) == 0:
            print(f"Processed {min(offset + args.batch_size, len(sample))}/{len(sample)} rows", flush=True)

    report = {
        "repo": str(args.repo.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "train_file": str(args.train_file.resolve()), "test_file": str(args.test_file.resolve()),
        "label_file": str(args.label_file.resolve()), "train_rows": n_train,
        "full_test_rows": n_test, "sample_rows": len(sample), "sample_seed": args.seed,
        "sample_rows_with_unseen": accumulators["supervised_only"].unseen_rows,
        "seen_labels": len(seen), "unseen_labels": len(unseen),
        "propensity_a": args.propensity_a, "propensity_b": args.propensity_b,
        "all_scores_are_fractions": True,
        "seen_only_supervised_psp": {f"PSP@{k}": v["numerator"] / v["denominator"] for k, v in seen_reference.items()},
        "variants": {name: result.compute() for name, result in accumulators.items()},
        "tie_diagnostics": tie_counts,
        "top5_preservation_verified": True, "seconds": time.perf_counter() - start,
        "baseline_ordering": "RRF seen-label ordering, including its tie handling, with unseen labels excluded.",
        "interpretation": "Exploratory test-sample audit; do not select deployment hyperparameters on test results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.save(args.output.with_suffix(".sample_indices.npy"), sample)
    print(json.dumps({name: v["metrics"] for name, v in report["variants"].items()}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
