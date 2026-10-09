import logging
from concurrent.futures import ThreadPoolExecutor
from math import ceil

import numpy as np
from tqdm import tqdm

import libmultilabel.linear as linear
from libmultilabel.common_utils import dump_log, is_multiclass_dataset
from libmultilabel.linear.tree import EnsembleTreeModel, TreeModel, train_ensemble_tree
from libmultilabel.linear.utils import LINEAR_TECHNIQUES
from libmultilabel.linear.memory import log_memory

def linear_test(config, model, datasets, label_mapping):

    train_y = datasets["train"]["y"]
    test_y = datasets["test"]["y"]

    if train_y.shape[1] != test_y.shape[1]:
        raise ValueError("Train and test must use the same label mapping")

    label_pos_counts = np.asarray(
        train_y.astype(bool).sum(axis=0)
    ).ravel().astype(np.int64)

    metrics = linear.get_metrics(
        config.monitor_metrics,
        num_classes=test_y.shape[1],
        multiclass=model.multiclass,
        label_pos_counts=label_pos_counts,
        num_instances=train_y.shape[0],
    )
    
    num_instance = datasets["test"]["x"].shape[0]
    k = config.save_k_predictions
    if k > 0:
        labels = np.zeros((num_instance, k), dtype=object)
        scores = np.zeros((num_instance, k), dtype="d")
    else:
        labels = []
        scores = []

    predict_kwargs = {}
    if isinstance(model, (TreeModel, EnsembleTreeModel)):
        predict_kwargs["beam_width"] = config.beam_width

    # Update the metrics with one batch while predicting the next. Updates still
    # run one at a time, in batch order.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="libmultilabel-metrics") as executor:
        update = None
        for i in tqdm(range(ceil(num_instance / config.eval_batch_size))):
            slice = np.s_[i * config.eval_batch_size : (i + 1) * config.eval_batch_size]
            preds = model.predict_values(datasets["test"]["x"][slice], **predict_kwargs)
            target = datasets["test"]["y"][slice].toarray()
            if update is not None:
                update.result()
            update = executor.submit(metrics.update, preds, target)
            if k > 0:
                labels[slice], scores[slice] = linear.get_topk_labels(preds, label_mapping, config.save_k_predictions)
            elif config.save_positive_predictions:
                res = linear.get_positive_labels(preds, label_mapping)
                labels.append(res[0])
                scores.append(res[1])
        if update is not None:
            update.result()
    metric_dict = metrics.compute()
    return metric_dict, labels, scores


def linear_train(datasets, config):
    # detect task type
    multiclass = is_multiclass_dataset(datasets["train"], "y")

    # train
    if config.linear_technique == "tree":
        if multiclass:
            raise ValueError("Tree model should only be used with multilabel datasets.")

        if config.tree_ensemble_models > 1:
            model = train_ensemble_tree(
                datasets["train"]["y"],
                datasets["train"]["x"],
                options=config.liblinear_options,
                K=config.tree_degree,
                dmax=config.tree_max_depth,
                n_trees=config.tree_ensemble_models,
                seed=config.seed,
            )
        else:
            model = LINEAR_TECHNIQUES[config.linear_technique](
                datasets["train"]["y"],
                datasets["train"]["x"],
                options=config.liblinear_options,
                K=config.tree_degree,
                dmax=config.tree_max_depth,
            )
    else:
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets["train"]["y"],
            datasets["train"]["x"],
            multiclass=multiclass,
            options=config.liblinear_options,
        )
    return model


def linear_run(config):
    log_memory("pipeline: start")
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.eval:
        preprocessor, model = linear.load_pipeline(config.checkpoint_path)
        datasets = linear.load_dataset(config.data_format, config.training_file, config.test_file)
        datasets = preprocessor.transform(datasets)
    else:
        preprocessor = linear.Preprocessor(config.include_test_labels, config.remove_no_label_data)
        datasets = linear.load_dataset(
            config.data_format,
            config.training_file,
            config.test_file,
            config.label_file,
        )
        log_memory("data: loaded")
        datasets = preprocessor.fit_transform(datasets)
        log_memory("data: preprocessed")
        model = linear_train(datasets, config)
        log_memory("training: complete")
        # Evaluation needs training labels for propensity metrics, not features.
        datasets["train"].pop("x", None)
        log_memory("training: features released")
        linear.save_pipeline(config.checkpoint_dir, preprocessor, model)
        log_memory("checkpoint: saved")

    if config.test_file is not None:
        assert not (
            config.save_positive_predictions and config.save_k_predictions > 0
        ), """
            If save_k_predictions is larger than 0, only top k labels are saved.
            Save all labels with decision value larger than 0 by using save_positive_predictions and save_k_predictions=0."""
        log_memory("evaluation: start")
        metric_dict, labels, scores = linear_test(config, model, datasets, preprocessor.label_mapping)
        log_memory("evaluation: complete")
        
        dump_log(config=config, metrics=metric_dict, split="test", log_path=config.log_path)
        print(linear.tabulate_metrics(metric_dict, "test"))
        if config.save_k_predictions > 0:
            with open(config.predict_out_path, "w") as fp:
                for label, score in zip(labels, scores):
                    out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                    fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
        elif config.save_positive_predictions:
            with open(config.predict_out_path, "w") as fp:
                for batch_labels, batch_scores in zip(labels, scores):
                    for label, score in zip(batch_labels, batch_scores):
                        out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                        fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
