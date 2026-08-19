import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    r2_score as sklearn_r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_probe_data import get_section_shapes, load_data
from probe_runner import (
    build_feature_lookup,
    fit_probe,
    select_feature_groups,
    split_train_test,
)
from utils import RESULTS_ROOT, pick_probe

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


SYNTAX_COMPONENT_TARGETS = [
    "syntax_POS_OH",
    "syntax_Dependency_Label_OH",
    "syntax_Tree_Depth",
    "syntax_Word_Position",
    "syntax_Total_Tree_Depth",
    "syntax_Total_Word_Count",
]
CATEGORICAL_SYNTAX_TARGETS = set(t for t in SYNTAX_COMPONENT_TARGETS if "_OH" in t)

EXPERIMENTAL_CONFIG = [
    {
        "category": "lexicon_syntax",
        "source_type": "input_group_list",
        "source_groups": [["word_embedding"]],
        "target_type": "syntax_components",
        "config_template": "{x_label}->{target_name}",
        "modelname": "input-feature",
        "input_features_only": True,
    },
    {
        "category": "input_speaker",
        "source_type": "input_group_list",
        "source_groups": [
            ["ppg_feature"],
            ["eGeMAPSv02"],
            ["ppg_feature", "eGeMAPSv02"],
        ],
        "target_type": "static_targets",
        "targets": [
            {
                "name": "SpeakerID",
                "context_key": "speaker_labels",
                "task_type": "classification",
            }
        ],
        "config_template": "{x_label}->{target_name}",
        "modelname": "input-feature",
        "input_features_only": True,
    },
    {
        "category": "hidden_speaker",
        "source_type": "hidden_layers",
        "target_type": "static_targets",
        "targets": [
            {
                "name": "SpeakerID",
                "context_key": "speaker_labels",
                "task_type": "classification",
            }
        ],
        "config_template": "{x_label}->{target_name}",
        "modelname": "args",
    },
    {
        "category": "hidden_phoneid",
        "source_type": "hidden_layers",
        "target_type": "static_targets",
        "targets": [
            {
                "name": "PhoneID",
                "context_key": "phone_labels",
                "task_type": "classification",
            }
        ],
        "config_template": "{x_label}->{target_name}",
        "modelname": "args",
    },
    {
        "category": "hidden_syntax",
        "source_type": "hidden_layers",
        "target_type": "syntax_components",
        "config_template": "{x_label}->{target_name}",
        "modelname": "args",
    },
    {
        "category": "classification_baselines",
        "source_type": "none",
        "target_type": "majority_class_targets",
        "config_template": "MajorityClass->{target_name}",
        "modelname": "majority-baseline",
    },
    {
        "category": "hidden_word_embedding",
        "source_type": "hidden_layers",
        "target_type": "static_targets",
        "targets": [
            {
                "name": "word_embedding",
                "context_key": "word_embedding",
                "task_type": "regression",
            }
        ],
        "config_template": "{x_label}->{target_name}",
        "modelname": "args",
    },
    {
        "category": "hidden_syntax_feature",
        "source_type": "hidden_layers",
        "target_type": "static_targets",
        "targets": [
            {
                "name": "syntax_feature",
                "context_key": "syntax_feature",
                "task_type": "regression",
            }
        ],
        "config_template": "{x_label}->{target_name}",
        "modelname": "args",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="decoding check overlap among probe input features."
    )
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="train-clean-100",
        help="LibriSpeech split.",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="facebook/wav2vec2-base",
        help="Model name used for hidden states.",
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        default="ridge",
        help="Regression probe name for continuous targets.",
    )
    parser.add_argument(
        "--classifier_probe_name",
        type=str,
        default="ridge_classifier",
        help="Classifier probe name for categorical targets.",
    )
    parser.add_argument(
        "--select_layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of hidden-state layers.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used by upstream preprocessing cache naming.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached formatted frame-level data.",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        help="Normalize acoustic features inside load_data.",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip probe runs and only aggregate existing CSV results.",
    )
    return parser.parse_args()


def _catch_too_few_classes(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove classes with fewer than 2 samples to avoid train/test split issues
    And log a warning with the affected classes and their counts

    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): class labels

    Returns:
        tuple[np.ndarray, np.ndarray]: filtered X and y with rare classes removed
    """

    unique_labels, label_counts = np.unique(y, return_counts=True)
    rare_classes = unique_labels[label_counts < 2]
    if len(rare_classes) > 0:
        logger.warning(
            "Found %s rare classes with fewer than 2 samples: %s",
            len(rare_classes),
            dict(zip(rare_classes, label_counts[label_counts < 2])),
        )
        keep_mask = ~np.isin(y, rare_classes)
        X = X[keep_mask]
        y = y[keep_mask]
        logger.info(
            "After removing rare classes, %s samples remain across %s classes.",
            len(y),
            len(np.unique(y)),
        )
    return X, y


def _run_classification_probe(
    estimator,
    param_grid,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    # Remove classes with fewer than 2 samples to avoid train/test split issues
    # And log a warning with the affected classes and their counts
    X, y = _catch_too_few_classes(X, y)

    x_train, x_test, y_train, y_test = split_train_test(
        X,
        y,
        stratify_labels=y.tolist(),
    )
    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = int(class_counts.min())
    cv_splits = max(2, min(5, min_class_count))

    if cv_splits < 5:
        logger.info(
            "Reducing classifier CV folds from 5 to %s due to rare classes "
            "(min class count=%s).",
            cv_splits,
            min_class_count,
        )
    grid = fit_probe(
        estimator,
        param_grid,
        x_train,
        y_train,
        scoring="accuracy",
        cv=cv_splits,
    )
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    return {
        "train_score": float(best_model.score(x_train, y_train)),
        "test_score": float(best_model.score(x_test, y_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "best_params": grid.best_params_,
    }


def _with_input_scaler(estimator, param_grid):
    """Wrap estimator in StandardScaler pipeline and remap param grid keys."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )
    scaled_param_grid = {
        f"estimator__{key}": value for key, value in param_grid.items()
    }
    return pipeline, scaled_param_grid


def _run_regression_probe(
    estimator,
    param_grid,
    X: np.ndarray,
    y: np.ndarray,
    *,
    stratify_labels: list[str],
) -> dict:
    """Memory-friendly local regression probe for decoding checks."""

    x_train, x_test, y_train, y_test = split_train_test(
        X,
        y,
        stratify_labels=stratify_labels,
    )
    grid = fit_probe(
        estimator,
        param_grid,
        x_train,
        y_train,
        scoring="r2",
        cv=3,
    )
    best_model = grid.best_estimator_
    train_score = sklearn_r2_score(
        y_train,
        best_model.predict(x_train),
        multioutput="variance_weighted",
    )
    test_score = sklearn_r2_score(
        y_test,
        best_model.predict(x_test),
        multioutput="variance_weighted",
    )
    return {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "best_params": grid.best_params_,
    }


def _build_category_savepath(
    args: argparse.Namespace,
    category: str,
    input_features_only: bool = False,
) -> str:
    split_id = args.librispeech_split
    if args.random_seed != 42:
        split_id = f"{split_id}_seed{args.random_seed}"
    model_slug = args.modelname.replace("/", "-")
    if input_features_only:
        model_slug = "input-feature"

    return os.path.join(
        RESULTS_ROOT,
        f"decoding_results/{category}_{model_slug}_{split_id}.csv",
    )


def aggregate_saved_results(
    librispeech_split: str,
    modelnames: list[str] | None = None,
    random_seed: int = 42,
    save_combined: bool = True,
) -> dict[str, pd.DataFrame]:
    """Aggregate category-level CSV results into combined DataFrame.

    Reads category files from figures/ directory and combines them across models.
    Each category file already contains all results for that category/model/split.

    Args:
        librispeech_split: LibriSpeech split name.
        modelnames: List of model names to aggregate. If None, finds all matching files.
        random_seed: Random seed used in original runs.
        save_combined: Whether to save the combined 'all' CSV.

    Returns:
        Dictionary with 'all' key containing combined DataFrame.
    """
    split_id = librispeech_split
    if random_seed != 42:
        split_id = f"{split_id}_seed{random_seed}"

    categories = [
        "lexicon_syntax",
        "input_speaker",
        "hidden_speaker",
        "hidden_phoneid",
        "hidden_syntax",
        "classification_baselines",
    ]

    if modelnames is None:
        pattern = os.path.join(RESULTS_ROOT, "decoding_results", f"*_{split_id}.csv")
        all_category_files = sorted(glob.glob(pattern))
    else:
        all_category_files = []
        for modelname in modelnames:
            model_slug = modelname.replace("/", "-")
            for category in categories:
                filepath = os.path.join(
                    RESULTS_ROOT,
                    "decoding_results",
                    f"{category}_{model_slug}_{split_id}.csv",
                )
                if os.path.isfile(filepath):
                    all_category_files.append(filepath)

    if not all_category_files:
        logger.warning(
            "No category CSV files found for split=%s, seed=%s in %s",
            librispeech_split,
            random_seed,
            RESULTS_ROOT,
        )
        return {"all": pd.DataFrame()}

    logger.info("Found %s category CSV files to aggregate", len(all_category_files))

    category_frames = [pd.read_csv(path) for path in all_category_files]
    combined_df = pd.concat(category_frames, ignore_index=True)

    if combined_df.empty:
        logger.warning("Combined DataFrame is empty after aggregation.")
        return {"all": pd.DataFrame()}

    logger.info("Aggregated columns: %s", combined_df.columns.tolist())

    combined_df["layer"] = combined_df["layer"].fillna(-1)
    combined_df["modelname"] = combined_df.apply(
        lambda row: "input-feature"
        if "hidden_state" not in str(row.get("x_groups", ""))
        else row["modelname"],
        axis=1,
    )
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.sort_values(
        by=["modelname", "layer", "config_name"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    if save_combined:
        combined_path = os.path.join(
            RESULTS_ROOT,
            f"combined_all_decoding_{split_id}.csv",
        )
        combined_df.to_csv(combined_path, index=False)
        logger.info("Saved combined all decoding CSV to %s", combined_path)

    return {"all": combined_df}


def _get_csv_fieldnames() -> list[str]:
    preferred_order = [
        "librispeech_split",
        "modelname",
        "config_name",
        "x_groups",
        "y_groups",
        "metric",
        "layer",
        "train_score",
        "test_score",
        "balanced_accuracy",
        "f1_macro",
        "best_params",
        "n_samples",
    ]
    return preferred_order


def _save_category_results(savepath: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    df = pd.DataFrame(rows)
    fieldnames = _get_csv_fieldnames()
    df = df[[col for col in fieldnames if col in df.columns]]
    df.to_csv(savepath, index=False)


def _get_layer_labels(
    select_layers: list[int] | None,
    model_hidden_states: np.ndarray,
) -> list[int]:
    if select_layers is not None:
        return select_layers
    return list(range(model_hidden_states.shape[1]))


def _build_syntax_feature_vector(
    feature_sets: np.ndarray,
    lookup: dict,
) -> np.ndarray:
    syntax_parts = []
    for syntax_target in SYNTAX_COMPONENT_TARGETS:
        y_part = select_feature_groups(feature_sets, lookup, [syntax_target])
        if syntax_target in CATEGORICAL_SYNTAX_TARGETS:
            y_part = np.argmax(y_part, axis=1, keepdims=True)
        syntax_parts.append(y_part)

    if not syntax_parts:
        logger.warning("No syntax features found to build syntax_feature vector.")
        raise ValueError("No syntax features found for syntax_feature vector.")

    return np.concatenate(syntax_parts, axis=1)


def _build_probe_row(
    *,
    args: argparse.Namespace,
    modelname: str,
    config_name: str,
    x_groups: list[str],
    y_groups: list[str],
    metric: str,
    n_samples: int,
    train_score: float,
    test_score: float,
    best_params: dict,
    layer: int | None = None,
    balanced_accuracy: float | None = None,
    f1_macro: float | None = None,
) -> dict:
    row = {
        "librispeech_split": args.librispeech_split,
        "modelname": modelname,
        "config_name": config_name,
        "x_groups": x_groups,
        "y_groups": y_groups,
        "metric": metric,
        "train_score": train_score,
        "test_score": test_score,
        "best_params": best_params,
        "n_samples": n_samples,
    }
    if balanced_accuracy is not None:
        row["balanced_accuracy"] = balanced_accuracy
    if f1_macro is not None:
        row["f1_macro"] = f1_macro
    if layer is not None:
        row["layer"] = int(layer)
    return row


def _run_probe_task(
    *,
    task_type: str,
    X: np.ndarray | None,
    y: np.ndarray,
    reg_estimator,
    reg_param_grid,
    clf_estimator,
    clf_param_grid,
    speaker_ids: list[str],
) -> tuple[dict, str]:
    if task_type == "majority_baseline":
        _, label_counts = np.unique(y, return_counts=True)
        majority_class_proportion = label_counts.max() / len(y)
        return {
            "train_score": float(majority_class_proportion),
            "test_score": float(majority_class_proportion),
            "best_params": {},
        }, "accuracy"

    if task_type == "classification":
        if X is None:
            raise ValueError("X must be provided for classification task")
        result = _run_classification_probe(
            clf_estimator,
            clf_param_grid,
            X,
            y,
        )
        return result, "accuracy"
    if task_type == "regression":
        if X is None:
            raise ValueError("X must be provided for regression task")
        result = _run_regression_probe(
            reg_estimator,
            reg_param_grid,
            X,
            y,
            stratify_labels=speaker_ids,
        )
        return result, "r2"
    raise ValueError(f"Unsupported task_type: {task_type}")


def _build_source_variants(experiment: dict, context: dict) -> list[dict]:
    feature_sets = context["feature_sets"]
    model_hidden_states = context["model_hidden_states"]
    lookup = context["lookup"]
    layer_labels = context["layer_labels"]

    source_type = experiment["source_type"]
    if source_type == "input_group_list":
        variants = []
        for source_groups in experiment["source_groups"]:
            variants.append(
                {
                    "x_groups": source_groups,
                    "x_label": "+".join(source_groups),
                    "X": select_feature_groups(feature_sets, lookup, source_groups),
                    "n_samples": int(feature_sets.shape[0]),
                }
            )
        return variants

    if source_type == "hidden_layers":
        return [
            {
                "x_groups": [f"hidden_state_L{layer_label}"],
                "x_label": f"hidden_state_L{layer_label}",
                "X": model_hidden_states[:, layer_idx, :],
                "n_samples": int(model_hidden_states.shape[0]),
                "layer": int(layer_label),
            }
            for layer_idx, layer_label in enumerate(layer_labels)
        ]

    if source_type == "none":
        return [
            {
                "x_groups": ["none"],
                "x_label": "none",
                "X": None,
                "n_samples": int(feature_sets.shape[0]),
            }
        ]

    raise ValueError(f"Unsupported source_type: {source_type}")


def _build_target_variants(experiment: dict, context: dict) -> list[dict]:
    target_type = experiment["target_type"]

    if target_type == "static_targets":
        return [
            {
                "target_name": target_spec["name"],
                "y": context[target_spec["context_key"]],
                "task_type": target_spec["task_type"],
            }
            for target_spec in experiment["targets"]
        ]

    if target_type == "syntax_components":
        variants = []
        for target_name in SYNTAX_COMPONENT_TARGETS:
            y_target = context["syntax_targets"][target_name]
            is_categorical = target_name in CATEGORICAL_SYNTAX_TARGETS
            variants.append(
                {
                    "target_name": target_name,
                    "y": np.argmax(y_target, axis=1) if is_categorical else y_target,
                    "task_type": "classification" if is_categorical else "regression",
                }
            )
        return variants

    if target_type == "majority_class_targets":
        variants = [
            {
                "target_name": "SpeakerID",
                "y": context["speaker_labels"],
                "task_type": "majority_baseline",
            },
            {
                "target_name": "PhoneID",
                "y": context["phone_labels"],
                "task_type": "majority_baseline",
            },
        ]
        for target_name in CATEGORICAL_SYNTAX_TARGETS:
            variants.append(
                {
                    "target_name": target_name,
                    "y": np.argmax(context["syntax_targets"][target_name], axis=1),
                    "task_type": "majority_baseline",
                }
            )
        return variants

    raise ValueError(f"Unsupported target_type: {target_type}")


def _build_probe_specs_for_experiment(experiment: dict, context: dict) -> list[dict]:
    modelname = (
        context["args"].modelname
        if experiment.get("modelname") == "args"
        else experiment["modelname"]
    )
    source_variants = _build_source_variants(experiment, context)
    target_variants = _build_target_variants(experiment, context)

    specs = []
    for source_variant in source_variants:
        for target_variant in target_variants:
            specs.append(
                {
                    "config_name": experiment["config_template"].format(
                        x_label=source_variant["x_label"],
                        target_name=target_variant["target_name"],
                    ),
                    "x_groups": source_variant["x_groups"],
                    "y_groups": [target_variant["target_name"]],
                    "modelname": modelname,
                    "X": source_variant["X"],
                    "y": target_variant["y"],
                    "task_type": target_variant["task_type"],
                    "n_samples": source_variant["n_samples"],
                    "layer": source_variant.get("layer"),
                }
            )
    return specs


def _run_category_experiment(
    args: argparse.Namespace,
    experiment: dict,
    context: dict,
    reg_estimator,
    reg_param_grid,
    clf_estimator,
    clf_param_grid,
) -> int:
    category = experiment["category"]
    savepath = _build_category_savepath(
        args,
        category=category,
        input_features_only=experiment.get("input_features_only", False),
    )

    if os.path.isfile(savepath) and not args.overwrite:
        logger.info("Skipping %s category: %s", category, savepath)
        return 0

    rows = []
    for spec in _build_probe_specs_for_experiment(experiment, context):
        logger.info("Running probe: %s", spec["config_name"])
        probe_result, metric = _run_probe_task(
            task_type=spec["task_type"],
            X=spec["X"],
            y=spec["y"],
            reg_estimator=reg_estimator,
            reg_param_grid=reg_param_grid,
            clf_estimator=clf_estimator,
            clf_param_grid=clf_param_grid,
            speaker_ids=context["speaker_ids"],
        )
        rows.append(
            _build_probe_row(
                args=args,
                modelname=spec["modelname"],
                config_name=spec["config_name"],
                x_groups=spec["x_groups"],
                y_groups=spec["y_groups"],
                metric=metric,
                n_samples=spec["n_samples"],
                train_score=probe_result["train_score"],
                test_score=probe_result["test_score"],
                best_params=probe_result["best_params"],
                layer=spec.get("layer"),
                balanced_accuracy=probe_result.get("balanced_accuracy"),
                f1_macro=probe_result.get("f1_macro"),
            )
        )

    _save_category_results(savepath, rows)
    logger.info("Saved %s results to %s", category, savepath)
    return len(rows)


def main() -> None:
    args = parse_args()

    if args.aggregate_only:
        logger.info("Running in aggregate-only mode. Skipping probe runs.")
        aggregate_saved_results(
            librispeech_split=args.librispeech_split,
            random_seed=args.random_seed,
            save_combined=True,
        )
        return

    selected_input_components = [
        "word_embedding",
        "syntax_feature",
        "ppg_feature",
        "eGeMAPSv02",
        "metadata",
    ]

    logger.info("Loading data via load_data(...)...")
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=args.librispeech_split,
        modelname=args.modelname,
        selected_input_components=selected_input_components,
        seq_sampling="random_frames",
        select_layers=args.select_layers,
        overwrite=args.overwrite,
        one_hot_encode_syntax=False,
        one_hot_encode_syntax_separate=True,
        one_hot_encode_metadata=True,
        argmax_ppg=False,
        normalize_features=args.normalize_features,
        random_seed=args.random_seed,
    )

    section_shapes = np.array(get_section_shapes(data_shape=data_shape))
    lookup = build_feature_lookup(section_shapes)

    reg_estimator, reg_param_grid = pick_probe(probe_name=args.probe_name)
    clf_estimator, clf_param_grid = pick_probe(probe_name=args.classifier_probe_name)
    reg_estimator, reg_param_grid = _with_input_scaler(reg_estimator, reg_param_grid)
    clf_estimator, clf_param_grid = _with_input_scaler(clf_estimator, clf_param_grid)

    layer_labels = _get_layer_labels(args.select_layers, model_hidden_states)
    speaker_ids = [x[0].split("-")[0] for x in filename_timestamp]
    speaker_onehot = select_feature_groups(feature_sets, lookup, ["SpeakerID-OH"])
    speaker_labels = np.argmax(speaker_onehot, axis=1)
    phone_ppg = select_feature_groups(feature_sets, lookup, ["ppg_feature"])
    phone_labels = np.argmax(phone_ppg, axis=1)
    syntax_targets = {
        syntax_target: select_feature_groups(feature_sets, lookup, [syntax_target])
        for syntax_target in SYNTAX_COMPONENT_TARGETS
    }
    word_embedding = select_feature_groups(feature_sets, lookup, ["word_embedding"])
    syntax_feature = _build_syntax_feature_vector(feature_sets, lookup)

    context = {
        "args": args,
        "feature_sets": feature_sets,
        "model_hidden_states": model_hidden_states,
        "lookup": lookup,
        "layer_labels": layer_labels,
        "speaker_ids": speaker_ids,
        "speaker_labels": speaker_labels,
        "phone_labels": phone_labels,
        "syntax_targets": syntax_targets,
        "word_embedding": word_embedding,
        "syntax_feature": syntax_feature,
    }

    category_counts = {}
    for experiment in EXPERIMENTAL_CONFIG:
        category = experiment["category"]
        category_counts[category] = _run_category_experiment(
            args,
            experiment,
            context,
            reg_estimator,
            reg_param_grid,
            clf_estimator,
            clf_param_grid,
        )

    count_parts = [
        f"{category}={count}" for category, count in sorted(category_counts.items())
    ]
    logger.info("Completed new probe runs: %s", ", ".join(count_parts))


if __name__ == "__main__":
    main()
