import argparse
import csv
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_probe_data import get_section_shapes, load_data
from probe_runner import (
    build_feature_lookup,
    fit_probe,
    select_feature_groups,
    split_train_test,
)
from utils import FIGURES_ROOT, RESULTS_ROOT, pick_probe

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


CATEGORICAL_SYNTAX_TARGETS = {
    "syntax_POS_OH",
    "syntax_Dependency_Label_OH",
}

SYNTAX_COMPONENT_TARGETS = [
    "syntax_POS_OH",
    "syntax_Dependency_Label_OH",
    "syntax_Tree_Depth",
    "syntax_Word_Position",
    "syntax_Total_Tree_Depth",
    "syntax_Total_Word_Count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check overlap among probe input features."
    )
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="dev-clean",
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
    return parser.parse_args()


def _run_classification_probe(
    estimator,
    param_grid,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
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
        n_jobs=1,
    )
    best_model = grid.best_estimator_
    return {
        "train_score": float(best_model.score(x_train, y_train)),
        "test_score": float(best_model.score(x_test, y_test)),
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
    """Memory-friendly local regression probe for sanity checks."""

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
        n_jobs=1,
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


def _slugify(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_").lower()


def _build_step_savepath(
    args: argparse.Namespace,
    *,
    category: str,
    config_name: str,
) -> str:
    split_id = args.librispeech_split
    if args.random_seed != 42:
        split_id = f"{split_id}_seed{args.random_seed}"

    if category == "speaker_hidden":
        run_dir = f"{split_id}_{args.modelname.replace('/', '-')}"
    else:
        run_dir = split_id

    return os.path.join(
        RESULTS_ROOT,
        "input_overlap_sanity",
        category,
        run_dir,
        f"{_slugify(config_name)}.csv",
    )


def aggregate_saved_results(
    librispeech_split: str,
    random_seed: int = 42,
    save_combined: bool = True,
) -> dict[str, pd.DataFrame]:
    """Aggregate per-step CSV results into category-level DataFrames.

    This helper is intentionally not called from main(). It can be used after
    step-wise runs complete to produce compact combined CSVs per category.
    """

    split_id = librispeech_split
    if random_seed != 42:
        split_id = f"{split_id}_seed{random_seed}"

    syntax_dir = os.path.join(
        RESULTS_ROOT,
        "input_overlap_sanity",
        "syntax",
        split_id,
    )
    speaker_input_dir = os.path.join(
        RESULTS_ROOT,
        "input_overlap_sanity",
        "speaker_input",
        split_id,
    )
    speaker_hidden_root = os.path.join(
        RESULTS_ROOT,
        "input_overlap_sanity",
        "speaker_hidden",
    )

    category_files = {
        "syntax": sorted(glob.glob(os.path.join(syntax_dir, "*.csv"))),
        "speaker_input": sorted(glob.glob(os.path.join(speaker_input_dir, "*.csv"))),
        "speaker_hidden": sorted(
            glob.glob(os.path.join(speaker_hidden_root, f"{split_id}_*", "*.csv"))
        ),
    }

    category_save_dirs = {
        "syntax": syntax_dir,
        "speaker_input": speaker_input_dir,
        "speaker_hidden": speaker_hidden_root,
    }

    aggregated: dict[str, pd.DataFrame] = {}
    for category, files in category_files.items():
        if not files:
            logger.warning(
                "No step CSV files found for %s (split=%s, seed=%s)",
                category,
                librispeech_split,
                random_seed,
            )
            aggregated[category] = pd.DataFrame()
            continue

        category_frames = [pd.read_csv(path) for path in files]
        category_df = pd.concat(category_frames, ignore_index=True)
        aggregated[category] = category_df

        if save_combined:
            combined_path = os.path.join(
                category_save_dirs[category],
                f"combined_{category}_{split_id}.csv",
            )
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            category_df.to_csv(combined_path, index=False)
            logger.info("Saved combined %s CSV to %s", category, combined_path)

    all_frames = [df for df in aggregated.values() if not df.empty]
    aggregated["all"] = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )
    aggregated["all"].to_csv(
        os.path.join(
            FIGURES_ROOT,
            f"combined_all_sanity_{split_id}.csv",
        )
    )
    return aggregated


def _serialize_row(row: dict) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            serialized[key] = json.dumps(value)
        else:
            serialized[key] = str(value)
    return serialized


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
        "best_params",
        "n_samples",
    ]
    return preferred_order


def _ensure_csv_header(savepath: str) -> None:
    if os.path.isfile(savepath):
        return

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_get_csv_fieldnames())
        writer.writeheader()


def _append_result_row(savepath: str, row: dict) -> None:
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.isfile(savepath):
        os.remove(savepath)
    _ensure_csv_header(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_get_csv_fieldnames(), extrasaction="ignore"
        )
        writer.writerow(_serialize_row(row))


def main() -> None:
    args = parse_args()

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

    speaker_ids = [x[0].split("-")[0] for x in filename_timestamp]

    speaker_onehot = select_feature_groups(feature_sets, lookup, ["SpeakerID-OH"])
    speaker_labels = np.argmax(speaker_onehot, axis=1)

    new_syntax_count = 0
    new_speaker_input_count = 0
    new_speaker_hidden_count = 0

    # 1) Decode individual syntax components from word embeddings.
    X_word = select_feature_groups(feature_sets, lookup, ["word_embedding"])
    for syntax_target in SYNTAX_COMPONENT_TARGETS:
        Y = select_feature_groups(feature_sets, lookup, [syntax_target])
        config_name = f"word_embedding->{syntax_target}"
        savepath = _build_step_savepath(
            args,
            category="syntax",
            config_name=config_name,
        )
        if os.path.isfile(savepath) and not args.overwrite:
            logger.info("Skipping existing syntax step: %s", savepath)
            continue
        logger.info("Running syntax overlap probe: word_embedding -> %s", syntax_target)

        if syntax_target in CATEGORICAL_SYNTAX_TARGETS:
            y_labels = np.argmax(Y, axis=1)
            probe_result = _run_classification_probe(
                clf_estimator,
                clf_param_grid,
                X_word,
                y_labels,
            )
            metric = "accuracy"
        else:
            probe_result = _run_regression_probe(
                reg_estimator,
                reg_param_grid,
                X_word,
                Y,
                stratify_labels=speaker_ids,
            )
            metric = "r2"

        row = {
            "librispeech_split": args.librispeech_split,
            "modelname": args.modelname,
            "config_name": config_name,
            "x_groups": ["word_embedding"],
            "y_groups": [syntax_target],
            "metric": metric,
            "train_score": probe_result["train_score"],
            "test_score": probe_result["test_score"],
            "best_params": probe_result["best_params"],
            "n_samples": int(feature_sets.shape[0]),
        }
        _append_result_row(savepath, row)
        new_syntax_count += 1
        logger.info("Saved syntax step result to %s", savepath)

    # 2) Decode speaker identity from phonetics/acoustics.
    speaker_source_configs = [
        ["ppg_feature"],
        ["eGeMAPSv02"],
        ["ppg_feature", "eGeMAPSv02"],
    ]
    for source_groups in speaker_source_configs:
        config_name = f"{'+'.join(source_groups)}->SpeakerID"
        savepath = _build_step_savepath(
            args,
            category="speaker_input",
            config_name=config_name,
        )
        if os.path.isfile(savepath) and not args.overwrite:
            logger.info("Skipping existing speaker-input step: %s", savepath)
            continue
        logger.info(
            "Running speaker overlap probe: %s -> SpeakerID",
            "+".join(source_groups),
        )
        X_source = select_feature_groups(feature_sets, lookup, source_groups)
        probe_result = _run_classification_probe(
            clf_estimator,
            clf_param_grid,
            X_source,
            speaker_labels,
        )
        row = {
            "librispeech_split": args.librispeech_split,
            "modelname": args.modelname,
            "config_name": config_name,
            "x_groups": source_groups,
            "y_groups": ["SpeakerID"],
            "metric": "accuracy",
            "train_score": probe_result["train_score"],
            "test_score": probe_result["test_score"],
            "best_params": probe_result["best_params"],
            "n_samples": int(feature_sets.shape[0]),
        }
        _append_result_row(savepath, row)
        new_speaker_input_count += 1
        logger.info("Saved speaker-input step result to %s", savepath)

    # 3) Decode speaker identity from hidden states (per layer).
    layer_labels = (
        args.select_layers
        if args.select_layers is not None
        else list(range(model_hidden_states.shape[1]))
    )
    for layer_idx, layer_label in enumerate(layer_labels):
        config_name = f"hidden_state_L{layer_label}->SpeakerID"
        savepath = _build_step_savepath(
            args,
            category="speaker_hidden",
            config_name=config_name,
        )
        if os.path.isfile(savepath) and not args.overwrite:
            logger.info("Skipping existing speaker-hidden step: %s", savepath)
            continue
        logger.info("Running speaker-from-hidden-state probe at layer=%s", layer_label)
        X_hidden = model_hidden_states[:, layer_idx, :]
        probe_result = _run_classification_probe(
            clf_estimator,
            clf_param_grid,
            X_hidden,
            speaker_labels,
        )
        row = {
            "librispeech_split": args.librispeech_split,
            "modelname": args.modelname,
            "config_name": config_name,
            "x_groups": [f"hidden_state_L{layer_label}"],
            "y_groups": ["SpeakerID"],
            "metric": "accuracy",
            "layer": int(layer_label),
            "train_score": probe_result["train_score"],
            "test_score": probe_result["test_score"],
            "best_params": probe_result["best_params"],
            "n_samples": int(model_hidden_states.shape[0]),
        }
        _append_result_row(savepath, row)
        new_speaker_hidden_count += 1
        logger.info("Saved speaker-hidden step result to %s", savepath)

    logger.info(
        "Completed new probe runs: syntax=%s, speaker_input=%s, speaker_hidden=%s",
        new_syntax_count,
        new_speaker_input_count,
        new_speaker_hidden_count,
    )


if __name__ == "__main__":
    main()
