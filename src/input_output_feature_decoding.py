import argparse
import glob
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
from utils import RESULTS_ROOT, pick_probe

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
        description="decoding check overlap among probe input features."
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

    speaker_ids = [x[0].split("-")[0] for x in filename_timestamp]

    speaker_onehot = select_feature_groups(feature_sets, lookup, ["SpeakerID-OH"])
    speaker_labels = np.argmax(speaker_onehot, axis=1)

    new_syntax_count = 0
    new_speaker_input_count = 0
    new_speaker_hidden_count = 0
    new_phoneid_count = 0
    new_syntax_hidden_count = 0

    # 1) Decode individual syntax components from word embeddings.
    lexicon_syntax_savepath = _build_category_savepath(
        args, category="lexicon_syntax", input_features_only=True
    )
    if os.path.isfile(lexicon_syntax_savepath) and not args.overwrite:
        logger.info("Skipping lexicon_syntax category: %s", lexicon_syntax_savepath)
    else:
        lexicon_syntax_rows = []
        X_word = select_feature_groups(feature_sets, lookup, ["word_embedding"])
        for syntax_target in SYNTAX_COMPONENT_TARGETS:
            Y = select_feature_groups(feature_sets, lookup, [syntax_target])
            config_name = f"word_embedding->{syntax_target}"
            logger.info(
                "Running syntax overlap probe: word_embedding -> %s", syntax_target
            )

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
                "modelname": "input-feature",
                "config_name": config_name,
                "x_groups": ["word_embedding"],
                "y_groups": [syntax_target],
                "metric": metric,
                "train_score": probe_result["train_score"],
                "test_score": probe_result["test_score"],
                "best_params": probe_result["best_params"],
                "n_samples": int(feature_sets.shape[0]),
            }
            lexicon_syntax_rows.append(row)
            new_syntax_count += 1

        _save_category_results(lexicon_syntax_savepath, lexicon_syntax_rows)
        logger.info("Saved lexicon_syntax results to %s", lexicon_syntax_savepath)

    # 2) Decode speaker identity from phonetics/acoustics.
    input_speaker_savepath = _build_category_savepath(
        args, category="input_speaker", input_features_only=True
    )
    if os.path.isfile(input_speaker_savepath) and not args.overwrite:
        logger.info("Skipping input_speaker category: %s", input_speaker_savepath)
    else:
        input_speaker_rows = []
        speaker_source_configs = [
            ["ppg_feature"],
            ["eGeMAPSv02"],
            ["ppg_feature", "eGeMAPSv02"],
        ]
        for source_groups in speaker_source_configs:
            config_name = f"{'+'.join(source_groups)}->SpeakerID"
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
                "modelname": "input-feature",
                "config_name": config_name,
                "x_groups": source_groups,
                "y_groups": ["SpeakerID"],
                "metric": "accuracy",
                "train_score": probe_result["train_score"],
                "test_score": probe_result["test_score"],
                "best_params": probe_result["best_params"],
                "n_samples": int(feature_sets.shape[0]),
            }
            input_speaker_rows.append(row)
            new_speaker_input_count += 1

        _save_category_results(input_speaker_savepath, input_speaker_rows)
        logger.info("Saved input_speaker results to %s", input_speaker_savepath)

    # 3) Decode speaker identity from hidden states (per layer).
    hidden_speaker_savepath = _build_category_savepath(args, category="hidden_speaker")
    if os.path.isfile(hidden_speaker_savepath) and not args.overwrite:
        logger.info("Skipping hidden_speaker category: %s", hidden_speaker_savepath)
    else:
        hidden_speaker_rows = []
        layer_labels = (
            args.select_layers
            if args.select_layers is not None
            else list(range(model_hidden_states.shape[1]))
        )
        for layer_idx, layer_label in enumerate(layer_labels):
            config_name = f"hidden_state_L{layer_label}->SpeakerID"
            logger.info(
                "Running speaker-from-hidden-state probe at layer=%s", layer_label
            )
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
            hidden_speaker_rows.append(row)
            new_speaker_hidden_count += 1

        _save_category_results(hidden_speaker_savepath, hidden_speaker_rows)
        logger.info("Saved hidden_speaker results to %s", hidden_speaker_savepath)

    # 4) decode phone ID from hidden states (per layer) as a decoding check that the probe can find known information.
    hidden_phoneid_savepath = _build_category_savepath(args, category="hidden_phoneid")
    if os.path.isfile(hidden_phoneid_savepath) and not args.overwrite:
        logger.info("Skipping hidden_phoneid category: %s", hidden_phoneid_savepath)
    else:
        hidden_phoneid_rows = []
        phone_PPG = select_feature_groups(feature_sets, lookup, ["ppg_feature"])
        phone_labels = np.argmax(phone_PPG, axis=1)
        layer_labels = (
            args.select_layers
            if args.select_layers is not None
            else list(range(model_hidden_states.shape[1]))
        )
        for layer_idx, layer_label in enumerate(layer_labels):
            config_name = f"hidden_state_L{layer_label}->PhoneID"
            logger.info(
                "Running decoding-check probe: hidden_state_L%s -> PhoneID", layer_label
            )
            X_hidden = model_hidden_states[:, layer_idx, :]
            probe_result = _run_classification_probe(
                clf_estimator,
                clf_param_grid,
                X_hidden,
                phone_labels,
            )
            row = {
                "librispeech_split": args.librispeech_split,
                "modelname": args.modelname,
                "config_name": config_name,
                "x_groups": [f"hidden_state_L{layer_label}"],
                "y_groups": ["PhoneID"],
                "metric": "accuracy",
                "layer": int(layer_label),
                "train_score": probe_result["train_score"],
                "test_score": probe_result["test_score"],
                "best_params": probe_result["best_params"],
                "n_samples": int(model_hidden_states.shape[0]),
            }
            hidden_phoneid_rows.append(row)
            new_phoneid_count += 1

        _save_category_results(hidden_phoneid_savepath, hidden_phoneid_rows)
        logger.info("Saved hidden_phoneid results to %s", hidden_phoneid_savepath)

    # 5) decode syntax components from hidden states (per layer) as a decoding check that the probe can find known information.
    hidden_syntax_savepath = _build_category_savepath(args, category="hidden_syntax")
    if os.path.isfile(hidden_syntax_savepath) and not args.overwrite:
        logger.info("Skipping hidden_syntax category: %s", hidden_syntax_savepath)
    else:
        hidden_syntax_rows = []
        layer_labels = (
            args.select_layers
            if args.select_layers is not None
            else list(range(model_hidden_states.shape[1]))
        )
        for syntax_target in SYNTAX_COMPONENT_TARGETS:
            Y = select_feature_groups(feature_sets, lookup, [syntax_target])
            for layer_idx, layer_label in enumerate(layer_labels):
                config_name = f"hidden_state_L{layer_label}->{syntax_target}"
                logger.info(
                    "Running syntax-check probe: hidden_state_L%s -> %s",
                    layer_label,
                    syntax_target,
                )

                X_hidden = model_hidden_states[:, layer_idx, :]
                if syntax_target in CATEGORICAL_SYNTAX_TARGETS:
                    y_labels = np.argmax(Y, axis=1)
                    probe_result = _run_classification_probe(
                        clf_estimator,
                        clf_param_grid,
                        X_hidden,
                        y_labels,
                    )
                    metric = "accuracy"
                else:
                    probe_result = _run_regression_probe(
                        reg_estimator,
                        reg_param_grid,
                        X_hidden,
                        Y,
                        stratify_labels=speaker_ids,
                    )
                    metric = "r2"

                row = {
                    "librispeech_split": args.librispeech_split,
                    "modelname": args.modelname,
                    "config_name": config_name,
                    "x_groups": [f"hidden_state_L{layer_label}"],
                    "y_groups": [syntax_target],
                    "metric": metric,
                    "layer": int(layer_label),
                    "train_score": probe_result["train_score"],
                    "test_score": probe_result["test_score"],
                    "best_params": probe_result["best_params"],
                    "n_samples": int(model_hidden_states.shape[0]),
                }
                hidden_syntax_rows.append(row)
                new_syntax_hidden_count += 1

        _save_category_results(hidden_syntax_savepath, hidden_syntax_rows)
        logger.info("Saved hidden_syntax results to %s", hidden_syntax_savepath)

    # 6) Add the majority baseline for all the classification probes to a separate csv file for easier plotting later, since the baseline is the same across all layers and models for a given target.
    baseline_savepath = _build_category_savepath(
        args, category="classification_baselines"
    )
    if os.path.isfile(baseline_savepath) and not args.overwrite:
        logger.info("Skipping classification_baselines category: %s", baseline_savepath)
    else:
        baseline_rows = []
        # Syntax baseline
        for syntax_target in CATEGORICAL_SYNTAX_TARGETS:
            Y = select_feature_groups(feature_sets, lookup, [syntax_target])
            y_labels = np.argmax(Y, axis=1)
            unique_labels, label_counts = np.unique(y_labels, return_counts=True)
            majority_class_count = label_counts.max()
            majority_class_proportion = majority_class_count / len(y_labels)
            row = {
                "librispeech_split": args.librispeech_split,
                "modelname": "majority-baseline",
                "config_name": f"MajorityClass->{syntax_target}",
                "x_groups": ["none"],
                "y_groups": [syntax_target],
                "metric": "accuracy",
                "train_score": majority_class_proportion,
                "test_score": majority_class_proportion,
                "best_params": {},
                "n_samples": int(feature_sets.shape[0]),
            }
            baseline_rows.append(row)

        # SpeakerID baseline
        unique_speakers, speaker_counts = np.unique(speaker_labels, return_counts=True)
        majority_speaker_count = speaker_counts.max()
        majority_speaker_proportion = majority_speaker_count / len(speaker_labels)
        row = {
            "librispeech_split": args.librispeech_split,
            "modelname": "majority-baseline",
            "config_name": "MajorityClass->SpeakerID",
            "x_groups": ["none"],
            "y_groups": ["SpeakerID"],
            "metric": "accuracy",
            "train_score": majority_speaker_proportion,
            "test_score": majority_speaker_proportion,
            "best_params": {},
            "n_samples": int(feature_sets.shape[0]),
        }
        baseline_rows.append(row)

        _save_category_results(baseline_savepath, baseline_rows)
        logger.info("Saved classification baselines to %s", baseline_savepath)

    logger.info(
        "Completed new probe runs: syntax=%s, speaker_input=%s, speaker_hidden=%s, phoneid=%s, syntax_hidden=%s",
        new_syntax_count,
        new_speaker_input_count,
        new_speaker_hidden_count,
        new_phoneid_count,
        new_syntax_hidden_count,
    )


if __name__ == "__main__":
    main()
