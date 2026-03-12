import argparse
import logging
import os
import pickle
import sys

import numpy as np

from load_probe_data import get_section_shapes, load_data
from probe_runner import (
    build_feature_lookup,
    fit_probe,
    run_standard_probe,
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
    grid = fit_probe(
        estimator,
        param_grid,
        x_train,
        y_train,
        scoring="accuracy",
    )
    best_model = grid.best_estimator_
    return {
        "train_score": float(best_model.score(x_train, y_train)),
        "test_score": float(best_model.score(x_test, y_test)),
        "best_params": grid.best_params_,
    }


def _build_savepath(args: argparse.Namespace) -> str:
    savepath = os.path.join(
        RESULTS_ROOT,
        "input_overlap_sanity",
        f"{args.librispeech_split}_{args.modelname.replace('/', '-')}_results.pkl",
    )
    if args.random_seed != 42:
        savepath = savepath.replace(
            "results.pkl", f"results-seed{args.random_seed}.pkl"
        )
    return savepath


def main() -> None:
    args = parse_args()

    selected_input_components = [
        "word_embedding",
        "syntax_feature",
        "ppg_feature",
        "eGeMAPSv02",
        "metadata",
    ]

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

    speaker_ids = [x[0].split("-")[0] for x in filename_timestamp]

    speaker_onehot = select_feature_groups(feature_sets, lookup, ["SpeakerID-OH"])
    speaker_labels = np.argmax(speaker_onehot, axis=1)

    results: list[dict] = []

    # 1) Decode individual syntax components from word embeddings.
    X_word = select_feature_groups(feature_sets, lookup, ["word_embedding"])
    for syntax_target in SYNTAX_COMPONENT_TARGETS:
        Y = select_feature_groups(feature_sets, lookup, [syntax_target])
        config_name = f"word_embedding->{syntax_target}"

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
            probe_result = run_standard_probe(
                reg_estimator,
                reg_param_grid,
                X_word,
                Y,
                stratify_labels=speaker_ids,
            )
            metric = "r2"

        results.append(
            {
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
        )

    # 2) Decode speaker identity from phonetics/acoustics.
    speaker_source_configs = [
        ["ppg_feature"],
        ["eGeMAPSv02"],
        ["ppg_feature", "eGeMAPSv02"],
    ]
    for source_groups in speaker_source_configs:
        X_source = select_feature_groups(feature_sets, lookup, source_groups)
        probe_result = _run_classification_probe(
            clf_estimator,
            clf_param_grid,
            X_source,
            speaker_labels,
        )
        results.append(
            {
                "librispeech_split": args.librispeech_split,
                "modelname": args.modelname,
                "config_name": f"{'+'.join(source_groups)}->SpeakerID",
                "x_groups": source_groups,
                "y_groups": ["SpeakerID"],
                "metric": "accuracy",
                "train_score": probe_result["train_score"],
                "test_score": probe_result["test_score"],
                "best_params": probe_result["best_params"],
                "n_samples": int(feature_sets.shape[0]),
            }
        )

    # 3) Decode speaker identity from hidden states (per layer).
    layer_labels = (
        args.select_layers
        if args.select_layers is not None
        else list(range(model_hidden_states.shape[1]))
    )
    for layer_idx, layer_label in enumerate(layer_labels):
        X_hidden = model_hidden_states[:, layer_idx, :]
        probe_result = _run_classification_probe(
            clf_estimator,
            clf_param_grid,
            X_hidden,
            speaker_labels,
        )
        results.append(
            {
                "librispeech_split": args.librispeech_split,
                "modelname": args.modelname,
                "config_name": f"hidden_state_L{layer_label}->SpeakerID",
                "x_groups": [f"hidden_state_L{layer_label}"],
                "y_groups": ["SpeakerID"],
                "metric": "accuracy",
                "layer": int(layer_label),
                "train_score": probe_result["train_score"],
                "test_score": probe_result["test_score"],
                "best_params": probe_result["best_params"],
                "n_samples": int(model_hidden_states.shape[0]),
            }
        )

    savepath = _build_savepath(args)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "wb") as f:
        pickle.dump(results, f)

    logger.info("Saved overlap sanity-check results to %s", savepath)
    logger.info("Completed %s probe runs", len(results))


if __name__ == "__main__":
    main()
