"""Run and plot PPG representation comparison experiments.

This script is an all-in-one entry point that:
1) loads probe-ready data,
2) runs encoding probes for different PPG representations,
3) runs decoding probes for the same representations,
4) saves tabular results under a dedicated subdirectory in ``results/``,
5) generates comparison figures including the legacy-style PPG figure.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from parse_results import (
    plot_decoding_representation,
    plot_encode_decode_comparison,
    plot_encoding_legacy_style,
)
from probe_runner import evaluate_probe, fit_probe, split_train_test
from utils import RESULTS_ROOT, pick_probe


class ExperimentInputs:
    def __init__(
        self,
        representations: dict[str, np.ndarray],
        model_hidden_states: np.ndarray,
        speaker_ids: list[str],
    ) -> None:
        self.representations = representations
        self.model_hidden_states = model_hidden_states
        self.speaker_ids = speaker_ids


def _slug_modelname(modelname: str) -> str:
    return modelname.split("/")[-1]


def build_output_dir(split: str, modelname: str, probe_name: str) -> str:
    model_slug = _slug_modelname(modelname)
    outdir = os.path.join(
        RESULTS_ROOT,
        "representation_comparison",
        f"{split}_{model_slug}_{probe_name}",
    )
    os.makedirs(outdir, exist_ok=True)
    return outdir


def load_experiment_inputs(
    librispeech_split: str,
    modelname: str,
    random_seed: int,
    overwrite: bool,
    select_layers: list[int] | None,
) -> ExperimentInputs:
    from load_probe_data import load_data

    feature_sets, model_hidden_states, filename_timestamp, _ = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        selected_input_components=["ppg_feature"],
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        normalize_features=False,
        one_hot_encode_syntax=False,
        one_hot_encode_syntax_separate=False,
        one_hot_encode_metadata=False,
        argmax_ppg=False,
        reduce_dnn_word_embedding=False,
        random_seed=random_seed,
    )

    ppg_feature = feature_sets
    ppg_id = np.argmax(ppg_feature, axis=1).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    ppg_feature_onehot = encoder.fit_transform(ppg_id)

    speaker_ids = [x[0].split("-")[0] for x in filename_timestamp]
    representations = {
        "ppg_feature": ppg_feature,
        "ppg_feature_onehot": ppg_feature_onehot,
    }

    return ExperimentInputs(
        representations=representations,
        model_hidden_states=model_hidden_states,
        speaker_ids=speaker_ids,
    )


def _fit_regression_layer(
    X: np.ndarray,
    y: np.ndarray,
    speaker_ids: list[str],
    probe_name: str,
    random_seed: int,
) -> tuple[float, float, dict]:
    x_train, x_test, y_train, y_test = split_train_test(
        X,
        y,
        stratify_labels=speaker_ids,
        random_state=random_seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    estimator, param_grid = pick_probe(probe_name)
    grid = fit_probe(estimator, param_grid, x_train, y_train, scoring="r2")
    train_score, test_score = evaluate_probe(
        grid.best_estimator_, x_train, y_train, x_test, y_test
    )
    return train_score, test_score, grid.best_params_


def _fit_classification_layer(
    X: np.ndarray,
    y: np.ndarray,
    speaker_ids: list[str],
    random_seed: int,
) -> tuple[float, float, dict]:
    x_train, x_test, y_train, y_test = split_train_test(
        X,
        y,
        stratify_labels=speaker_ids,
        random_state=random_seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    estimator, param_grid = pick_probe("ridge_classifier")
    grid = fit_probe(
        estimator,
        param_grid,
        x_train,
        y_train,
        scoring="accuracy",
    )
    y_pred = grid.best_estimator_.predict(x_test)
    accuracy = float(np.mean(y_pred == y_test))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    return accuracy, f1_weighted, grid.best_params_


def run_encoding_experiment(
    inputs: ExperimentInputs,
    modelname: str,
    librispeech_split: str,
    probe_name: str,
    random_seed: int,
) -> pd.DataFrame:
    records: list[dict] = []
    n_layers = inputs.model_hidden_states.shape[1]
    for rep_name, rep_values in inputs.representations.items():
        for layer in tqdm(range(n_layers), desc=f"Encoding {rep_name}"):
            y_layer = inputs.model_hidden_states[:, layer, :]
            train_score, test_score, best_params = _fit_regression_layer(
                X=rep_values,
                y=y_layer,
                speaker_ids=inputs.speaker_ids,
                probe_name=probe_name,
                random_seed=random_seed,
            )
            records.append(
                {
                    "direction": "encoding",
                    "representation": rep_name,
                    "layer": layer,
                    "metric_name": "r2",
                    "train_score": train_score,
                    "test_score": test_score,
                    "modelname": _slug_modelname(modelname),
                    "librispeech_split": librispeech_split,
                    "probe_name": probe_name,
                    "best_params": json.dumps(best_params),
                }
            )
    return pd.DataFrame(records)


def run_decoding_experiment(
    inputs: ExperimentInputs,
    modelname: str,
    librispeech_split: str,
    probe_name: str,
    random_seed: int,
) -> pd.DataFrame:
    records: list[dict] = []
    n_layers = inputs.model_hidden_states.shape[1]

    for rep_name, rep_values in inputs.representations.items():
        for layer in tqdm(range(n_layers), desc=f"Decoding {rep_name}"):
            x_layer = inputs.model_hidden_states[:, layer, :]
            if rep_name == "ppg_ID":
                acc, f1_weighted, best_params = _fit_classification_layer(
                    X=x_layer,
                    y=rep_values.ravel(),
                    speaker_ids=inputs.speaker_ids,
                    random_seed=random_seed,
                )
                records.append(
                    {
                        "direction": "decoding",
                        "representation": rep_name,
                        "layer": layer,
                        "metric_name": "accuracy",
                        "train_score": np.nan,
                        "test_score": acc,
                        "aux_metric_name": "f1_weighted",
                        "aux_metric_value": f1_weighted,
                        "modelname": _slug_modelname(modelname),
                        "librispeech_split": librispeech_split,
                        "probe_name": "ridge_classifier",
                        "best_params": json.dumps(best_params),
                    }
                )
            else:
                train_score, test_score, best_params = _fit_regression_layer(
                    X=x_layer,
                    y=rep_values,
                    speaker_ids=inputs.speaker_ids,
                    probe_name=probe_name,
                    random_seed=random_seed,
                )
                records.append(
                    {
                        "direction": "decoding",
                        "representation": rep_name,
                        "layer": layer,
                        "metric_name": "r2",
                        "train_score": train_score,
                        "test_score": test_score,
                        "modelname": _slug_modelname(modelname),
                        "librispeech_split": librispeech_split,
                        "probe_name": probe_name,
                        "best_params": json.dumps(best_params),
                    }
                )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_split", type=str, default="train-clean-100")
    parser.add_argument("--modelname", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--probe_name", type=str, default="ridge")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--select_layers", nargs="+", type=int, default=None)
    parser.add_argument("--skip_experiments", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = build_output_dir(
        split=args.librispeech_split,
        modelname=args.modelname,
        probe_name=args.probe_name,
    )

    encoding_path = os.path.join(outdir, "encoding_ppg_representation_scores.csv")
    decoding_path = os.path.join(outdir, "decoding_ppg_representation_scores.csv")

    if (
        args.skip_experiments
        and os.path.exists(encoding_path)
        and os.path.exists(decoding_path)
    ):
        encoding_df = pd.read_csv(encoding_path)
        decoding_df = pd.read_csv(decoding_path)
    else:
        inputs = load_experiment_inputs(
            librispeech_split=args.librispeech_split,
            modelname=args.modelname,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
            select_layers=args.select_layers,
        )

        encoding_df = run_encoding_experiment(
            inputs=inputs,
            modelname=args.modelname,
            librispeech_split=args.librispeech_split,
            probe_name=args.probe_name,
            random_seed=args.random_seed,
        )
        decoding_df = run_decoding_experiment(
            inputs=inputs,
            modelname=args.modelname,
            librispeech_split=args.librispeech_split,
            probe_name=args.probe_name,
            random_seed=args.random_seed,
        )

        encoding_df.to_csv(encoding_path, index=False)
        decoding_df.to_csv(decoding_path, index=False)

        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(
                {
                    "librispeech_split": args.librispeech_split,
                    "modelname": args.modelname,
                    "probe_name": args.probe_name,
                    "random_seed": args.random_seed,
                    "select_layers": args.select_layers,
                },
                f,
                indent=2,
            )

    combined_df = pd.concat([encoding_df, decoding_df], ignore_index=True)
    combined_df.to_csv(
        os.path.join(outdir, "combined_ppg_representation_scores.csv"), index=False
    )

    plot_encoding_legacy_style(
        encoding_df=encoding_df,
        modelname=args.modelname,
        librispeech_split=args.librispeech_split,
        outdir=outdir,
    )
    plot_decoding_representation(
        decoding_df=decoding_df,
        modelname=args.modelname,
        librispeech_split=args.librispeech_split,
        outdir=outdir,
    )
    plot_encode_decode_comparison(
        combined_df=combined_df,
        modelname=args.modelname,
        librispeech_split=args.librispeech_split,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()
