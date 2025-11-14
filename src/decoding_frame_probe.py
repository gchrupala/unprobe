import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Import r2 score for regression evaluation
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm, trange

from frame_probe import pick_probe
from load_probe_data import get_section_shapes, load_data

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Get the hostname of the machine running the code
hostname = os.uname().nodename
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# Setting up environmental variables depending on the cluster this code is running on

if "snellius" in hostname:
    # If running on Snellius, use the Snellius dataset root
    DATASET_ROOT = os.path.realpath("/projects/prjs1586/corpora/LibriSpeech")
    ALIGNMENT_ROOT = DATASET_ROOT.replace("LibriSpeech", "librispeech_textgrids")
    SAVEPATH = "/projects/prjs1586/experimental_data"
    RESULTS_ROOT = "/projects/prjs1586/experimental_results"

else:
    # If running on local machine, use the local dataset root
    DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENT_ROOT = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENT_ROOT = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def run_probe(
    model_hidden_states: np.ndarray,
    feature_sets: np.ndarray,
    data_shape: dict,
    filename_timestamp: list[tuple],
    probe_name: str = "ridge",
) -> list[dict]:
    """Run decoding frame probe on processed data.

    Args:
        model_hidden_states (np.ndarray): Processed input features.
        feature_sets (np.ndarray): Processed target features.
        data_shape (dict): Shape of the data.
        filename_timestamp (list[tuple]): List of filename and timestamp tuples.
        feature_groups (list[tuple]): List of feature group tuples.
        probe_name (str, optional): Name of the probe. Defaults to "ridge".
        select_layers (list[int] | None, optional): Layers to select. Defaults to None.
        zeroing (bool, optional): Whether to perform zeroing ablation. Defaults to True.
        ablation (bool, optional): Whether to perform ablation study. Defaults to True.
        permutation (bool, optional): Whether to perform permutation test. Defaults to True.
        results_path (str, optional): Path to save results. Defaults to RESULTS_ROOT.
        save_predictions (bool, optional): Whether to save predictions. Defaults to False.
        dim_reduction (int | bool | None, optional): Dimensionality reduction parameter. Defaults to False.

    Returns:
        list[dict]: List of results dictionaries.
    """
    results = []

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, train_filenames, test_filenames = (
        train_test_split(
            model_hidden_states,
            feature_sets,
            filename_timestamp,
            test_size=0.2,
            random_state=42,
        )
    )

    section_shapes = np.array(get_section_shapes(data_shape=data_shape))

    for start_idx, end_idx, name in tqdm(section_shapes, desc="Feature Groups"):
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        actual_y_train = y_train[:, start_idx:end_idx]
        actual_y_test = y_test[:, start_idx:end_idx]

        GS = GridSearchCV(
            pick_probe(probe_name)[0],
            pick_probe(probe_name)[1],
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        for layer in tqdm(
            range(model_hidden_states.shape[1]), desc="Layers", leave=False
        ):
            X_train_layer = X_train[:, layer, :]
            X_test_layer = X_test[:, layer, :]
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_layer = scaler.fit_transform(X_train_layer)
            X_test_layer = scaler.transform(X_test_layer)

            GS.fit(X_train_layer, actual_y_train)
            best_model = GS.best_estimator_
            predictions = best_model.predict(X_test_layer)
            r2 = r2_score(actual_y_test, predictions, multioutput="variance_weighted")
            layer_result = {
                "feature_group": name,
                "layer": layer,
                "probe": probe_name,
                "r2_score": r2,
            }
            results.append(layer_result)

    return results


def main():
    librispeech_split = "dev-clean"
    modelname = "wav2vec2-base"
    select_layers = None
    normalize_features = True
    overwrite = False
    probe_name = "ridge"
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        normalize_features=normalize_features,
        overwrite=overwrite,
    )

    # Check the variance of model_hidden_states and feature_sets
    logger.info(f"Variance of model hidden states: {np.var(model_hidden_states)}")
    logger.info(f"Variance of feature sets: {np.var(feature_sets)}")

    results = run_probe(
        model_hidden_states=model_hidden_states,
        feature_sets=feature_sets,
        data_shape=data_shape,
        filename_timestamp=filename_timestamp,
        probe_name=probe_name,
    )

    results_df = pd.DataFrame(results)
    # plot the results
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df, x="layer", y="r2_score", hue="feature_group", marker="o"
    )
    plt.title(f"Decoding Frame Probe Results for {modelname} on {librispeech_split}")
    plt.xlabel("Layer")
    plt.ylabel("R2 Score")
    plt.legend(title="Feature Group")
    plt.grid()
    plt.show()
