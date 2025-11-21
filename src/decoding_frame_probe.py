import argparse
import json
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
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


# Syntax feature indices mapping
syntax_feature_idx = {
    "POS": 0,
    "Dependency_Label": 1,
    "Constituent_Label": 2,
    "Tree_Depth": 3,
    "Tree_Depth_Normed": 4,
    "Word_Position": 5,
    "Word_Position_Normed": 6,
    "Path_from_Root": (7, -1),
}


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
    # Use speakerID to stratify the train-test-split to avoid data leakage
    speakerID = [x[0].split("-")[0] for x in filename_timestamp]

    X_train, X_test, y_train, y_test, train_filenames, test_filenames = (
        train_test_split(
            model_hidden_states,
            feature_sets,
            filename_timestamp,
            test_size=0.2,
            random_state=42,
            stratify=speakerID,
        )
    )
    train_speakerID = [x[0].split("-")[0] for x in train_filenames]
    test_speakerID = [x[0].split("-")[0] for x in test_filenames]

    section_shapes = np.array(get_section_shapes(data_shape=data_shape))

    for start_idx, end_idx, name in tqdm(section_shapes, desc="Feature Groups"):
        start_idx = int(start_idx)
        if name == "syntax_feature":
            end_idx = start_idx + 7  # Exclude Path_from_Root for now
        else:
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
            raw_r2 = r2_score(
                actual_y_test, np.zeros_like(actual_y_test), multioutput="raw_values"
            )
            layer_result = {
                "feature_group": name,
                "layer": layer,
                "probe": probe_name,
                "r2_score": r2,
                "raw_r2": raw_r2,
            }
            results.append(layer_result)

    return results


def syntax_deep_dive(
    model_hidden_states,
    feature_sets,
    data_shape,
    filename_timestamp,
    probe_name="ridge_classifier",
):
    syntax_results = []

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

    # Only get the syntax feature group
    syntax_shape = [x for x in section_shapes if x[2] == "syntax_feature"][0]

    start_idx = int(syntax_shape[0])
    end_idx = int(syntax_shape[1])

    actual_y_train = y_train[:, start_idx:end_idx]
    actual_y_test = y_test[:, start_idx:end_idx]

    # Analyze syntax features in more detail
    # For syntax features, we further split the features and skip the Path_from_Root feature
    for feature_name, idx in syntax_feature_idx.items():
        if feature_name in [
            "Path_from_Root",
            "Word_Position_Normed",
            "Tree_Depth_Normed",
        ]:
            continue
        if isinstance(idx, tuple):
            continue
        else:
            train_feat = actual_y_train[:, idx]
            test_feat = actual_y_test[:, idx]

        GS = GridSearchCV(
            pick_probe(probe_name)[0],
            pick_probe(probe_name)[1],
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        for layer in tqdm(
            range(model_hidden_states.shape[1]),
            desc=f"Layers - {feature_name}",
            leave=False,
        ):
            X_train_layer = X_train[:, layer, :]
            X_test_layer = X_test[:, layer, :]
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_layer = scaler.fit_transform(X_train_layer)
            X_test_layer = scaler.transform(X_test_layer)

            # Turn train_feat and test_feat into class labels
            train_feat = train_feat.astype(int)
            test_feat = test_feat.astype(int)

            GS.fit(X_train_layer, train_feat)
            best_model = GS.best_estimator_
            predictions = best_model.predict(X_test_layer)
            # r2 = r2_score(test_feat, predictions, multioutput="variance_weighted")
            # raw_r2 = r2_score(
            #     test_feat, np.zeros_like(test_feat), multioutput="raw_values"
            # )
            accuracy = np.mean(predictions == test_feat)
            layer_result = {
                "feature_group": f"{feature_name}",
                "layer": layer,
                "probe": probe_name,
                "accuracy": accuracy,
            }
            syntax_results.append(layer_result)
    # Add baseline accuracy for each syntax feature based on the most frequent class
    for feature_name, idx in syntax_feature_idx.items():
        if feature_name in [
            "Path_from_Root",
            "Word_Position_Normed",
            "Tree_Depth_Normed",
        ]:
            continue
        if isinstance(idx, tuple):
            continue
        else:
            test_feat = actual_y_test[:, idx]
        test_feat = test_feat.astype(int)
        # Calculate the most frequent class accuracy
        label, count = np.unique(test_feat, return_counts=True)
        most_frequent_class = label[np.argmax(count)]
        baseline_accuracy = np.mean(test_feat == most_frequent_class)
        layer_result = {
            "feature_group": f"{feature_name}_baseline",
            "layer": -1,
            "probe": probe_name,
            "accuracy": baseline_accuracy,
        }
        syntax_results.append(layer_result)
    return syntax_results


def main():
    librispeech_split = "dev-clean"
    librispeech_split = "train-clean-100"
    # modelname = "wav2vec2-base"
    modelname = "bert-base-uncased"
    select_layers = None
    normalize_features = False
    overwrite = False
    probe_name = "ridge"
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        reduce_dnn_word_embedding=True,
        one_hot_encode_syntax=True,
        normalize_features=normalize_features,
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
    rename_feature_group = {
        "OtherAcoustic": "Other Acoustic Features",
        "SpectralInfo": "Spectral Features",
        "Formants": "Formants",
        "syntax_feature": "Syntactic Features",
        "ppg_feature": "PPG",
        "spk_embedding": "Speaker Embedding",
        "dnn_word_embedding": "DNN Word Embedding",
    }

    # plot the results
    results_df["feature_group"] = results_df["feature_group"].map(rename_feature_group)

    all_results_plot = (
        p9.ggplot(
            results_df,
            p9.aes(
                x="layer", y="r2_score", color="feature_group", shape="feature_group"
            ),
        )
        + p9.geom_line()
        + p9.geom_point()
        + p9.ggtitle(
            f"Decoding Frame Probe Results for {modelname} on {librispeech_split}"
        )
        + p9.xlab("Layer")
        + p9.ylab("R2 Score")
        + p9.theme(legend_position="right")
        + p9.scale_color_discrete(name="Feature Group")
        + p9.scale_shape_discrete(name="Feature Group")
        + p9.theme(figure_size=(10, 6), dpi=300)
    )
    # Save the plot
    all_results_plot.save(
        filename=f"{RESULTS_ROOT}/figures/decoding_frame_probe_results_{modelname}_{librispeech_split}.png",
        bbox_inches="tight",
    )

    results_df.to_csv(
        f"{RESULTS_ROOT}/decoding_frame_probe_results_{modelname}_{librispeech_split}.csv",
        index=False,
    )

    # Switch feature loading
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        reduce_dnn_word_embedding=False,
        one_hot_encode_syntax=False,
        normalize_features=normalize_features,
    )

    syntax_results = syntax_deep_dive(
        model_hidden_states=model_hidden_states,
        feature_sets=feature_sets,
        data_shape=data_shape,
        filename_timestamp=filename_timestamp,
        probe_name=probe_name,
    )

    # plot syntax deep dive results
    syntax_results_df = pd.DataFrame(syntax_results)

    syntax_results_baseline = syntax_results_df[
        syntax_results_df["layer"] == -1
    ].reset_index(drop=True)
    # Rename feature groups for better plotting
    syntax_results_baseline["feature_group"] = syntax_results_baseline[
        "feature_group"
    ].str.replace("_baseline", "")
    syntax_results_df = syntax_results_df[syntax_results_df["layer"] != -1]

    syntax_results_plot = (
        p9.ggplot(
            syntax_results_df,
            p9.aes(
                x="layer", y="accuracy", color="feature_group", shape="feature_group"
            ),
        )
        + p9.geom_line()
        + p9.geom_point()
        + p9.ggtitle(
            f"Decoding Frame Probe Syntax Deep Dive Results for {modelname} on {librispeech_split}"
        )
        + p9.xlab("Layer")
        + p9.ylab("Probing Accuracy")
        + p9.theme(legend_position="right")
        + p9.scale_color_discrete(name="Syntax Feature")
        + p9.scale_shape_discrete(name="Syntax Feature")
        + p9.theme(figure_size=(10, 6), dpi=300)
        + p9.geom_hline(
            p9.aes(
                yintercept="accuracy",
                color="feature_group",
            ),
            data=syntax_results_baseline,
            show_legend=True,
            linetype="dashed",
        )
    )
    # Save the plot
    syntax_results_plot.save(
        filename=f"{RESULTS_ROOT}/figures/decoding_frame_probe_syntax_deep_dive_{modelname}_{librispeech_split}.png",
        bbox_inches="tight",
    )

    syntax_results_df.to_csv(
        f"{RESULTS_ROOT}/decoding_frame_probe_syntax_deep_dive_{modelname}_{librispeech_split}.csv",
        index=False,
    )

    all_results_plot.show()
    syntax_results_plot.show()


if __name__ == "__main__":
    main()
