# This script runs target experiments using the encoding probe paradigm
# to investigate how well different feature groups (syntax, lexical) interact in the
# hidden representations of a DNN model pre-trained with audio data.

import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm, trange

from frame_probe import parse_args, pick_probe, r2_score
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


def main():
    args = parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname.split("/")[-1]
    select_layers = args.select_layers
    normalize_features = args.normalize_features
    overwrite = args.overwrite
    probe_name = args.probe_name
    syntax_deep_dive_flag = args.syntax_deep_dive

    # 1. We load the data for the experiment
    logger.info("Loading data for target experiment...")
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        reduce_dnn_word_embedding=True,
        one_hot_encode_syntax=True,
        one_hot_encode_metadata=True,
        argmax_ppg=True,
        normalize_features=normalize_features,
    )

    section_shapes = get_section_shapes(data_shape)
    all_feature_names = section_shapes[:, -1].tolist()

    # 2. We pick the probe for the experiment
    logger.info("Picking probe for target experiment...")
    estimator, param_grid = pick_probe(probe_name=probe_name)

    # 3. We further format the feature sets for the experiment
    logger.info("Formatting feature sets for target experiment...")
    # We run the encoding probe with all features combined and save results with name "all_features"
    # Then we remove syntax features and save results with name "no_syntax"
    # Thirdly, we remove lexical features and save results with name "no_lexical"
    # Lastly we remove both syntax and lexical features and save results with name "no_syntax_lexical"

    feature_set_removal_configurations = {
        "all_features": [],
        "no_syntax": ["syntax_feature"],
        "no_lexical": ["dnn_word_embedding"],
        "no_syntax_lexical": ["syntax_feature", "dnn_word_embedding"],
    }

    results = []

    for (
        config_name,
        feature_names_to_remove,
    ) in feature_set_removal_configurations.items():
        logger.info(
            f"Running target experiment with feature set configuration: {config_name}..."
        )
        mask_array = np.ones(feature_sets.shape[1], dtype=bool)
        for feature_name in feature_names_to_remove:
            feature_idx = all_feature_names.index(feature_name)
            start_idx, end_idx = (
                int(section_shapes[feature_idx][0]),
                int(section_shapes[feature_idx][1]),
            )
            mask_array[start_idx:end_idx] = False

        # Use mask_array to select features
        selected_feature_set = feature_sets.copy()[:, mask_array]

        for layer in trange(model_hidden_states.shape[1], desc="Layers"):
            logger.info(f"Running target experiment for layer {layer}...")
            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = train_test_split(
                selected_feature_set,
                model_hidden_states[:, layer, :],
                test_size=0.2,
                random_state=42,
            )

            GS = GridSearchCV(
                estimator,
                param_grid,
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            GS.fit(X_train, y_train)
            train_score = r2_score(
                y_train,
                GS.predict(X_train),
                multioutput="variance_weighted",
                train_data=y_train,
            )
            test_score = r2_score(
                y_test,
                GS.predict(X_test),
                multioutput="variance_weighted",
                train_data=y_train,
            )

            result = {
                "librispeech_split": librispeech_split,
                "modelname": modelname,
                "layer": layer,
                "config_name": config_name,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": GS.best_params_,
            }
            results.append(result)

    # 4. Plot and save the results
    plotting_results(results, modelname, librispeech_split)


def plotting_results(results, modelname, librispeech_split):
    # Plot the results
    import pandas as pd
    import plotnine as p9

    results_df = pd.DataFrame(results)

    # Rename config_name for better plotting labels
    rename_config_name = {
        "all_features": "All Features",
        "no_syntax": "No Syntax Features",
        "no_lexical": "No Lexical Features",
        "no_syntax_lexical": "No Syntax & Lexical Features",
    }
    results_df["config_name"] = results_df["config_name"].replace(rename_config_name)

    plot = (
        p9.ggplot(
            results_df,
            p9.aes(
                x="layer",
                y="test_score",
                color="config_name",
            ),
        )
        + p9.geom_line()
        + p9.geom_point()
        + p9.labs(
            title=f"Target Experiment Syntax and Lexical Features Interaction\nModel: {modelname}, Split: {librispeech_split}",
            x="Layer",
            y="Test R² Score",
            color="Feature Set Configuration",
        )
    )
    plot.show()

    logger.info("Saving results for target experiment...")
    results_savepath = os.path.join(
        RESULTS_ROOT,
        "target_experiment_syntax_lexical",
        f"{librispeech_split}_{modelname}_results.pkl",
    )
    os.makedirs(os.path.dirname(results_savepath), exist_ok=True)
    with open(results_savepath, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_savepath}")

    plot.save(
        filename=results_savepath.replace(".pkl", ".png"),
        dpi=300,
        width=10,
        height=6,
    )


if __name__ == "__main__":
    main()
