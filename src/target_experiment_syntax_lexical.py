# This script runs target experiments using the encoding probe paradigm
# to investigate how well different feature groups (syntax, lexical) interact in the
# hidden representations of a DNN model pre-trained with audio data.

import logging
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import plotnine as p9
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


def run_probe(
    feature_set_configurations,
    feature_sets,
    model_hidden_states,
    section_shapes,
    all_feature_names,
    librispeech_split,
    modelname,
    estimator,
    param_grid,
    bottom_up=False,
):
    results = []

    for (
        config_name,
        feature_names_to_remove,
    ) in tqdm(
        feature_set_configurations.items(), position=1, desc="Feature Set Configs"
    ):
        logger.info(
            f"Running target experiment with feature set configuration: {config_name}..."
        )
        if bottom_up:
            mask_array = np.zeros(feature_sets.shape[1], dtype=bool)
        else:
            mask_array = np.ones(feature_sets.shape[1], dtype=bool)
        for feature_name in feature_names_to_remove:
            feature_idx = all_feature_names.index(feature_name)
            start_idx, end_idx = (
                int(section_shapes[feature_idx][0]),
                int(section_shapes[feature_idx][1]),
            )
            if bottom_up:
                mask_array[start_idx:end_idx] = True
            else:
                mask_array[start_idx:end_idx] = False

        # Use mask_array to select features
        if config_name == "all_features":
            # Use all features (no masking)
            selected_feature_set = feature_sets.copy()
        else:
            # Use mask_array to select specific features
            selected_feature_set = feature_sets.copy()[:, mask_array]

        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=2, leave=False
        ):
            logger.debug(f"Running target experiment for layer {layer}...")
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

    return results


def main():
    args = parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname.split("/")[-1]
    select_layers = args.select_layers
    normalize_features = args.normalize_features
    overwrite = args.overwrite
    probe_name = args.probe_name
    bottom_up = True

    # 1. We load the data for the experiment
    logger.info("Loading data for target experiment...")
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        reduce_dnn_word_embedding=True,
        one_hot_encode_syntax=False,
        one_hot_encode_syntax_separate=True,
        one_hot_encode_metadata=True,
        argmax_ppg=False,
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

    # Programmatically create feature set removal configurations according to all_feature_names
    # We want all combinations of features that start with syntax_ and dnn_word_embedding

    syntax_features_names = [f for f in all_feature_names if f.startswith("syntax_")]

    def select_combinations(r):
        combined_feature_names = list(
            combinations(syntax_features_names + ["dnn_word_embedding"], r)
        )
        # Create a dictionary to hold the configurations of combined_feature_names and give them config_names
        config_names = [
            "+".join(
                [
                    name.replace("syntax_", "")
                    .replace("_OH", "")
                    .replace("dnn_word_embedding", "Lexical")
                    for name in x
                ]
            )
            for x in combined_feature_names
        ]

        feature_set_configurations = dict(zip(config_names, combined_feature_names))
        return feature_set_configurations

    baseline_configurations = {
        "name": "baseline",
        "feature_set_configurations": {
            "all_features": [],
            "no_syntax_no_lexical": syntax_features_names + ["dnn_word_embedding"],
            "no_syntax": syntax_features_names,
            "no_lexical": ["dnn_word_embedding"],
        },
        "save_path": os.path.join(
            RESULTS_ROOT,
            "target_experiment_syntax_lexical",
            f"{librispeech_split}_{modelname}_baseline_results.pkl",
        ),
    }
    experimental_configurations = []
    experimental_configurations.append(baseline_configurations)
    for r in range(1, 5):
        intermediate_savepath = os.path.join(
            RESULTS_ROOT,
            "target_experiment_syntax_lexical",
            f"{librispeech_split}_{modelname}_intermediate_r-{r}_results.pkl",
        )

        feature_set_configurations = select_combinations(r)
        experimental_configurations.append(
            {
                "name": f"r-{r}",
                "feature_set_configurations": feature_set_configurations,
                "save_path": intermediate_savepath,
            }
        )

    results = {}

    for config in tqdm(experimental_configurations, position=0, desc="Configurations"):
        name = config["name"]
        feature_set_configurations = config["feature_set_configurations"]
        intermediate_savepath = config["save_path"]
        r_results = run_probe(
            feature_set_configurations=feature_set_configurations,
            feature_sets=feature_sets,
            model_hidden_states=model_hidden_states,
            section_shapes=section_shapes,
            all_feature_names=all_feature_names,
            librispeech_split=librispeech_split,
            modelname=modelname,
            estimator=estimator,
            param_grid=param_grid,
            bottom_up=bottom_up,
        )
        results.update({name: r_results})

        # Save intermediate results after each r
        if bottom_up:
            intermediate_savepath = intermediate_savepath.replace(
                ".pkl", "_bottom_up.pkl"
            )
        with open(intermediate_savepath, "wb") as f:
            pickle.dump(r_results, f)
        logger.info(f"Intermediate results for {name} saved to {intermediate_savepath}")


def plot_existing_results(modelname, librispeech_split, results):
    results_df = pd.DataFrame(results)

    # Rename config_name for better plotting labels
    results_df["config_name"] = results_df["config_name"].str.replace(
        "dnn_word_embedding", "Lexical"
    )

    plot = (
        p9.ggplot(
            results_df,
            p9.aes(
                x="layer",
                y="test_score",
                color="config_name",
                shape="config_name",
            ),
        )
        + p9.geom_line()
        + p9.geom_point()
        + p9.labs(
            title=f"Target Experiment Syntax and Lexical Features Interaction\nModel: {modelname}, Split: {librispeech_split}",
            x="Layer",
            y="Test R² Score",
            color="Feature Set Configuration",
            shape="Feature Set Configuration",
        )
    )
    return plot


def plot_results_split(librispeech_split, modelname, r=3, bottom_up=False):
    # Split the big plot into 2x3 grid and use the individual feature names as the header and show the results that has that individual
    results_file = f"{RESULTS_ROOT}/target_experiment_syntax_lexical/{librispeech_split}_{modelname}_intermediate_r-{r}_results.pkl"
    baseline_file = f"{RESULTS_ROOT}/target_experiment_syntax_lexical/{librispeech_split}_{modelname}_baseline_results.pkl"
    if bottom_up:
        results_file = results_file.replace(".pkl", "_bottom_up.pkl")
        baseline_file = baseline_file.replace(".pkl", "_bottom_up.pkl")
        figure_path_all = results_file.replace(".pkl", "_bottom_up.png")
        figure_path_lexical = results_file.replace(".pkl", "_bottom_up_lexical.png")
    else:
        figure_path_all = results_file.replace(".pkl", ".png")
        figure_path_lexical = results_file.replace(".pkl", "_lexical.png")
    figure_path_all = figure_path_all.replace(
        f"{RESULTS_ROOT}/target_experiment_syntax_lexical/",
        f"{RESULTS_ROOT}/target_experiment_syntax_lexical/figures/",
    )
    figure_path_lexical = figure_path_lexical.replace(
        f"{RESULTS_ROOT}/target_experiment_syntax_lexical/",
        f"{RESULTS_ROOT}/target_experiment_syntax_lexical/figures/",
    )
    os.makedirs(os.path.dirname(figure_path_all), exist_ok=True)
    with open(
        results_file,
        "rb",
    ) as f:
        r_results = pickle.load(f)

    # Load baseline results
    with open(
        baseline_file,
        "rb",
    ) as f:
        baseline_results = pickle.load(f)

    panels_names = [
        "POS",
        "Dependency_Label",
        "Tree_Depth",
        "Word_Position",
        "Lexical",
    ]

    results_df = pd.DataFrame(r_results)
    # Separate topline with all_features out from two_combi_results_df
    baseline_df = pd.DataFrame(baseline_results)
    # Ignore no_syntax in baseline
    baseline_df = baseline_df[
        baseline_df["config_name"].isin(
            ["all_features", "no_syntax_no_lexical", "no_lexical"]
        )
    ].reset_index(drop=True)
    # Rename config names for better plotting labels
    rename_config_name = {
        "all_features": "Topline -- All Features",
        "no_syntax_no_lexical": "Syntax & Lexical Features",
        "no_syntax": "Syntax Features",
        "no_lexical": "Lexical Features",
    }

    # Keep the config names not in the rename_config_name as is
    results_df["config_name"] = results_df["config_name"].apply(
        lambda x: rename_config_name.get(x, x)
    )
    baseline_df["config_name"] = baseline_df["config_name"].apply(
        lambda x: rename_config_name.get(x, x)
    )

    # We can remove the topline value from bottom up results since that's not a valid comparison
    if bottom_up:
        baseline_df = baseline_df[
            baseline_df["config_name"] != "Topline -- All Features"
        ]

    import matplotlib.pyplot as plt

    # Set the color mapping programmatically according to config_name
    unique_config_names = results_df["config_name"].unique().tolist()
    color_palette = plt.get_cmap("tab10").colors  # type: ignore
    color_mapping = {
        config_name: color_palette[i % len(color_palette)]
        for i, config_name in enumerate(unique_config_names)
    }
    # GIve the baseline lines with special shapes
    markers_mapping = {
        "Topline -- All Features": "s",
        "Syntax & Lexical Features": "X",
        "Syntax Features": "o",
        "Lexical Features": "d",
    }

    # # Setup grid
    # plt.figure(figsize=(20, 30))
    # for i, panel_name in enumerate(panels_names):
    #     plt.subplot(3, 2, i + 1)
    #     subset_df = results_df[
    #         (results_df["config_name"].str.contains(panel_name))
    #     ].reset_index(drop=True)
    #     # Add topline to each subplot
    #     subset_df = pd.concat([baseline_df, subset_df], ignore_index=True)
    #     for config_name, group in subset_df.groupby("config_name"):
    #         color = color_mapping.get(
    #             config_name, "#000000"
    #         )  # Default to black if not found
    #         marker = markers_mapping.get(config_name, "o")  # type: ignore
    #         plt.plot(
    #             group["layer"],
    #             group["test_score"],
    #             marker=marker,
    #             label=config_name,
    #             color=color,
    #         )
    #     if bottom_up:
    #         title = f"Bottom-Up Addition of {panel_name} + {r - 1} feature"
    #     else:
    #         title = f"Top-Down Removal of {panel_name} + {r - 1} feature"
    #     plt.title(title)
    #     plt.xlabel("Layer")
    #     plt.ylabel("Test R² Score")
    #     plt.legend()
    # plt.tight_layout()
    # # plt.show()

    # plt.savefig(figure_path_all, dpi=300)
    # # Clear the figure to avoid overlap
    # plt.clf()
    # logger.info(f"Saved figure to {figure_path_all}")

    if r == 1:
        return
    # Make a plot specifically focused on removals involving lexical features
    plt.figure(figsize=(10, 8))
    subset_df = results_df[
        results_df["config_name"].str.contains("Lexical")
    ].reset_index(drop=True)
    subset_df = pd.concat([baseline_df, subset_df], ignore_index=True)
    for config_name, group in subset_df.groupby("config_name"):
        color = color_mapping.get(
            config_name, "#808080"
        )  # Default to gray if not found
        marker = markers_mapping.get(config_name, "o")  # type: ignore
        plt.plot(
            group["layer"],
            group["test_score"],
            marker=marker,
            label=config_name,
            color=color,
        )
    # Set y-axis limits to focus on the relevant range
    plt.ylim(0, 0.4)
    if bottom_up:
        plt.title(f"Additions Involving Lexical Features + {r - 1} Feature(s)")
    else:
        plt.title(f"Removals Involving Lexical Features + {r - 1} Feature(s)")
    plt.xlabel("Layer")
    plt.ylabel("Test R² Score")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figure_path_lexical, dpi=300)
    logger.info(f"Saved figure to {figure_path_lexical}")
    plt.clf()


def save_figs_results_split():
    for bottom_up in (True, False):
        for r in range(1, 5):
            for modelname in ["wav2vec2-base", "bert-base-uncased"]:
                plot_results_split(
                    "train-clean-100", modelname, r=r, bottom_up=bottom_up
                )
    bottom_up = True
    librispeech_split = "train-clean-100"
    modelname = "wav2vec2-base"
    r = 3


def check_lexical_information():
    """Use the lexical information to predict syntactic information directly one-by-one and save results to a csv file"""
    librispeech_split = "train-clean-100"
    modelname = "wav2vec2-base"
    select_layers = None
    normalize_features = True
    overwrite = False

    # 1. We load the data for the experiment
    logger.info("Loading data for target experiment...")
    feature_sets, _, _, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        reduce_dnn_word_embedding=True,
        one_hot_encode_syntax=False,
        one_hot_encode_syntax_separate=False,
        one_hot_encode_metadata=True,
        argmax_ppg=False,
        normalize_features=normalize_features,
    )
    syntax_feature_idx = {
        "POS": 0,
        "Dependency_Label": 1,
        "Constituent_Label": 2,
        "Tree_Depth": 3,
        # "Tree_Depth_Normed": 4,
        "Word_Position": 5,
        # "Word_Position_Normed": 6,
    }

    section_shapes = get_section_shapes(data_shape)

    lexical_information = feature_sets[
        :, int(section_shapes[3, 0]) : int(section_shapes[3, 1])
    ]

    syntax_features = feature_sets[
        :, int(section_shapes[1, 0]) : int(section_shapes[1, 1])
    ]

    results = []

    for syntax_feature_name, idx in tqdm(
        syntax_feature_idx.items(), desc="Syntax Features", position=0
    ):
        syntax_feature = syntax_features[:, idx]

        # Turn syntax_feature into class labels
        syntax_feature = syntax_feature.astype(int)

        estimator, param_grid = pick_probe(
            probe_name="ridge_classifier"
        )  # Use classifier for syntax features
        X_train, X_test, y_train, y_test = train_test_split(
            lexical_information,
            syntax_feature,
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
        train_score = GS.score(X_train, y_train)
        test_score = GS.score(X_test, y_test)

        results.append(
            {
                "syntax_feature": syntax_feature_name,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": GS.best_params_,
            }
        )
    results_df = pd.DataFrame(results)
    results_savepath = os.path.join(
        RESULTS_ROOT,
        "word_embedding_to_syntax_results.csv",
    )
    results_df.to_csv(results_savepath, index=False)
    logger.info(f"Lexical to syntax results saved to {results_savepath}")


if __name__ == "__main__":
    main()
