import glob
import logging
import os
import pickle
import sys
from typing import Union

import numpy as np
import pandas as pd
import plotnine as p9
from matplotlib import legend
from pandas import plotting
from plotnine.options import figure_size
from pyparsing import line

from utils import FIGURES_ROOT, RESULTS_ROOT

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Rename config_name
CONFIG_NAME_RENAME: dict = {
    "AllFeatures": "All Features",
    "AcousticOnly": "Acoustics Only",
    "eGeMAPSv02": "-Acoustics",
    "ChapterID-OH": "-Chapter ID",
    "SpeakerID-OH": "-Speaker",
    "dnn_word_embedding": "-Lexicon",
    "word_embedding": "-Lexicon",
    "ppg_feature": "-Phonetics",
    "syntax_feature": "-Syntax",
    "syntax_head_word_embedding": "Syntactic Head Lexicon",
    "syntax_feature+word_embedding": "-Syntax -Lexicon",
    "ppg_feature+eGeMAPSv02": "-Phonetics - Acoustics",
    "ppg_feature+SpeakerID-OH": "-Phonetics -Speaker",
    "SpeakerID-OH+eGeMAPSv02": "-Acoustics -Speaker",
    "SpeakerID-OH+eGeMAPSv02+ppg_feature": "-Acoustics -Phonetics -Speaker",
}


MODELNAME_ORDER: list = [
    "wav2vec2-base",
    "wav2vec2-base-960h",
    "wav2vec2-large",
    "hubert-base-ls960",
    "hubert-large-ll60k",
    "wavlm-base",
    "wav2vec2-base-superb-sid",
    "wav2vec2-ls100-sid",
    "bert-base-uncased",
    "roberta-base",
    "ModernBERT-base",
]

CONFIG_NAME_ORDER: list = [
    "All Features",
    "Acoustic Only",
    "-Acoustics",
    "-Lexicon",
    "-Phonetics",
    "-Syntax",
    "Syntactic Head Lexical",
    "-Speaker",
    "-Syntax -Lexicon",
    "-Acoustics -Speaker",
    "-Phonetics -Speaker",
    "Combined -Lexicon -Syntax",
    "Sum of Individual Effects",
]

PLOT_COLOR_MAPPING: dict = {
    "All Features": "#7f7f7f",
    "Acoustics Only": "#7f7f7f",
    "-Acoustics": "#1f77b4",
    "-Speaker": "#d62728",
    "-Lexicon": "#1f77b4",
    "-Phonetics": "#1f77b4",
    "-Syntax": "#ff7f0e",
    "Syntactic Head Lexical": "#bcbd22",
    "-Syntax -Lexicon": "#17cf76",
    "-Acoustics -Speaker": "#17cf76",
    "-Phonetics -Speaker": "#17cf76",
    "Combined -Lexicon -Syntax": "#8c564b",
    "Sum of Individual Effects": "#8c564b",
}


def plot_helper(
    results_df: pd.DataFrame,
    comparison_results_df: pd.DataFrame,
    facet: str = "modelname",
    color_mapping: Union[dict, None] = None,
    x_col: str = "normalized_layer",
    y_col: str = "test_score",
    legend_n_row: Union[int, None] = 2,
    include_sum_of_individual: bool = True,
) -> p9.ggplot:
    if not include_sum_of_individual:
        col_name = "Sum of Individual Effects"
        results_df = results_df[results_df["config_name"] != col_name].copy()
        comparison_results_df = comparison_results_df[
            comparison_results_df["config_name"] != col_name
        ].copy()

    # Order the modelname by a set list
    results_df["modelname"] = pd.Categorical(
        results_df["modelname"], categories=MODELNAME_ORDER, ordered=True
    )

    comparison_results_df["modelname"] = pd.Categorical(
        comparison_results_df["modelname"], categories=MODELNAME_ORDER, ordered=True
    )

    all_config_names = list(results_df["config_name"].unique()) + list(
        comparison_results_df["config_name"].unique()
    )

    modelname_order = [x for x in CONFIG_NAME_ORDER if x in all_config_names]

    results_df["config_name"] = pd.Categorical(
        results_df["config_name"],
        categories=modelname_order,
        ordered=True,
    )
    comparison_results_df["config_name"] = pd.Categorical(
        comparison_results_df["config_name"],
        categories=modelname_order,
        ordered=True,
    )
    linetype_mapping = {
        "All Features": "dashed",
        "Acoustics Only": "dashed",
    }
    linetype_mapping = {
        config: linetype_mapping.get(config, "solid") for config in all_config_names
    }

    # Drop all nan columns
    results_df = results_df.dropna(axis=1, how="any")
    comparison_results_df = comparison_results_df.dropna(axis=1, how="any")

    y_lim_max = max(results_df[y_col].max(), comparison_results_df[y_col].max()) * 1.05
    y_lim_min = min(results_df[y_col].min(), comparison_results_df[y_col].min()) * 0.95
    combi_df = pd.concat([results_df, comparison_results_df], ignore_index=True)

    p = (
        p9.ggplot()
        + p9.geom_line(
            data=combi_df,
            mapping=p9.aes(
                x=x_col,
                y=y_col,
                color="config_name",
                group="config_name",
                linetype="config_name",
            ),
        )
        + p9.geom_point(
            data=results_df,
            mapping=p9.aes(
                x=x_col,
                y=y_col,
                color="config_name",
                shape="config_name",
            ),
            size=0.7,
        )
        + p9.facet_wrap(facet)
        + p9.theme_minimal()
        + p9.scale_y_continuous(limits=(y_lim_min, y_lim_max))
        + p9.scale_linetype_manual(values=linetype_mapping)
    )
    if legend_n_row is not None:
        p += p9.guides(
            color=p9.guide_legend(nrow=legend_n_row, byrow=True),
            linetype=p9.guide_legend(nrow=legend_n_row, byrow=True),
            shape=p9.guide_legend(nrow=legend_n_row, byrow=True),
        )
    if color_mapping is not None:
        p += p9.scale_color_manual(values=color_mapping)
    if x_col == "normalized_layer":
        # Set the x-axis to be from 0 to 1 with breaks at every 0.25
        p += p9.scale_x_continuous(breaks=(0, 0.25, 0.5, 0.75, 1))
    else:
        p += p9.scale_x_continuous(
            breaks=np.arange(
                results_df["layer"].min(), results_df["layer"].max() + 1, 3
            )
        )
    return p


def read_all_results(librispeech_split: str = "dev-clean") -> pd.DataFrame:
    modelnames = (
        "facebook/wav2vec2-base",
        "facebook/wav2vec2-base-960h",
        "superb/wav2vec2-base-superb-sid",
        "facebook/wav2vec2-large",
        "facebook/hubert-base-ls960",
        "facebook/hubert-large-ll60k",
        "microsoft/wavlm-base",
        "FacebookAI/roberta-base",
        "google-bert/bert-base-uncased",
        "answerdotai/ModernBERT-base",
        "techsword/wav2vec2-ls100-sid",
    )

    all_results = []

    results_dirs = [
        "single_feat_removal",
        "syntax_feat_removal",
        "speakerid_phonetic_acoustic_removal",
    ]

    for results_dir in results_dirs:
        for modelname in modelnames:
            probe_name = "ridge"
            savepath = os.path.join(
                RESULTS_ROOT,
                results_dir,
                f"{librispeech_split}_{modelname.replace('/', '-')}_{probe_name}_results.pkl",
            )
            if os.path.exists(savepath):
                with open(savepath, "rb") as f:
                    results = pickle.load(f)
                results_df = pd.DataFrame(results)
                results_df["experiment"] = results_dir
                all_results.append(results_df)
            else:
                logger.warning(
                    f"Results not found for {librispeech_split} and {modelname} under {results_dir}. Skipping."
                )
    all_results_df = pd.concat(all_results, ignore_index=True)

    all_results_df["config_name"] = all_results_df["config_name"].map(
        lambda x: CONFIG_NAME_RENAME.get(x, x)
    )

    all_results_df["modelname"] = all_results_df["modelname"].map(
        lambda x: (
            x.split("/")[-1]
            # "Text: " + x.split("/")[-1]
            # if ("wav" not in x) and ("hubert" not in x)
            # else "Audio: " + x.split("/")[-1]
        )
    )

    # Normalize layer for each model to be from 0 to 1
    all_results_df["normalized_layer"] = all_results_df.groupby("modelname")[
        "layer"
    ].transform(lambda x: x / x.max())

    # Set the config_name to be a categorical variable with a specific order
    order_config_name = all_results_df["config_name"].unique().tolist()
    # Sort the config_name by the number of - in the name, with fewer - first, and if tie, sort alphabetically
    order_config_name.sort(key=lambda x: (x.count("-"), x))
    all_results_df["config_name"] = pd.Categorical(
        all_results_df["config_name"], categories=order_config_name, ordered=True
    )

    return all_results_df


def read_random_seed_results(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    probename: str = "ridge",
) -> pd.DataFrame:
    all_results = []

    results_dirs = [
        "single_feat_removal",
        "syntax_feat_removal",
        "speakerid_phonetic_acoustic_removal",
    ]

    for results_dir in results_dirs:
        savepath = os.path.join(
            RESULTS_ROOT,
            results_dir,
            f"{librispeech_split}_{modelname.replace('/', '-')}_{probename}_results.pkl",
        )
        savefiles = glob.glob(savepath.replace(".pkl", "-*.pkl"))
        for savefile in savefiles:
            with open(savefile, "rb") as f:
                results = pickle.load(f)
            results_df = pd.DataFrame(results)
            seed = savefile.split("-")[-1].split(".")[0]
            results_df["random_seed"] = int(seed.replace("seed", ""))
            results_df["experiment"] = results_dir
            all_results.append(results_df)
        if os.path.isfile(savepath):
            with open(savepath, "rb") as f:
                results = pickle.load(f)
            results_df = pd.DataFrame(results)
            results_df["random_seed"] = 42
            results_df["experiment"] = results_dir
            all_results.append(results_df)
    all_results_df = pd.concat(all_results, ignore_index=True)

    all_results_df["config_name"] = all_results_df["config_name"].map(
        lambda x: CONFIG_NAME_RENAME.get(x, x)
    )

    all_results_df["modelname"] = all_results_df["modelname"].map(
        lambda x: (
            x.split("/")[-1]
            # "Text: " + x.split("/")[-1]
            # if ("wav" not in x) and ("hubert" not in x)
            # else "Audio: " + x.split("/")[-1]
        )
    )

    # Normalize layer for each model to be from 0 to 1
    all_results_df["normalized_layer"] = all_results_df.groupby("modelname")[
        "layer"
    ].transform(lambda x: x / x.max())

    return all_results_df


def get_combined_single_results(
    mode_results_df,
    mode_comparison_results_df,
    target_configs: list,
    new_config_name: Union[str, None],
) -> pd.DataFrame:
    # Compute the amount of test_score departure of the different config_names to the topline for all layers
    if "random_seed" in mode_results_df.columns:
        mode_results_df["departure_to_topline"] = mode_results_df.apply(
            lambda row: (
                row["test_score"]
                - mode_comparison_results_df[
                    (mode_comparison_results_df["modelname"] == row["modelname"])
                    & (
                        mode_comparison_results_df["normalized_layer"]
                        == row["normalized_layer"]
                    )
                    & (mode_comparison_results_df["random_seed"] == row["random_seed"])
                ]["test_score"].values[0]
            ),
            axis=1,
        )
    else:
        mode_results_df["departure_to_topline"] = mode_results_df.apply(
            lambda row: (
                row["test_score"]
                - mode_comparison_results_df[
                    (mode_comparison_results_df["modelname"] == row["modelname"])
                    & (
                        mode_comparison_results_df["normalized_layer"]
                        == row["normalized_layer"]
                    )
                ]["test_score"].values[0]
            ),
            axis=1,
        )

    # Add the departure_to_topline from -Lexical and -Syntactic together and append that to the df under a new config_name
    combined_lex_syntx_departure = mode_results_df[
        mode_results_df["config_name"].isin(target_configs)
    ].copy()
    if "random_seed" in mode_results_df.columns:
        combined_lex_syntx_departure = (
            combined_lex_syntx_departure.groupby(
                [
                    combined_lex_syntx_departure["modelname"],
                    combined_lex_syntx_departure["normalized_layer"],
                    combined_lex_syntx_departure["random_seed"],
                ]
            )
            .agg({"departure_to_topline": "sum"})
            .reset_index()
        )
    else:
        combined_lex_syntx_departure = (
            combined_lex_syntx_departure.groupby(
                [
                    combined_lex_syntx_departure["modelname"],
                    combined_lex_syntx_departure["normalized_layer"],
                    combined_lex_syntx_departure["layer"],
                ]
            )
            .agg({"departure_to_topline": "sum"})
            .reset_index()
        )
    combined_lex_syntx_departure["config_name"] = (
        new_config_name
        if new_config_name is not None
        else "Combined " + " + ".join(target_configs)
    )
    # Use the departure to calculate the theoretical test_score from the topline
    if "random_seed" in mode_results_df.columns:
        combined_lex_syntx_departure["test_score"] = combined_lex_syntx_departure.apply(
            lambda row: (
                row["departure_to_topline"]
                + mode_comparison_results_df[
                    (mode_comparison_results_df["modelname"] == row["modelname"])
                    & (
                        mode_comparison_results_df["normalized_layer"]
                        == row["normalized_layer"]
                    )
                    & (mode_comparison_results_df["random_seed"] == row["random_seed"])
                ]["test_score"].values[0]
            ),
            axis=1,
        )
    else:
        combined_lex_syntx_departure["test_score"] = combined_lex_syntx_departure.apply(
            lambda row: (
                row["departure_to_topline"]
                + mode_comparison_results_df[
                    (mode_comparison_results_df["modelname"] == row["modelname"])
                    & (
                        mode_comparison_results_df["normalized_layer"]
                        == row["normalized_layer"]
                    )
                ]["test_score"].values[0]
            ),
            axis=1,
        )

    mode_results_df = pd.concat(
        [mode_results_df, combined_lex_syntx_departure], ignore_index=True
    )

    return mode_results_df


def plot_main_figures(all_results_df: pd.DataFrame):
    # For each mode, we plot the test score with the topline
    # Topline and baseline comparison points are "All Features" and "Acoustics Only"
    mode = "top-down"
    comparison_configs = ["All Features", "Acoustics Only"]
    mode_results_df = all_results_df[all_results_df["mode"] == mode]
    mode_comparison_results_df = mode_results_df[
        mode_results_df["config_name"].isin(comparison_configs)
    ].copy()
    mode_results_df = mode_results_df[
        ~mode_results_df["config_name"].isin(comparison_configs)
    ].copy()

    # Set up plotting configs
    plotting_configs = {
        "syntax_lexical": {
            "target_configs": [
                "-Lexicon",
                "-Syntax",
                "-Syntax Head Lexicon",
                "-Syntax -Lexicon",
            ],
            "target_models": [
                "bert-base-uncased",
                "wav2vec2-base",
            ],
        },
        "syntax_lexical_2": {
            "target_configs": [
                "-Lexicon",
                "-Syntax",
                "-Syntax Head Lexicon",
                "-Syntax -Lexicon",
            ],
            "target_models": [
                "wav2vec2-base-960h",
                "hubert-base-ls960",
            ],
        },
        "acoustics_speaker_id": {
            "target_configs": [
                "-Acoustics",
                "-Speaker",
                "-Acoustics -Speaker",
            ],
            "target_models": [
                "wav2vec2-base",
                "wav2vec2-base-960h",
                "wav2vec2-ls100-sid",
            ],
        },
        "phonetic_speaker_id": {
            "target_configs": [
                "-Phonetics",
                "-Speaker",
                "-Phonetics -Speaker",
            ],
            "target_models": [
                "wav2vec2-base",
                "wav2vec2-base-960h",
                "wav2vec2-ls100-sid",
            ],
        },
        "all_models_syntax_lexical": {
            "target_configs": [
                "-Lexicon",
                "-Syntax",
                "-Syntax Head Lexicon",
                "-Syntax -Lexicon",
            ],
            "target_models": list(mode_results_df["modelname"].unique()),
            "x_col": "normalized_layer",
            "figure_size": (8, 8),
        },
        "all_models_acoustic_speaker": {
            "target_configs": [
                "-Acoustics",
                "-Speaker",
                "-Acoustics -Speaker",
            ],
            "target_models": list(mode_results_df["modelname"].unique()),
            "x_col": "normalized_layer",
            "figure_size": (8, 8),
        },
        "all_models_phonetic_speaker": {
            "target_configs": [
                "-Phonetics",
                "-Speaker",
                "-Phonetics -Speaker",
            ],
            "target_models": list(mode_results_df["modelname"].unique()),
            "x_col": "normalized_layer",
            "figure_size": (8, 8),
        },
        "syntax_lexical_wav2vec2": {
            "target_configs": [
                "-Lexicon",
                "-Syntax",
                "-Syntactic Head Lexical",
                "-Syntax -Lexicon",
            ],
            "target_models": ["wav2vec2-base"],
            "x_col": "layer",
            "y_col": "test_score",
            "figure_size": (4, 4),
            "legend_n_row": 2,
        },
        "acoustics_speaker_id_wav2vec2": {
            "target_configs": [
                "-Acoustics",
                "-Speaker",
                "-Acoustics -Speaker",
            ],
            "target_models": ["wav2vec2-base"],
            "x_col": "layer",
            "y_col": "test_score",
            "figure_size": (4, 4),
            "legend_n_row": 2,
        },
        "phonetic_speaker_id_wav2vec2": {
            "target_configs": [
                "-Phonetics",
                "-Speaker",
                "-Phonetics -Speaker",
            ],
            "target_models": ["wav2vec2-base"],
            "x_col": "layer",
            "y_col": "test_score",
            "figure_size": (4, 4),
            "legend_n_row": 2,
        },
    }

    for featname, plotting_config in plotting_configs.items():
        target_configs = plotting_config["target_configs"]
        target_models = plotting_config["target_models"]
        x_col = plotting_config.get("x_col", "layer")
        y_col = plotting_config.get("y_col", "test_score")
        legend_n_row = plotting_config.get("legend_n_row", None)
        figure_size = plotting_config.get("figure_size", (6, 3))

        plot_df = mode_results_df[
            (mode_results_df["config_name"].isin(target_configs))
            & (mode_results_df["modelname"].isin(target_models))
        ]
        plot_compare_df = mode_comparison_results_df[
            mode_comparison_results_df["modelname"].isin(target_models)
        ]

        plot_df = get_combined_single_results(
            plot_df,
            plot_compare_df,
            target_configs=[
                x for x in target_configs if x.count("-") == 1
            ],  # Only include the configs with single feature removal for calculating the sum of individual effects
            new_config_name="Sum of Individual Effects",
        )

        # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
        p = plot_helper(
            plot_df,
            plot_compare_df,
            color_mapping=PLOT_COLOR_MAPPING,
            x_col=x_col,
            y_col=y_col,
            legend_n_row=legend_n_row,
            include_sum_of_individual=False,
        )

        p += p9.labs(
            x="Layer (From shallow to deep)",
            y=r"$R^2$ Score",
            color="Feature Group",
            shape="Feature Group",
            linetype="Feature Group",
        )
        p += p9.theme(
            legend_position="bottom",
            legend_justification="center",
            figure_size=figure_size,
            dpi=300,
            legend_title=p9.element_blank(),
        )
        p.show()

        p.save(os.path.join(FIGURES_ROOT, f"{mode}_{featname.lower()}_results.png"))


def plot_focus_random_seed(
    random_seed_results_df: pd.DataFrame, focus: str = "syntax_lexical"
):
    # Plot the syntax and lexical removal results,
    # to see if the random seed has an impact on the results

    focus_lookup = {
        "syntax_lexical": ["-Lexicon", "-Syntax", "-Syntax -Lexicon"],
        "phonetic_speaker": ["-Phonetics", "-Speaker", "-Phonetic -Speaker"],
        "acoustic_speaker": ["-Acoustics", "-Speaker", "-Speaker -Acoustics"],
    }

    subset_results_df = random_seed_results_df[
        random_seed_results_df["config_name"].isin(focus_lookup[focus])
    ].copy()
    subset_results_comparison_df = random_seed_results_df[
        random_seed_results_df["config_name"].isin(["All Features", "Acoustics Only"])
    ].copy()

    subset_results_df = get_combined_single_results(
        subset_results_df,
        subset_results_comparison_df,
        target_configs=focus_lookup[focus][:-1],  # Exclude the combined config
        new_config_name="Sum of Individual Effects",
    )

    subset_results_df["layer"] = subset_results_df["normalized_layer"] * 12
    subset_results_comparison_df["layer"] = (
        subset_results_comparison_df["normalized_layer"] * 12
    )  # Assuming wav2vec2-base has 12 layers

    # Compute the mean and std of the test_score for each config_name and layer across different random seeds
    subset_results_df_mean = (
        subset_results_df.groupby(["config_name", "modelname", "layer"])
        .agg(
            test_score_mean=("test_score", "mean"), test_score_std=("test_score", "std")
        )
        .reset_index()
    )
    subset_results_comparison_df_mean = (
        subset_results_comparison_df.groupby(["config_name", "modelname", "layer"])
        .agg(
            test_score_mean=("test_score", "mean"), test_score_std=("test_score", "std")
        )
        .reset_index()
    )

    # Print the mean std of the test_score for each config_name and layer across different random seeds
    print(f"Subset Results with Random Seeds for focus '{focus}':")
    print(
        subset_results_df_mean.groupby(["config_name"])[
            ["test_score_mean", "test_score_std"]
        ].mean()
    )

    # Calculate the confidence interval for the test_score_mean using the test_score_std and the number of random seeds
    num_random_seeds = random_seed_results_df["random_seed"].nunique()
    print(f"Number of random seeds: {num_random_seeds}")
    subset_results_df_mean["ci_lower"] = subset_results_df_mean[
        "test_score_mean"
    ] - 1.96 * subset_results_df_mean["test_score_std"] / np.sqrt(num_random_seeds)
    subset_results_df_mean["ci_upper"] = subset_results_df_mean[
        "test_score_mean"
    ] + 1.96 * subset_results_df_mean["test_score_std"] / np.sqrt(num_random_seeds)

    subset_results_comparison_df_mean["ci_lower"] = subset_results_comparison_df_mean[
        "test_score_mean"
    ] - 1.96 * subset_results_comparison_df_mean["test_score_std"] / np.sqrt(
        num_random_seeds
    )
    subset_results_comparison_df_mean["ci_upper"] = subset_results_comparison_df_mean[
        "test_score_mean"
    ] + 1.96 * subset_results_comparison_df_mean["test_score_std"] / np.sqrt(
        num_random_seeds
    )

    mean_ci_per_config = subset_results_df_mean.groupby("config_name").apply(
        lambda x: np.mean(x["ci_upper"] - x["ci_lower"])
    )
    mean_ci_width = np.mean(mean_ci_per_config)
    print(f"Mean confidence interval width: {mean_ci_width:.4f}")

    # Calculate the pairwise t-test between the different config_names for each layer to see if the difference is statistically significant
    from scipy.stats import ttest_ind

    config_names = subset_results_df["config_name"].unique()
    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            config_name_i = config_names[i]
            config_name_j = config_names[j]
            for layer in subset_results_df["layer"].unique():
                scores_i = subset_results_df[
                    (subset_results_df["config_name"] == config_name_i)
                    & (subset_results_df["layer"] == layer)
                ]["test_score"]
                scores_j = subset_results_df[
                    (subset_results_df["config_name"] == config_name_j)
                    & (subset_results_df["layer"] == layer)
                ]["test_score"]
                t_stat, p_value = ttest_ind(scores_i, scores_j)
                print(
                    f"T-test between {config_name_i} and {config_name_j} at layer {layer}: t-statistic={t_stat:.4f}, p-value={p_value:.4f}"
                )

    # Plot the random seed results with error bars using plotnine
    p = (
        p9.ggplot()
        + p9.geom_line(
            data=subset_results_df_mean,
            mapping=p9.aes(
                x="layer",
                y="test_score_mean",
                color="config_name",
                group="config_name",
            ),
        )
        + p9.geom_point(
            data=subset_results_df_mean,
            mapping=p9.aes(
                x="layer",
                y="test_score_mean",
                color="config_name",
                shape="config_name",
            ),
            size=0.7,
        )
        + p9.geom_errorbar(
            data=subset_results_df_mean,
            mapping=p9.aes(
                x="layer",
                ymin=subset_results_df_mean["ci_lower"],
                ymax=subset_results_df_mean["ci_upper"],
                color="config_name",
            ),
            width=0.02,
        )
        + p9.geom_line(
            data=subset_results_comparison_df_mean,
            mapping=p9.aes(x="layer", y="test_score_mean"),
            linetype="dashed",
            color="grey",
            size=1,
        )
        + p9.geom_errorbar(
            data=subset_results_comparison_df_mean,
            mapping=p9.aes(
                x="layer",
                ymin=subset_results_comparison_df_mean["ci_lower"],
                ymax=subset_results_comparison_df_mean["ci_upper"],
            ),
            width=0.02,
            linetype="dashed",
            color="grey",
        )
        + p9.theme_minimal()
        # + p9.scale_y_continuous(limits=(0, syntax_lexical_df["test_score_mean"].max() * 1.1))
        + p9.theme(
            figure_size=(5, 4),
            dpi=300,
            legend_position="bottom",
            legend_justification="center",
            legend_title=p9.element_blank(),
        )
        + p9.scale_x_continuous(
            breaks=np.arange(
                subset_results_df["layer"].min(),
                subset_results_df["layer"].max() + 1,
                3,
            )
        )
        + p9.labs(
            x="Layer (From shallow to deep)",
            y=r"$R^2$ Score",
            color="Feature",
            shape="Feature",
        )
        + p9.guides(color=p9.guide_legend(nrow=2, byrow=True))
        + p9.scale_color_manual(values=PLOT_COLOR_MAPPING)
    )
    p.show()
    modelname = random_seed_results_df["modelname"].iloc[0]
    modelname = modelname.split(": ")[-1]
    p.save(
        os.path.join(
            FIGURES_ROOT,
            f"{modelname}_random_seed_{focus}_results.png",
        )
    )


def main():
    librispeech_split = "dev-clean"
    librispeech_split = "train-clean-100"
    all_results_df = read_all_results(librispeech_split)
    plot_main_figures(all_results_df)

    librispeech_split = "train-clean-100"
    modelname = "facebook/wav2vec2-base"
    random_seed_results_df = read_random_seed_results(
        librispeech_split=librispeech_split, modelname=modelname
    )
    plot_focus_random_seed(random_seed_results_df, focus="syntax_lexical")
    plot_focus_random_seed(random_seed_results_df, focus="acoustic_speaker")
    plot_focus_random_seed(random_seed_results_df, focus="phonetic_speaker")

    modelname = "google-bert/bert-base-uncased"
    random_seed_results_df = read_random_seed_results(
        librispeech_split=librispeech_split, modelname=modelname
    )
    plot_focus_random_seed(random_seed_results_df)


if __name__ == "__main__":
    main()
