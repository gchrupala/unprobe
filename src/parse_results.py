import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Union

import numpy as np
import pandas as pd
import plotnine as p9

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

SYNTAX_COMPONENT_RENAME: dict[str, str] = {
    "syntax_POS_OH": "-Syntax POS",
    "syntax_Dependency_Label_OH": "-Syntax Dependency",
    "syntax_Tree_Depth": "-Syntax Tree Depth",
    "syntax_Word_Position": "-Syntax Position",
    "syntax_Total_Tree_Depth": "-Syntax Total Tree Depth",
    "syntax_Total_Word_Count": "-Syntax Total Word Count",
}

CONFIG_NAME_ORDER: list = [
    "All Features",
    "Acoustic Only",
    "-Acoustics",
    "-Lexicon",
    "-Phonetics",
    "-Syntax",
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
    "-Syntax -Lexicon": "#17cf76",
    "-Acoustics -Speaker": "#17cf76",
    "-Phonetics -Speaker": "#17cf76",
    "Combined -Lexicon -Syntax": "#8c564b",
    "Sum of Individual Effects": "#8c564b",
}


def _rename_syntax_component_config(config_name: str) -> str:
    for component, label in SYNTAX_COMPONENT_RENAME.items():
        token = f"word_embedding+{component}"
        if config_name == token:
            return f"-Lexicon {label.replace('-Syntax ', '-Syntax ')}"
    return config_name


def _safe_list(value, default: list[str]) -> list[str]:
    return value if isinstance(value, list) else default


def _safe_str(value, default: str) -> str:
    return value if isinstance(value, str) else default


def _safe_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    return value if isinstance(value, int) else default


def _safe_tuple2(value, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    return default


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
        "syntax_lexicon_decomposition",
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
        "syntax_lexicon_decomposition",
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


def plot_main_figures(all_results_df: pd.DataFrame, show_plots: bool = False):
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
        "syntax_lexicon_decomposition_wav2vec2": {
            "target_configs": [
                "-Lexicon",
                "-Lexicon -Syntax POS",
                "-Lexicon -Syntax Dependency",
                "-Lexicon -Syntax Tree Depth",
                "-Lexicon -Syntax Position",
                "-Lexicon -Syntax Total Tree Depth",
                "-Lexicon -Syntax Total Word Count",
            ],
            "target_models": ["wav2vec2-base"],
            "x_col": "layer",
            "y_col": "test_score",
            "figure_size": (4, 4),
            "legend_n_row": 2,
        },
    }

    for featname, plotting_config in plotting_configs.items():
        target_configs = _safe_list(plotting_config["target_configs"], [])
        target_models = _safe_list(plotting_config["target_models"], [])
        x_col = _safe_str(plotting_config.get("x_col", "layer"), "layer")
        y_col = _safe_str(plotting_config.get("y_col", "test_score"), "test_score")
        legend_n_row = _safe_int(plotting_config.get("legend_n_row", None), None)
        figure_size = _safe_tuple2(plotting_config.get("figure_size", (6, 3)), (6, 3))

        plot_df = mode_results_df.copy()
        plot_df = plot_df.copy()
        plot_df["config_name"] = plot_df["config_name"].map(
            _rename_syntax_component_config
        )
        plot_df = plot_df[
            (plot_df["config_name"].isin(target_configs))
            & (plot_df["modelname"].isin(target_models))
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
        if show_plots:
            p.show()

        p.save(os.path.join(FIGURES_ROOT, f"{mode}_{featname.lower()}_results.png"))


def plot_focus_random_seed(
    random_seed_results_df: pd.DataFrame,
    focus: str = "syntax_lexical",
    show_plot: bool = False,
    print_ttest: bool = False,
):
    # Plot the syntax and lexical removal results,
    # to see if the random seed has an impact on the results

    focus_lookup = {
        "syntax_lexical": ["-Lexicon", "-Syntax", "-Syntax -Lexicon"],
        "phonetic_speaker": ["-Phonetics", "-Speaker", "-Phonetics -Speaker"],
        "acoustic_speaker": ["-Acoustics", "-Speaker", "-Acoustics -Speaker"],
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

    # Order subset_results_comparison_df_mean based on how many - is in the config_name, with fewer - first, and if tie, sort alphabetically
    subset_results_comparison_df_mean["config_name"] = pd.Categorical(
        subset_results_comparison_df_mean["config_name"],
        categories=CONFIG_NAME_ORDER,
        ordered=True,
    )

    # Print the mean std of the test_score for each config_name and layer across different random seeds
    logger.info("Subset Results with Random Seeds for focus '%s':", focus)
    logger.info(
        "\n%s",
        subset_results_df_mean.groupby(["config_name"])[
            ["test_score_mean", "test_score_std"]
        ].mean(),
    )

    # Calculate the confidence interval for the test_score_mean using the test_score_std and the number of random seeds
    num_random_seeds = random_seed_results_df["random_seed"].nunique()
    logger.info("Number of random seeds: %s", num_random_seeds)
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
    logger.info("Mean confidence interval width: %.4f", mean_ci_width)

    if print_ttest:
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
                    logger.info(
                        "T-test between %s and %s at layer %s: t-statistic=%.4f, p-value=%.4f",
                        config_name_i,
                        config_name_j,
                        layer,
                        t_stat,
                        p_value,
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
    if show_plot:
        p.show()
    modelname = random_seed_results_df["modelname"].iloc[0]
    modelname = modelname.split(": ")[-1]
    p.save(
        os.path.join(
            FIGURES_ROOT,
            f"{modelname}_random_seed_{focus}_results.png",
        )
    )


def summarize_random_seed_line_differences(
    random_seed_results_df: pd.DataFrame,
    focus_lookup: dict[str, Union[list[str], dict[str, object]]],
    output_dir: str = RESULTS_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize pairwise line differences across random seeds and layers.

    Args:
        random_seed_results_df: DataFrame containing random-seed runs.
        focus_lookup: Mapping of focus names to either:
            - ordered config list (single removals + joint removal), or
            - dict with keys `single_configs`, `joint_config`, and optional
              `comparison_configs` / `line_configs`.
        output_dir: Directory to save CSV summaries.

    Returns:
        Tuple of (summary_df, layerwise_df).
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_rows: list[dict] = []
    layerwise_rows: list[dict] = []

    for focus, focus_spec in focus_lookup.items():

        def _norm_key(value: float) -> float:
            return round(float(value), 8)

        if isinstance(focus_spec, dict):
            single_configs = _safe_list(focus_spec.get("single_configs", []), [])
            joint_config = focus_spec.get("joint_config")
            if isinstance(joint_config, str):
                line_configs = _safe_list(
                    focus_spec.get("line_configs", [*single_configs, joint_config]),
                    [*single_configs, joint_config],
                )
            else:
                line_configs = _safe_list(
                    focus_spec.get("line_configs", single_configs), single_configs
                )
            comparison_configs = _safe_list(
                focus_spec.get(
                    "comparison_configs", ["All Features", "Acoustics Only"]
                ),
                ["All Features", "Acoustics Only"],
            )
        else:
            line_configs = list(focus_spec)
            single_configs = (
                line_configs[:-1] if len(line_configs) > 1 else line_configs
            )
            joint_config = line_configs[-1] if line_configs else None
            comparison_configs = ["All Features", "Acoustics Only"]

        subset = random_seed_results_df[
            random_seed_results_df["config_name"].isin(line_configs)
        ].copy()
        if subset.empty:
            logger.warning("No rows found for focus '%s'; skipping summary.", focus)
            continue

        subset_comparison = random_seed_results_df[
            random_seed_results_df["config_name"].isin(comparison_configs)
        ].copy()

        layer_lookup: dict[float, int] = {}
        if "layer" in subset.columns:
            layer_rows = subset[["normalized_layer", "layer"]].dropna()
            if not layer_rows.empty:
                layer_lookup = (
                    layer_rows.groupby("normalized_layer")["layer"]
                    .agg(lambda x: int(round(float(np.median(x)))))
                    .rename(index=lambda x: _norm_key(x))
                    .to_dict()
                )

        if (
            len(single_configs) >= 2
            and isinstance(joint_config, str)
            and joint_config in subset["config_name"].unique()
            and not subset_comparison.empty
        ):
            subset = get_combined_single_results(
                subset,
                subset_comparison,
                target_configs=single_configs,
                new_config_name="Sum of Individual Effects",
            )

        pivot_df = subset.pivot_table(
            index=["random_seed", "normalized_layer"],
            columns="config_name",
            values="test_score",
            aggfunc="mean",
        )

        compare_order = [*line_configs, "Sum of Individual Effects"]
        present_configs = [cfg for cfg in compare_order if cfg in pivot_df.columns]
        if len(present_configs) < 2:
            logger.warning(
                "Focus '%s' has fewer than 2 present config lines (%s); skipping.",
                focus,
                present_configs,
            )
            continue

        for i in range(len(present_configs)):
            for j in range(i + 1, len(present_configs)):
                cfg_a = present_configs[i]
                cfg_b = present_configs[j]
                pair_name = f"{cfg_a} - {cfg_b}"

                delta_series = (pivot_df[cfg_a] - pivot_df[cfg_b]).dropna()
                if delta_series.empty:
                    continue

                n_points = int(delta_series.shape[0])
                mean_delta = float(delta_series.mean())
                std_delta = float(delta_series.std(ddof=1)) if n_points > 1 else 0.0
                sem_delta = std_delta / np.sqrt(n_points) if n_points > 1 else 0.0
                ci_low = mean_delta - 1.96 * sem_delta
                ci_high = mean_delta + 1.96 * sem_delta
                mean_abs_delta = float(np.abs(delta_series).mean())
                pct_positive = float((delta_series > 0).mean())

                # Seed-level aggregation to avoid overweighting layers
                seed_level_delta = (
                    delta_series.reset_index().groupby("random_seed")[0].mean().dropna()
                )
                n_seed = int(seed_level_delta.shape[0])
                seed_mean_delta = float(seed_level_delta.mean())
                seed_std_delta = (
                    float(seed_level_delta.std(ddof=1)) if n_seed > 1 else 0.0
                )
                seed_sem_delta = seed_std_delta / np.sqrt(n_seed) if n_seed > 1 else 0.0
                seed_ci_low = seed_mean_delta - 1.96 * seed_sem_delta
                seed_ci_high = seed_mean_delta + 1.96 * seed_sem_delta

                from scipy.stats import ttest_1samp

                t_stat, p_value = (
                    ttest_1samp(seed_level_delta, popmean=0.0)
                    if n_seed > 1
                    else (np.nan, np.nan)
                )

                summary_rows.append(
                    {
                        "focus": focus,
                        "pair": pair_name,
                        "config_a": cfg_a,
                        "config_b": cfg_b,
                        "joint_config": joint_config,
                        "single_configs": "+".join(single_configs),
                        "is_joint_vs_sum": int(
                            {
                                cfg_a,
                                cfg_b,
                            }
                            == {joint_config, "Sum of Individual Effects"}
                        )
                        if isinstance(joint_config, str)
                        else 0,
                        "n_points": n_points,
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "mean_abs_delta": mean_abs_delta,
                        "pct_positive": pct_positive,
                        "n_seed": n_seed,
                        "seed_mean_delta": seed_mean_delta,
                        "seed_std_delta": seed_std_delta,
                        "seed_ci_low": seed_ci_low,
                        "seed_ci_high": seed_ci_high,
                        "t_stat_seed_mean_delta": float(t_stat)
                        if not np.isnan(t_stat)
                        else np.nan,
                        "p_value_seed_mean_delta": float(p_value)
                        if not np.isnan(p_value)
                        else np.nan,
                    }
                )

                for (seed, normalized_layer), delta_val in delta_series.items():
                    norm_key = _norm_key(normalized_layer)
                    layerwise_rows.append(
                        {
                            "focus": focus,
                            "pair": pair_name,
                            "config_a": cfg_a,
                            "config_b": cfg_b,
                            "joint_config": joint_config,
                            "single_configs": "+".join(single_configs),
                            "is_joint_vs_sum": int(
                                {
                                    cfg_a,
                                    cfg_b,
                                }
                                == {joint_config, "Sum of Individual Effects"}
                            )
                            if isinstance(joint_config, str)
                            else 0,
                            "random_seed": int(seed),
                            "normalized_layer": float(normalized_layer),
                            "layer": layer_lookup.get(norm_key, np.nan),
                            "delta": float(delta_val),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    layerwise_df = pd.DataFrame(layerwise_rows)

    if summary_df.empty:
        logger.warning("No random-seed difference summaries were generated.")
        return summary_df, layerwise_df

    modelname = random_seed_results_df["modelname"].iloc[0].split(": ")[-1]
    split_name = random_seed_results_df["librispeech_split"].iloc[0]
    summary_path = os.path.join(
        output_dir,
        f"{modelname}_{split_name}_random_seed_line_diff_summary.csv",
    )
    layerwise_path = os.path.join(
        output_dir,
        f"{modelname}_{split_name}_random_seed_line_diff_layerwise.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    layerwise_df.to_csv(layerwise_path, index=False)

    logger.info("Saved random-seed difference summary to %s", summary_path)
    logger.info("Saved random-seed layerwise deltas to %s", layerwise_path)

    for focus in focus_lookup:
        focus_rows = summary_df[summary_df["focus"] == focus]
        if focus_rows.empty:
            continue
        interaction_rows = focus_rows[focus_rows["is_joint_vs_sum"] == 1]
        if not interaction_rows.empty:
            logger.info(
                "Joint vs Sum-of-individuals for focus '%s':\n%s",
                focus,
                interaction_rows[
                    [
                        "pair",
                        "mean_delta",
                        "ci_low",
                        "ci_high",
                        "seed_mean_delta",
                        "seed_ci_low",
                        "seed_ci_high",
                        "p_value_seed_mean_delta",
                    ]
                ].to_string(index=False),
            )
        logger.info(
            "Top pairwise differences for focus '%s':\n%s",
            focus,
            focus_rows.sort_values("mean_abs_delta", ascending=False)
            .head(3)[
                [
                    "pair",
                    "mean_delta",
                    "ci_low",
                    "ci_high",
                    "mean_abs_delta",
                    "pct_positive",
                    "p_value_seed_mean_delta",
                ]
            ]
            .to_string(index=False),
        )

    return summary_df, layerwise_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display generated plots interactively. If unset, plots are only saved.",
    )
    parser.add_argument(
        "--print_ttest",
        action="store_true",
        help="Print per-layer pairwise t-test results to the console.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    librispeech_split = "dev-clean"
    librispeech_split = "train-clean-100"
    all_results_df = read_all_results(librispeech_split)
    plot_main_figures(all_results_df, show_plots=args.show_plots)

    librispeech_split = "train-clean-100"
    modelname = "facebook/wav2vec2-base"
    random_seed_results_df = read_random_seed_results(
        librispeech_split=librispeech_split, modelname=modelname
    )

    all_focus_lookup: dict[str, Union[list[str], dict[str, object]]] = {
        "syntax_lexical": {
            "single_configs": ["-Lexicon", "-Syntax"],
            "joint_config": "-Syntax -Lexicon",
        },
        "acoustic_speaker": {
            "single_configs": ["-Acoustics", "-Speaker"],
            "joint_config": "-Acoustics -Speaker",
        },
        "phonetic_speaker": {
            "single_configs": ["-Phonetics", "-Speaker"],
            "joint_config": "-Phonetics -Speaker",
        },
    }
    summarize_random_seed_line_differences(
        random_seed_results_df=random_seed_results_df,
        focus_lookup=all_focus_lookup,
    )

    plot_focus_random_seed(
        random_seed_results_df,
        focus="syntax_lexical",
        show_plot=args.show_plots,
        print_ttest=args.print_ttest,
    )
    plot_focus_random_seed(
        random_seed_results_df,
        focus="acoustic_speaker",
        show_plot=args.show_plots,
        print_ttest=args.print_ttest,
    )
    plot_focus_random_seed(
        random_seed_results_df,
        focus="phonetic_speaker",
        show_plot=args.show_plots,
        print_ttest=args.print_ttest,
    )

    modelname = "google-bert/bert-base-uncased"
    random_seed_results_df = read_random_seed_results(
        librispeech_split=librispeech_split, modelname=modelname
    )
    summarize_random_seed_line_differences(
        random_seed_results_df=random_seed_results_df,
        focus_lookup=all_focus_lookup,
    )
    plot_focus_random_seed(
        random_seed_results_df,
        show_plot=args.show_plots,
        print_ttest=args.print_ttest,
    )


if __name__ == "__main__":
    main()
