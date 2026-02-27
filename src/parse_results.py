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
    "AcousticOnly": "Acoustic Only",
    "eGeMAPSv02": "-Acoustic",
    "ChapterID-OH": "-Chapter ID",
    "SpeakerID-OH": "-Speaker ID",
    "dnn_word_embedding": "-Lexical",
    "word_embedding": "-Lexical",
    "ppg_feature": "-Phonetic",
    "syntax_feature": "-Syntactic",
    "syntax_head_word_embedding": "Syntactic Head Lexical",
    "syntax_feature+word_embedding": "-Syntactic -Lexical",
    "ppg_feature+eGeMAPSv02": "-Phonetic + Acoustic",
    "ppg_feature+SpeakerID-OH": "-Phonetic -Speaker ID",
    "SpeakerID-OH+eGeMAPSv02": "-Speaker ID -Acoustic",
    "SpeakerID-OH+eGeMAPSv02+ppg_feature": "-Speaker ID -Acoustic -Phonetic",
}


MODELNAME_ORDER: list = [
    "wav2vec2-base",
    "wav2vec2-base-960h",
    "wav2vec2-large",
    "hubert-base-ls960",
    "hubert-large-ll60k",
    "wav2vec2-base-superb-sid",
    "bert-base-uncased",
    "roberta-base",
    "ModernBERT-base",
]

CONFIG_NAME_ORDER: list = [
    "All Features",
    "Acoustic Only",
    "-Acoustic",
    "-Lexical",
    "-Phonetic",
    "-Syntactic",
    "Syntactic Head Lexical",
    "-Speaker ID",
    "-Syntactic -Lexical",
    "-Speaker ID -Acoustic",
    "-Phonetic -Speaker ID",
    "Combined -Lexical -Syntactic",
    "Sum of Individual Effects",
]

PLOT_COLOR_MAPPING: dict = {
    "All Features": "#7f7f7f",
    "Acoustic Only": "#7f7f7f",
    "-Acoustic": "#1f77b4",
    "-Speaker ID": "#d62728",
    "-Lexical": "#1f77b4",
    "-Phonetic": "#1f77b4",
    "-Syntactic": "#ff7f0e",
    "Syntactic Head Lexical": "#bcbd22",
    "-Syntactic -Lexical": "#17cf76",
    "-Speaker ID -Acoustic": "#17cf76",
    "-Phonetic -Speaker ID": "#17cf76",
    "Combined -Lexical -Syntactic": "#8c564b",
    "Sum of Individual Effects": "#8c564b",
}


def plot_helper(
    results_df: pd.DataFrame,
    comparison_results_df: pd.DataFrame,
    facet: str = "modelname",
    color_mapping: Union[dict, None] = None,
    x_col: str = "normalized_layer",
    y_col: str = "test_score",
) -> p9.ggplot:
    # Order the modelname by a set list
    results_df["modelname"] = pd.Categorical(
        results_df["modelname"], categories=MODELNAME_ORDER, ordered=True
    )
    comparison_results_df["modelname"] = pd.Categorical(
        comparison_results_df["modelname"], categories=MODELNAME_ORDER, ordered=True
    )

    modelname_order = [
        x for x in CONFIG_NAME_ORDER if x in results_df["config_name"].unique()
    ]

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

    y_lim_max = max(results_df[y_col].max(), comparison_results_df[y_col].max()) * 1.05
    y_lim_min = min(results_df[y_col].min(), comparison_results_df[y_col].min()) * 0.95
    p = (
        p9.ggplot()
        + p9.geom_line(
            data=results_df,
            mapping=p9.aes(
                x=x_col,
                y=y_col,
                color="config_name",
                group="config_name",
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
        + p9.geom_line(
            data=comparison_results_df,
            mapping=p9.aes(x=x_col, y=y_col),
            linetype="dashed",
            color="grey",
            size=1,
        )
        + p9.facet_wrap(facet)
        + p9.theme_minimal()
        # Set y-axis to 0 to 0.5
        + p9.scale_y_continuous(limits=(y_lim_min, y_lim_max))
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


def plot_results_syntax_lexical(all_results_df: pd.DataFrame):
    # For each mode, we plot the test score with the topline
    # Topline and baseline comparison points are "All Features" and "Acoustic Only"
    mode = "top-down"
    comparison_configs = ["All Features", "Acoustic Only"]
    mode_results_df = all_results_df[all_results_df["mode"] == mode]
    mode_comparison_results_df = mode_results_df[
        mode_results_df["config_name"].isin(comparison_configs)
    ]
    mode_results_df = mode_results_df[
        ~mode_results_df["config_name"].isin(comparison_configs)
    ]

    # Filter out relevant config_name for the plot

    target_configs = [
        "-Lexical",
        "-Syntactic",
        "-Syntactic Head Lexical",
        "-Syntactic -Lexical",
    ]
    target_models = [
        "bert-base-uncased",
        # "roberta-base",
        # "ModernBERT-base",
        "wav2vec2-base",
        # "wav2vec2-base-960h",
        # "wav2vec2-large",
        "hubert-base-ls960",
        # "hubert-large-ll60k",
    ]

    plotting_df = mode_results_df[
        (mode_results_df["config_name"].isin(target_configs))
        & (mode_results_df["modelname"].isin(target_models))
    ]
    plotting_compare_df = mode_comparison_results_df[
        (mode_comparison_results_df["modelname"].isin(target_models))
    ]

    plotting_df = get_combined_single_results(
        plotting_df,
        plotting_compare_df,
        target_configs=["-Lexical", "-Syntactic"],
        new_config_name="Sum of Individual Effects",
    )

    # Plot the results in plotting_df and use plotting_compare_df as the baseline with dashed gray line
    p = plot_helper(
        plotting_df,
        plotting_compare_df,
        color_mapping=PLOT_COLOR_MAPPING,
        x_col="layer",
    )
    p += p9.labs(
        x="Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p += p9.theme(
        legend_justification="center",
        dpi=300,
        legend_position="none",
        figure_size=(6, 2.5),
        axis_text_x=p9.element_blank(),
        axis_title_x=p9.element_blank(),
    )

    p.save(os.path.join(FIGURES_ROOT, f"{mode}_syntax_lexical_results.png"))

    p.show()

    target_models = [
        # "bert-base-uncased",
        # "roberta-base",
        # "ModernBERT-base",
        "wav2vec2-base",
        "wav2vec2-base-960h",
        # "wav2vec2-large",
        "hubert-base-ls960",
        # "hubert-large-ll60k",
    ]

    plotting_df = mode_results_df[
        (mode_results_df["config_name"].isin(target_configs))
        & (mode_results_df["modelname"].isin(target_models))
    ]
    plotting_compare_df = mode_comparison_results_df[
        (mode_comparison_results_df["modelname"].isin(target_models))
    ]

    plotting_df = get_combined_single_results(
        plotting_df,
        plotting_compare_df,
        target_configs=["-Lexical", "-Syntactic"],
        new_config_name="Sum of Individual Effects",
    )

    # Plot the results in plotting_df and use plotting_compare_df as the baseline with dashed gray line
    p = plot_helper(
        plotting_df,
        plotting_compare_df,
        color_mapping=PLOT_COLOR_MAPPING,
        x_col="layer",
    )
    p += p9.labs(
        x="Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p += p9.theme(
        legend_position="bottom",
        legend_justification="center",
        dpi=300,
        figure_size=(6, 3),
        legend_title=p9.element_blank(),
    )
    p.save(os.path.join(FIGURES_ROOT, f"{mode}_syntax_lexical_results_2.png"))

    p.show()


def plot_results_acoustic_phonetic_speaker(all_results_df: pd.DataFrame):
    # For each mode, we plot the test score with the topline
    # Topline and baseline comparison points are "All Features" and "Acoustic Only"
    mode = "top-down"
    comparison_configs = ["All Features", "Acoustic Only"]
    mode_results_df = all_results_df[all_results_df["mode"] == mode]
    mode_comparison_results_df = mode_results_df[
        mode_results_df["config_name"].isin(comparison_configs)
    ]
    mode_results_df = mode_results_df[
        ~mode_results_df["config_name"].isin(comparison_configs)
    ]

    # Filter out relevant config_name for the plot

    target_configs = [
        "-Acoustic",
        "-Speaker ID",
        "-Speaker ID -Acoustic",
    ]
    target_models = [
        "wav2vec2-base",
        # "hubert-base-ls960",
        "wav2vec2-base-960h",
        # "hubert-large-ll60k",
        "wav2vec2-base-superb-sid",
    ]

    plot_df = mode_results_df[
        (mode_results_df["config_name"].isin(target_configs))
        & (mode_results_df["modelname"].isin(target_models))
    ]
    plot_compare_df = mode_comparison_results_df[
        (mode_comparison_results_df["modelname"].isin(target_models))
    ]

    plot_df = get_combined_single_results(
        plot_df,
        plot_compare_df,
        target_configs=["-Acoustic", "-Speaker ID"],
        new_config_name="Sum of Individual Effects",
    )

    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(
        plot_df, plot_compare_df, color_mapping=PLOT_COLOR_MAPPING, x_col="layer"
    )

    p += p9.labs(
        x="Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p += p9.theme(
        legend_position="bottom",
        legend_justification="center",
        figure_size=(6, 3),
        dpi=300,
        legend_title=p9.element_blank(),
    )
    p.save(os.path.join(FIGURES_ROOT, f"{mode}_acoustic_speaker_id_results.png"))
    p.show()

    target_configs = [
        "-Phonetic",
        "-Speaker ID",
        "-Phonetic -Speaker ID",
    ]
    target_models = [
        "wav2vec2-base",
        # "hubert-base-ls960",
        "wav2vec2-base-960h",
        # "hubert-large-ll60k",
        "wav2vec2-base-superb-sid",
    ]

    plot_df = mode_results_df[
        (mode_results_df["config_name"].isin(target_configs))
        & (mode_results_df["modelname"].isin(target_models))
    ]
    plot_compare_df = mode_comparison_results_df[
        (mode_comparison_results_df["modelname"].isin(target_models))
    ]

    plot_df = get_combined_single_results(
        plot_df,
        plot_compare_df,
        target_configs=["-Phonetic", "-Speaker ID"],
        new_config_name="Sum of Individual Effects",
    )
    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(
        plot_df, plot_compare_df, color_mapping=PLOT_COLOR_MAPPING, x_col="layer"
    )

    p += p9.labs(
        x="Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p += p9.theme(
        legend_position="bottom",
        legend_justification="center",
        figure_size=(6, 3),
        dpi=300,
        legend_title=p9.element_blank(),
    )
    p.save(os.path.join(FIGURES_ROOT, f"{mode}_phonetic_speaker_id_results.png"))


def plot_focus_wav2vec2_base(all_results_df: pd.DataFrame):
    """Focus on the wav2vec2 results

    Args:
        all_results_df (pd.DataFrame): _description_
    """

    plotting_df = all_results_df[
        (all_results_df["modelname"] == "wav2vec2-base")
        & (all_results_df["mode"] == "top-down")
    ]

    plotting_comparison_df = plotting_df[
        plotting_df["config_name"].isin(["All Features", "Acoustic Only"])
    ]
    plotting_df = plotting_df[
        ~plotting_df["config_name"].isin(["All Features", "Acoustic Only"])
    ]

    syntax_focus_configs = [
        "-Lexical",
        "-Syntactic",
        "-Syntactic Head Lexical",
        "-Syntactic -Lexical",
    ]

    syntax_df = plotting_df[
        plotting_df["config_name"].isin(syntax_focus_configs)
    ].copy()

    syntax_df = get_combined_single_results(
        syntax_df,
        plotting_comparison_df,
        target_configs=["-Lexical", "-Syntactic"],
        new_config_name="Sum of Individual Effects",
    )

    syntax_p = plot_helper(
        syntax_df,
        plotting_comparison_df,
        color_mapping={
            "All Features": "#7f7f7f",
            "Acoustic Only": "#7f7f7f",
            "-Lexical": "#ff7f0e",
            "-Syntactic": "#1f77b4",
            "-Syntactic Head Lexical": "#bcbd22",
            "-Syntactic -Lexical": "#17cf76",
            "Combined -Lexical -Syntactic": "#8c564b",
            "Sum of Individual Effects": "#8c564b",
        },
        x_col="layer",
    )
    syntax_p += p9.labs(
        x="Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature",
        shape="Feature",
    )
    syntax_p += p9.theme(
        figure_size=(5, 4),
        dpi=300,
        legend_position="bottom",
        legend_justification="center",
        legend_title=p9.element_blank(),
        # Remove the facet label since we only have one facet
        strip_background=p9.element_blank(),
        strip_text=p9.element_blank(),
    )
    syntax_p += p9.guides(color=p9.guide_legend(nrow=2, byrow=True))
    syntax_p += p9.scale_y_continuous(
        limits=(0.15, plotting_comparison_df["test_score"].max() * 1.1)
    )

    syntax_p.show()

    syntax_p.save(
        os.path.join(FIGURES_ROOT, "wav2vec2-base_top-down_syntax_lexical_results.png")
    )

    acoustic_focus_configs = [
        "-Acoustic",
        "-Speaker ID",
        "-Speaker ID -Acoustic",
    ]

    acoustic_df = plotting_df[
        plotting_df["config_name"].isin(acoustic_focus_configs)
    ].copy()
    acoustic_df = get_combined_single_results(
        acoustic_df,
        plotting_comparison_df,
        target_configs=["-Acoustic", "-Speaker ID"],
        new_config_name="Sum of Individual Effects",
    )
    acoustic_df["layer"] = (
        acoustic_df["normalized_layer"] * 12
    )  # Assuming wav2vec2-base has 12 layers
    acoustic_p = plot_helper(
        acoustic_df,
        plotting_comparison_df,
        color_mapping={
            "All Features": "#7f7f7f",
            "Acoustic Only": "#7f7f7f",
            "-Acoustic": "#029e73",
            "-Speaker ID": "#0173b2",
            "-Speaker ID -Acoustic": "#CC4B4B",
            "Combined -Acoustic -Speaker ID": "#8c564b",
            "Sum of Individual Effects": "#8c564b",
        },
        x_col="layer",
    )
    acoustic_p += p9.labs(
        x="Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature",
        shape="Feature",
    )
    acoustic_p += p9.theme(
        figure_size=(5, 4),
        dpi=300,
        legend_position="bottom",
        legend_justification="center",
        legend_title=p9.element_blank(),
        # Remove the facet label since we only have one facet
        strip_background=p9.element_blank(),
        strip_text=p9.element_blank(),
    )
    acoustic_p += p9.guides(color=p9.guide_legend(nrow=2, byrow=True))
    acoustic_p += p9.scale_y_continuous(
        limits=(0.1, plotting_comparison_df["test_score"].max() * 1.1)
    )
    acoustic_p += p9.scale_x_continuous(
        breaks=np.arange(acoustic_df["layer"].min(), acoustic_df["layer"].max() + 1, 3)
    )
    acoustic_p.show()
    acoustic_p.save(
        os.path.join(
            FIGURES_ROOT, "wav2vec2-base_top-down_acoustic_speaker_id_results.png"
        )
    )


def plot_focus_random_seed(
    random_seed_results_df: pd.DataFrame, focus: str = "syntax_lexical"
):
    # Plot the syntax and lexical removal results,
    # to see if the random seed has an impact on the results

    focus_lookup = {
        "syntax_lexical": ["-Lexical", "-Syntactic", "-Syntactic -Lexical"],
        "phonetic_speaker": ["-Phonetic", "-Speaker ID", "-Phonetic -Speaker ID"],
        "acoustic_speaker": ["-Acoustic", "-Speaker ID", "-Speaker ID -Acoustic"],
    }

    subset_results_df = random_seed_results_df[
        random_seed_results_df["config_name"].isin(focus_lookup[focus])
    ].copy()
    subset_results_comparison_df = random_seed_results_df[
        random_seed_results_df["config_name"].isin(["All Features", "Acoustic Only"])
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
            y="Test R2 Score",
            color="Feature",
            shape="Feature",
        )
        + p9.guides(color=p9.guide_legend(nrow=2, byrow=True))
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
    plot_results_syntax_lexical(all_results_df)
    plot_results_acoustic_phonetic_speaker(all_results_df)
    plot_focus_wav2vec2_base(all_results_df)

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
