import glob
import logging
import os
import pickle
import sys
from typing import Union

import numpy as np
import pandas as pd
import plotnine as p9
from tqdm.auto import tqdm, trange

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


def plot_helper(
    results_df: pd.DataFrame,
    comparison_results_df: pd.DataFrame,
    facet: str = "modelname",
    color_mapping: Union[dict, None] = None,
    x_col: str = "normalized_layer",
    y_col: str = "test_score",
) -> p9.ggplot:
    y_lim = max(results_df[y_col].max(), comparison_results_df[y_col].max()) * 1.1
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
        + p9.scale_y_continuous(limits=(0, y_lim))
        + p9.theme(figure_size=(8, 6), dpi=300)
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
        lambda x: "Text: " + x.split("/")[-1]
        if ("wav" not in x) and ("hubert" not in x)
        else "Audio: " + x.split("/")[-1]
    )

    # Normalize layer for each model to be from 0 to 1
    all_results_df["normalized_layer"] = all_results_df.groupby("modelname")[
        "layer"
    ].transform(lambda x: x / x.max())

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
        lambda x: "Text: " + x.split("/")[-1]
        if ("wav" not in x) and ("hubert" not in x)
        else "Audio: " + x.split("/")[-1]
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
    mode_results_df["departure_to_topline"] = mode_results_df.apply(
        lambda row: row["test_score"]
        - mode_comparison_results_df[
            (mode_comparison_results_df["modelname"] == row["modelname"])
            & (
                mode_comparison_results_df["normalized_layer"]
                == row["normalized_layer"]
            )
        ]["test_score"].values[0],
        axis=1,
    )

    # Add the departure_to_topline from -Lexical and -Syntactic together and append that to the df under a new config_name
    combined_lex_syntx_departure = mode_results_df[
        mode_results_df["config_name"].isin(target_configs)
    ].copy()
    combined_lex_syntx_departure = (
        combined_lex_syntx_departure["departure_to_topline"]
        .groupby(
            [
                combined_lex_syntx_departure["modelname"],
                combined_lex_syntx_departure["normalized_layer"],
            ]
        )
        .sum()
    ).reset_index()
    combined_lex_syntx_departure["config_name"] = (
        new_config_name
        if new_config_name is not None
        else "Combined " + " + ".join(target_configs)
    )
    # Use the departure to calculate the theoretical test_score from the topline
    combined_lex_syntx_departure["test_score"] = combined_lex_syntx_departure.apply(
        lambda row: row["departure_to_topline"]
        + mode_comparison_results_df[
            (mode_comparison_results_df["modelname"] == row["modelname"])
            & (
                mode_comparison_results_df["normalized_layer"]
                == row["normalized_layer"]
            )
        ]["test_score"].values[0],
        axis=1,
    )

    mode_results_df = pd.concat(
        [mode_results_df, combined_lex_syntx_departure], ignore_index=True
    )

    return mode_results_df


def plot_results_syntax_lexical(all_results_df: pd.DataFrame):
    plot_color_mapping: dict = {
        "All Features": "#7f7f7f",
        "Acoustic Only": "#7f7f7f",
        "-Acoustic": "#2ca02c",
        "Chapter ID": "#d62728",
        "-Speaker ID": "#9467bd",
        "-Lexical": "#ff7f0e",
        "-Phonetic": "#e377c2",
        "-Syntactic": "#1f77b4",
        "Syntactic Head Lexical": "#bcbd22",
        "-Syntactic -Lexical": "#17cf76",
        "Combined -Lexical -Syntactic": "#8c564b",
    }

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
        "Text: bert-base-uncased",
        # "Text: roberta-base",
        # "Text: ModernBERT-base",
        "Audio: wav2vec2-base",
        "Audio: wav2vec2-base-960h",
        "Audio: wav2vec2-large",
        "Audio: hubert-base-ls960",
        "Audio: hubert-large-ll60k",
    ]

    mode_results_df = mode_results_df[
        (mode_results_df["config_name"].isin(target_configs))
        & (mode_results_df["modelname"].isin(target_models))
    ]
    mode_comparison_results_df = mode_comparison_results_df[
        (mode_comparison_results_df["modelname"].isin(target_models))
    ]

    mode_results_df = get_combined_single_results(
        mode_results_df,
        mode_comparison_results_df,
        target_configs=["-Lexical", "-Syntactic"],
        new_config_name="Combined -Lexical -Syntactic",
    )

    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(
        mode_results_df, mode_comparison_results_df, color_mapping=plot_color_mapping
    )
    p += p9.labs(
        x="Normalized Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p += p9.theme(legend_position="bottom", legend_justification="center")

    p.save(os.path.join(FIGURES_ROOT, f"{mode}_syntax_lexical_results.png"))

    p.show()


def plot_results_acoustic_phonetic_speaker(all_results_df: pd.DataFrame):
    plot_color_mapping: dict = {
        "All Features": "#7f7f7f",
        "-Speaker ID": "#0173b2",
        "-Phonetic": "#de8f05",
        "-Acoustic": "#029e73",
        "-Phonetic -Acoustic": "#cc78bc",
        "-Phonetic -Speaker ID": "#b27700",
        "-Speaker ID -Acoustic": "#CC4B4B",
        "-Speaker ID -Acoustic -Phonetic": "#ece133",
        "Combined -Phonetic -Speaker ID": "#8c564b",
    }

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
        "Audio: wav2vec2-base",
        "Audio: hubert-base-ls960",
        "Audio: wav2vec2-base-960h",
        # "Audio: hubert-large-ll60k",
        "Audio: wav2vec2-base-superb-sid",
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
        new_config_name="Combined -Acoustic -Speaker ID",
    )

    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(plot_df, plot_compare_df, color_mapping=plot_color_mapping)

    p += p9.labs(
        x="Normalized Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p += p9.theme(legend_position="bottom", legend_justification="center")
    p.save(os.path.join(FIGURES_ROOT, f"{mode}_acoustic_speaker_id_results.png"))
    p.show()

    target_configs = [
        "-Phonetic",
        "-Speaker ID",
        "-Phonetic -Speaker ID",
    ]
    target_models = [
        "Audio: wav2vec2-base",
        "Audio: hubert-base-ls960",
        "Audio: wav2vec2-base-960h",
        # "Audio: hubert-large-ll60k",
        "Audio: wav2vec2-base-superb-sid",
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
        new_config_name="Combined -Phonetic -Speaker ID",
    )
    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(plot_df, plot_compare_df, color_mapping=plot_color_mapping)

    p += p9.labs(
        x="Normalized Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p.save(os.path.join(FIGURES_ROOT, f"{mode}_phonetic_speaker_id_results.png"))


def plot_focus_wav2vec2_base(all_results_df: pd.DataFrame):
    """Focus on the wav2vec2 results

    Args:
        all_results_df (pd.DataFrame): _description_
    """

    plotting_df = all_results_df[
        (all_results_df["modelname"] == "Audio: wav2vec2-base")
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
        new_config_name="Combined -Lexical -Syntactic",
    )

    syntax_df["layer"] = (
        syntax_df["normalized_layer"] * 12
    )  # Assuming wav2vec2-base has 12 layers

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
        dpi=200,
        legend_position="bottom",
        legend_justification="center",
        legend_title=p9.element_blank(),
        # Remove the facet label since we only have one facet
        strip_background=p9.element_blank(),
        strip_text=p9.element_blank(),
    )
    syntax_p += p9.guides(color=p9.guide_legend(nrow=2, byrow=True))
    syntax_p += p9.scale_y_continuous(
        limits=(0.15, syntax_df["test_score"].max() * 1.1)
    )

    syntax_p.show()

    syntax_p.save(
        os.path.join(FIGURES_ROOT, "wav2vec2-base_top-down_syntax_lexical_results.png")
    )


def plot_focus_random_seed(random_seed_results_df: pd.DataFrame):
    # Plot the syntax and lexical removal results,
    # to see if the random seed has an impact on the results
    syntax_lexical_df = random_seed_results_df[
        random_seed_results_df["config_name"].isin(
            [
                "-Lexical",
                "-Syntactic",
                "Combined -Lexical -Syntactic",
            ]
        )
    ].copy()
    syntax_lexical_comparison_df = random_seed_results_df[
        random_seed_results_df["config_name"].isin(["All Features", "Acoustic Only"])
    ].copy()
    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(
        syntax_lexical_df,
        syntax_lexical_comparison_df,
        facet="random_seed",
        color_mapping={
            "All Features": "#7f7f7f",
            "Acoustic Only": "#7f7f7f",
            "-Lexical": "#ff7f0e",
            "-Syntactic": "#1f77b4",
            "Combined -Lexical -Syntactic": "#8c564b",
        },
    )
    p += p9.labs(
        x="Normalized Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature",
        shape="Feature",
    )
    p += p9.theme(legend_position="bottom", legend_justification="center")
    # p.save(os.path.join(FIGURES_ROOT, f"random_seed_syntax_lexical_results.png"))
    p.show()


def main():
    librispeech_split = "dev-clean"
    librispeech_split = "train-clean-100"
    all_results_df = read_all_results(librispeech_split)
    plot_results_syntax_lexical(all_results_df)
    plot_results_acoustic_phonetic_speaker(all_results_df)
    plot_focus_wav2vec2_base(all_results_df)

    random_seed_results_df = read_random_seed_results(librispeech_split)


if __name__ == "__main__":
    main()
