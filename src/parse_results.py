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


def plot_helper(
    results_df: pd.DataFrame,
    comparison_results_df: pd.DataFrame,
    facet: str = "modelname",
    color_mapping: Union[dict, None] = None,
) -> p9.ggplot:
    p = (
        p9.ggplot()
        + p9.geom_line(
            data=results_df,
            mapping=p9.aes(
                x="normalized_layer",
                y="test_score",
                color="config_name",
                group="config_name",
            ),
        )
        + p9.geom_point(
            data=results_df,
            mapping=p9.aes(
                x="normalized_layer",
                y="test_score",
                color="config_name",
                shape="config_name",
            ),
        )
        + p9.geom_line(
            data=comparison_results_df,
            mapping=p9.aes(x="normalized_layer", y="test_score"),
            linetype="dashed",
            color="grey",
            size=1,
        )
        + p9.facet_wrap(facet)
        + p9.theme_minimal()
        # Set the x-axis to be from 0 to 1 with breaks at every 0.25
        + p9.scale_x_continuous(breaks=(0, 0.25, 0.5, 0.75, 1))
        # Set y-axis to 0 to 0.5
        + p9.scale_y_continuous(limits=(0, results_df["test_score"].max() * 1.1))
        + p9.theme(figure_size=(8, 6), dpi=300)
    )
    if color_mapping is not None:
        p += p9.scale_color_manual(values=color_mapping)
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

    # Rename config_name
    config_name_rename: dict = {
        "AllFeatures": "All Features",
        "AcousticOnly": "Acoustic Only",
        "eGeMAPSv02": "Acoustic",
        "ChapterID-OH": "Chapter ID",
        "SpeakerID-OH": "Speaker ID",
        "dnn_word_embedding": "Lexical",
        "word_embedding": "Lexical",
        "ppg_feature": "Phonetic",
        "syntax_feature": "Syntactic",
        "syntax_head_word_embedding": "Syntactic Head Lexical",
        "syntax_feature+word_embedding": "Syntactic + Lexical",
        "ppg_feature+eGeMAPSv02": "Phonetic + Acoustic",
        "ppg_feature+SpeakerID-OH": "Phonetic + Speaker ID",
        "SpeakerID-OH+eGeMAPSv02": "Speaker ID + Acoustic",
        "SpeakerID-OH+eGeMAPSv02+ppg_feature": "Speaker ID + Acoustic + Phonetic",
    }

    all_results_df["config_name"] = all_results_df["config_name"].map(
        lambda x: config_name_rename.get(x, x)
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


def plot_results_syntax_lexical(all_results_df: pd.DataFrame):
    plot_color_mapping: dict = {
        "All Features": "#7f7f7f",
        "Acoustic Only": "#7f7f7f",
        "Acoustic": "#2ca02c",
        "Chapter ID": "#d62728",
        "Speaker ID": "#9467bd",
        "Lexical": "#ff7f0e",
        "Phonetic": "#e377c2",
        "Syntactic": "#1f77b4",
        "Syntactic Head Lexical": "#bcbd22",
        "Syntactic + Lexical": "#17cf76",
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
        "Lexical",
        "Syntactic",
        "Syntactic Head Lexical",
        "Syntactic + Lexical",
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

    p.show()


def plot_results_acoustic_phonetic_speaker(all_results_df: pd.DataFrame):
    plot_color_mapping: dict = {
        "All Features": "#7f7f7f",
        "Speaker ID": "#0173b2",
        "Phonetic": "#de8f05",
        "Acoustic": "#029e73",
        "Phonetic + Acoustic": "#cc78bc",
        "Phonetic + Speaker ID": "#b27700",
        "Speaker ID + Acoustic": "#CC4B4B",
        "Speaker ID + Acoustic + Phonetic": "#ece133",
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
        "Acoustic",
        # "Phonetic",
        "Speaker ID",
        # "Phonetic + Acoustic",
        # "Phonetic + Speaker ID",
        "Speaker ID + Acoustic",
        # "Speaker ID + Acoustic + Phonetic",
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
    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(plot_df, plot_compare_df, color_mapping=plot_color_mapping)

    p += p9.labs(
        x="Normalized Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p.show()

    target_configs = [
        "Phonetic",
        "Speaker ID",
        "Phonetic + Speaker ID",
        # "Speaker ID + Acoustic",
        "Speaker ID + Acoustic + Phonetic",
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
    # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
    p = plot_helper(plot_df, plot_compare_df, color_mapping=plot_color_mapping)

    p += p9.labs(
        x="Normalized Layer (From shallow to deep)",
        y="Test R2 Score",
        color="Feature Group",
        shape="Feature Group",
    )
    p.show()


def main():
    librispeech_split = "dev-clean"
    all_results_df = read_all_results(librispeech_split)
    plot_results_syntax_lexical(all_results_df)
    plot_results_acoustic_phonetic_speaker(all_results_df)


if __name__ == "__main__":
    main()
