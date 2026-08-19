"""Unified plotting entry point for probe outputs.

This script keeps the legacy `src/parse_results.py` untouched while
collecting the functionality that previously lived in `plotting_results.py`
and `analyze_results.py`.  It exposes a small CLI so operators can generate
figures for different result modalities from a single place.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
import seaborn as sns

from parse_results import (
    _build_main_figure_filename,
    _build_random_seed_figure_filename,
    plot_focus_random_seed,
    plot_main_figures,
    plot_manipulation_comparison,
    read_all_results,
    read_manipulation_comparison_results,
    read_random_seed_results,
)
from utils import FIGURES_ROOT, RESULTS_ROOT


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


MODEL_LAYER_COUNTS: dict[str, int] = {
    "wav2vec2-base": 12,
    "wav2vec2-base-960h": 12,
    "wav2vec2-large": 24,
    "wav2vec2-large-960h": 24,
    "wav2vec2-large-xlsr-53": 24,
    "hubert-base-ls960": 12,
    "hubert-large-ll60k": 24,
    "hubert-large-ls960-ft": 24,
    "wavlm-base": 12,
    "roberta-base": 12,
    "bert-base-uncased": 12,
    "ModernBERT-base": 22,
    "wav2vec2-base-superb-sid": 12,
}


RENAME_FEATURE_GROUPS: dict[str, str] = {
    "Prosodic & Voice Quality": "Prosodic Features",
    "Spectral Envelope": "Spectral Features",
    "SpectralInfo": "Spectral Features",
    "Formant Characteristics": "Formants",
    "syntax_feature": "Syntactic Features",
    "ppg_feature": "PPG",
    "spk_embedding": "Speaker Embedding",
    "metadata": "Metadata",
    "dnn_word_embedding": "DNN Word Embedding",
    "eGeMAPSv02": "eGeMAPSv02",
    "SpeakerID-OH": "Speaker ID",
    "ChapterID-OH": "Chapter ID",
    "Book ID": "Chapter ID",
    "ppg_ID": "Phone ID from PPGs",
    "ppg_feature_onehot": "One-hot Encoded PPGs",
}


def load_layer_csvs(
    results_dir: str = RESULTS_ROOT,
    pattern: str = "librispeech-*/**/layer_*.csv",
) -> pd.DataFrame:
    """Load per-layer CSV outputs produced by the frame probes."""

    search_glob = os.path.join(results_dir, pattern)
    candidate_files = glob.glob(search_glob, recursive=True)
    files = [
        path
        for path in candidate_files
        if "all_layers" not in path
        and "dimreduction" not in path
        and "_normalized" in path
    ]
    if not files:
        raise FileNotFoundError(f"No CSV layers found under {results_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        file_path = Path(path)
        try:
            config_dir = file_path.parent.name
            modelname = file_path.parent.parent.name
            librispeech_split = file_path.parent.parent.parent.name
        except IndexError as exc:
            raise ValueError(f"Unexpected path layout for {file_path}") from exc

        parts = config_dir.split("_")
        if len(parts) < 4:
            logger.warning(
                "Skipping %s because the config directory name is malformed", file_path
            )
            continue
        probename, *_prefix, normalization = parts[:4]
        df = pd.read_csv(file_path)
        prefixed_model = (
            f"Text: {modelname}"
            if ("wav" not in modelname and "hubert" not in modelname)
            else f"Audio: {modelname}"
        )
        df["modelname"] = prefixed_model
        df["librispeech_split"] = librispeech_split
        df["probename"] = "random_forest" if "random" in probename else probename
        df["normalization"] = normalization
        df["norm_layer"] = df["layer"] / MODEL_LAYER_COUNTS.get(
            modelname, max(df["layer"], default=1)
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["manipulated_feature_group"] = combined["manipulated_feature_group"].map(
        lambda val: RENAME_FEATURE_GROUPS.get(val, val)
    )
    return combined


def load_dim_reduction_results(
    results_dir: str = RESULTS_ROOT,
    modelname: str | None = None,
    librispeech_split: str | None = None,
) -> pd.DataFrame:
    """Load dimensionality-reduction sweeps saved as CSVs."""

    pattern = os.path.join(results_dir, "librispeech-*-dimreduction/**/*.csv")
    files = glob.glob(pattern, recursive=True)
    if modelname:
        files = [path for path in files if modelname in path]
    if librispeech_split:
        files = [path for path in files if librispeech_split in path]
    files = [
        path for path in files if "all_layers" not in path and "_normalized" in path
    ]
    if not files:
        raise FileNotFoundError("No dimensionality reduction CSVs found")

    frames: list[pd.DataFrame] = []
    for path in files:
        file_path = Path(path)
        config_dir = file_path.parent.name
        model_dir = file_path.parent.parent.name
        split_dir = file_path.parent.parent.parent.name.replace("-dimreduction", "")
        probename, *_prefix, normalization, dimension_reduction = config_dir.split("_")
        df = pd.read_csv(file_path)
        df["modelname"] = model_dir
        df["librispeech_split"] = split_dir
        df["probename"] = "random_forest" if "random" in probename else probename
        df["normalization"] = normalization
        df["dimensions"] = dimension_reduction.split("-")[-1]
        df["norm_layer"] = df["layer"] / MODEL_LAYER_COUNTS.get(
            model_dir, max(df["layer"], default=1)
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["dimensions"] = combined["dimensions"].replace(
        {"None": "Untruncated PCA", "768": "Original Dimension"}
    )
    dimension_order = [
        *sorted({int(dim) for dim in combined["dimensions"].unique() if dim.isdigit()}),
        "Untruncated PCA",
        "Original Dimension",
        "Original Dimension (StandardScaled)",
        "Original Dimension (Pipeline)",
    ]
    combined["dimensions"] = pd.Categorical(
        combined["dimensions"],
        categories=[str(dim) for dim in dimension_order],
        ordered=True,
    )
    return combined


def _friendly_model_label(modelname: str) -> str:
    if "checkpoint-" in modelname:
        return f"finetuned ({Path(modelname).name})"
    return modelname


def load_sid_decodability_results(
    librispeech_split: str = "train-clean-100",
    results_dir: str = RESULTS_ROOT,
) -> pd.DataFrame:
    """Load layerwise SID decodability metrics produced by `finetune-sid.py`."""

    csv_path = os.path.join(
        results_dir,
        f"speakerid_hiddenstate_decoding_{librispeech_split}.csv",
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SID decodability CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {
        "modelname",
        "librispeech_split",
        "layer",
        "accuracy",
        "f1_weighted",
        "f1_macro",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"SID decodability CSV is missing columns: {sorted(missing)}")

    df["model_display"] = df["modelname"].map(_friendly_model_label)
    return df


def plot_sid_decodability(
    sid_results_df: pd.DataFrame,
    metric: str = "accuracy",
    save: bool = False,
) -> None:
    """Plot layerwise speaker-ID decodability across models."""

    metric_choices = {"accuracy", "f1_weighted", "f1_macro"}
    if metric not in metric_choices:
        raise ValueError(f"metric must be one of {sorted(metric_choices)}")

    plot_df = (
        sid_results_df.groupby(["model_display", "layer"], as_index=False)[metric]
        .mean()
        .copy()
    )
    if plot_df.empty:
        raise ValueError("No SID decodability rows to plot")

    pretty_metric = {
        "accuracy": "Accuracy",
        "f1_weighted": "Weighted F1",
        "f1_macro": "Macro F1",
    }[metric]

    figure = (
        p9.ggplot(plot_df)
        + p9.geom_line(
            p9.aes(
                x="layer",
                y=metric,
                color="model_display",
                shape="model_display",
                group="model_display",
            ),
            alpha=0.8,
        )
        + p9.geom_point(
            p9.aes(
                x="layer",
                y=metric,
                color="model_display",
                shape="model_display",
            ),
            alpha=0.8,
        )
        + p9.scale_x_continuous(breaks=range(0, int(plot_df["layer"].max()) + 1, 3))
        + p9.theme(figure_size=(10, 6), dpi=300)
        + p9.labs(
            x="Layer (from shallow to deep)",
            y=pretty_metric,
            color="Model",
            shape="Model",
            title="Speaker ID Decodability from Hidden States",
        )
    )

    figure.show()
    if save:
        split = sid_results_df["librispeech_split"].iloc[0]
        filename = f"sid_decodability_{split}_{metric}.png"
        figure.save(os.path.join(FIGURES_ROOT, filename))


def _baseline_split(
    df: pd.DataFrame,
    probename: str,
    librispeech_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = df[
        (df["probename"] == probename) & (df["librispeech_split"] == librispeech_split)
    ].copy()
    all_feature = subset[subset["manipulation_mode"] == "none"].copy()
    random_baseline = subset[subset["manipulation_mode"] == "random_baseline"].copy()
    return subset, all_feature, random_baseline


def plot_bottom_up_results(
    df: pd.DataFrame,
    save: bool = False,
    manipulations: Sequence[str] | None = None,
) -> None:
    """Replicate the per-manipulation plots from the legacy script."""

    manipulations = manipulations or ["zeroing", "permutation", "ablation"]
    probenames = df["probename"].unique()
    librispeech_splits = df["librispeech_split"].unique()

    for probename, librispeech_split, manipulation in [
        (probe, split, mode)
        for probe in probenames
        for split in librispeech_splits
        for mode in manipulations
    ]:
        subset_df, all_feature_baseline, random_baseline = _baseline_split(
            df, probename, librispeech_split
        )
        subset_df = subset_df[subset_df["manipulation_mode"] == manipulation].copy()
        if subset_df.empty:
            logger.info(
                "Skipping probe=%s split=%s manipulation=%s (no rows)",
                probename,
                librispeech_split,
                manipulation,
            )
            continue
        subset_df["baseline"] = False
        random_baseline = random_baseline.copy()
        random_baseline["baseline"] = True
        all_feature_baseline = all_feature_baseline.copy()
        all_feature_baseline["baseline"] = True

        figure = (
            p9.ggplot()
            + p9.facet_wrap("~ modelname", ncol=3)
            + p9.geom_line(
                p9.aes(
                    x="norm_layer",
                    y="test_score",
                    color="manipulated_feature_group",
                    shape="manipulated_feature_group",
                    group="manipulated_feature_group",
                ),
                alpha=0.7,
                data=subset_df,
            )
            + p9.geom_point(
                p9.aes(
                    x="norm_layer",
                    y="test_score",
                    color="manipulated_feature_group",
                    shape="manipulated_feature_group",
                ),
                data=subset_df,
            )
            + p9.geom_line(
                p9.aes(x="norm_layer", y="test_score"),
                alpha=0.7,
                data=all_feature_baseline,
                color="black",
                linetype="dotted",
            )
            + p9.theme(figure_size=(10, 10), dpi=300)
            + p9.scale_color_discrete(name=f"{manipulation.capitalize()} Feature Group")
            + p9.scale_shape_discrete(name=f"{manipulation.capitalize()} Feature Group")
            + p9.labs(
                x="Layer (from shallow to deep, normalized)",
                y="Test Score (R²)",
                title="Encoding Probe Performance Across Model Layers",
                subtitle=(
                    f"Probe: {probename}, Librispeech Split: {librispeech_split}, Manipulation: {manipulation}"
                ),
            )
        )

        figure.show()
        if save:
            filename = f"probe_{probename}_librispeech-{librispeech_split}_manipulation-{manipulation}_results.png"
            figure.save(os.path.join(FIGURES_ROOT, filename))


def plot_dim_reduction_grid(df: pd.DataFrame, save: bool = False) -> None:
    """Plot dimensionality reduction sweeps across layers."""

    subset_df = df[
        (df["manipulated_feature_group"] != "none")
        & (df["manipulation_mode"] == "ablation")
        & (df["probename"] == "ridge")
    ].copy()
    if subset_df.empty:
        raise ValueError("Dimensionality reduction DataFrame is empty")

    all_feature_baseline = df[
        (df["manipulated_feature_group"] == "none")
        & (df["manipulation_mode"] == "none")
        & (df["probename"] == "ridge")
    ].copy()

    figure = (
        p9.ggplot(subset_df)
        + p9.facet_wrap("~ dimensions", ncol=3)
        + p9.geom_line(
            p9.aes(
                x="layer",
                y="test_score",
                color="manipulated_feature_group",
                shape="manipulated_feature_group",
                group="manipulated_feature_group",
            ),
            alpha=0.7,
        )
        + p9.geom_point(
            p9.aes(
                x="layer",
                y="test_score",
                color="manipulated_feature_group",
                shape="manipulated_feature_group",
            ),
            data=subset_df,
        )
        + p9.geom_line(
            p9.aes(x="layer", y="test_score"),
            alpha=0.7,
            data=all_feature_baseline,
            color="black",
            linetype="dotted",
        )
        + p9.scale_color_discrete(name="Manipulated Feature Group")
        + p9.scale_shape_discrete(name="Manipulated Feature Group")
        + p9.scale_x_continuous(breaks=range(0, subset_df["layer"].max() + 1, 3))
        + p9.theme(figure_size=(10, 10), dpi=300)
        + p9.labs(
            x="Layer (from shallow to deep)",
            y="Test Score (R²)",
            title=(
                "Encoding Probe Performance Across Model Layers with Dimensionality Reduction"
            ),
        )
    )

    figure.show()
    if save:
        figure.save(os.path.join(FIGURES_ROOT, "dimensionality_reduction_grid.png"))


def plot_encode_decode_comparison(
    bottom_up_results: pd.DataFrame,
    librispeech_split: str,
    modelname: str,
    save: bool = False,
) -> None:
    """Compare encoding vs decoding probes for a single model."""

    decoding_file = os.path.join(
        RESULTS_ROOT,
        f"decoding_frame_probe_results_{modelname}_{librispeech_split}.csv",
    )
    if not os.path.exists(decoding_file):
        raise FileNotFoundError(f"Missing decoding probe CSV: {decoding_file}")

    decoding_df = pd.read_csv(decoding_file).rename(columns={"score": "test_score"})
    decoding_df["Probing_Direction"] = "Decoding Probe"

    subset_df = bottom_up_results[
        (bottom_up_results["modelname"].str.endswith(modelname))
        & (bottom_up_results["librispeech_split"] == librispeech_split)
        & (bottom_up_results["manipulation_mode"] == "bottom-up")
    ].copy()
    if subset_df.empty:
        raise ValueError("No bottom-up encoding probe rows for the requested model")

    baselines = subset_df[
        subset_df["feature_group"].isin(["all", "random_baseline"])
    ].copy()
    subset_df = subset_df[
        ~subset_df["feature_group"].isin(["all", "random_baseline"])
    ].copy()
    subset_df["Probing_Direction"] = "Encoding Probe"

    comparison_df = pd.concat([subset_df, decoding_df], ignore_index=True)
    comparison_df["feature_group"] = comparison_df["feature_group"].map(
        lambda val: RENAME_FEATURE_GROUPS.get(val, val)
    )

    plot = (
        p9.ggplot(comparison_df)
        + p9.geom_line(
            p9.aes(
                x="layer",
                y="test_score",
                color="feature_group",
                shape="feature_group",
                group="feature_group",
            ),
            alpha=0.7,
        )
        + p9.geom_point(
            p9.aes(
                x="layer",
                y="test_score",
                color="feature_group",
                shape="feature_group",
                group="feature_group",
            ),
            alpha=0.7,
        )
        + p9.facet_wrap("~ Probing_Direction", ncol=2, scales="free_y")
        + p9.geom_line(
            p9.aes(x="layer", y="test_score"),
            alpha=0.7,
            data=baselines[baselines["feature_group"] == "random_baseline"],
            color="black",
            linetype="dashed",
        )
        + p9.geom_line(
            p9.aes(x="layer", y="test_score"),
            alpha=0.7,
            data=baselines[baselines["feature_group"] == "all"],
            color="black",
            linetype="dotted",
        )
        + p9.scale_color_discrete(name="Feature Group")
        + p9.scale_shape_discrete(name="Feature Group")
        + p9.scale_x_continuous(breaks=range(0, comparison_df["layer"].max() + 1, 3))
        + p9.theme(figure_size=(10, 6), dpi=300)
        + p9.labs(
            x="Layer (from shallow to deep)",
            y="Test Score (R²)",
            title=(
                f"Encoding vs Decoding Probe Performance for {modelname} on {librispeech_split}"
            ),
        )
    )

    plot.show()
    if save:
        filename = f"encode_decode_comparison_{modelname}_{librispeech_split}.png"
        plot.save(os.path.join(FIGURES_ROOT, filename))


def plot_ppg_feature_representation(
    bottom_up_results: pd.DataFrame,
    librispeech_split: str,
    modelname: str,
    save: bool = False,
) -> None:
    """Compare different PPG feature encodings."""

    subset_df = bottom_up_results[
        (bottom_up_results["modelname"].str.endswith(modelname))
        & (bottom_up_results["librispeech_split"] == librispeech_split)
        & (bottom_up_results["feature_group"].str.contains("ppg", case=False, na=False))
    ].copy()
    if subset_df.empty:
        raise ValueError("No PPG rows for the requested model")

    subset_df["feature_group"] = subset_df["feature_group"].map(
        lambda val: RENAME_FEATURE_GROUPS.get(val, val)
    )

    plot = (
        p9.ggplot(subset_df)
        + p9.geom_line(
            p9.aes(
                x="layer",
                y="test_score",
                color="feature_group",
                shape="feature_group",
                group="feature_group",
            ),
            alpha=0.7,
        )
        + p9.geom_point(
            p9.aes(
                x="layer",
                y="test_score",
                color="feature_group",
                shape="feature_group",
                group="feature_group",
            ),
            alpha=0.7,
        )
        + p9.scale_x_continuous(breaks=range(0, subset_df["layer"].max() + 1, 3))
        + p9.theme(figure_size=(8, 6), dpi=300)
        + p9.labs(
            x="Layer (from shallow to deep)",
            y="Test Score (R²)",
            title=(
                f"Encoding Probe Performance on PPG Representations\nAcross {modelname} ({librispeech_split})"
            ),
        )
    )

    plot.show()
    if save:
        filename = f"ppg_feature_representation_{modelname}_{librispeech_split}.png"
        plot.save(os.path.join(FIGURES_ROOT, filename))


def plot_permutation_results(
    probe: str = "ridge",
    librispeech_split: str = "train-clean-100",
    save: bool = False,
) -> None:
    """Port the permutation visualizations from analyze_results.py."""

    csv_file = os.path.join(
        RESULTS_ROOT,
        f"{probe}_{librispeech_split}_permutations.csv",
    )
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Permutation CSV not found: {csv_file}")

    df = pd.read_csv(csv_file, sep=";")
    df["mode"] = df["mode"].fillna("None")
    topline_df = df[df["mode"] == "None"].copy()
    df = df[df["mode"] != "None"].copy()

    probe_name_lookup = {
        "ridge": "Ridge",
        "rf": "Random Forest",
        "kernel_ridge": "Kernel Ridge",
        "MLPRegressor": "MLP",
        "LinearRegression": "Linear Regression",
        "Lasso": "Lasso",
    }

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.lineplot(
        data=topline_df, x="layer", y="r^2_score", label="Test R² Score", color="blue"
    )
    ax2 = plt.twinx()
    topline_df["train_test_diff"] = (
        topline_df["train_r^2_score"] - topline_df["r^2_score"]
    ) / topline_df["train_r^2_score"]
    sns.lineplot(
        data=topline_df,
        x="layer",
        y="train_test_diff",
        color="red",
        ax=ax2,
        label="Train-Test Difference %",
    )
    plt.title(
        f"Topline Results for {topline_df['pred_representation'].unique()[0]} on {probe_name_lookup.get(probe, probe)}"
    )
    plt.xlabel("Layer")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        topline_path = os.path.join(
            FIGURES_ROOT,
            f"{librispeech_split}_{topline_df['pred_representation'].unique()[0]}_{probe}_topline_results.png",
        )
        plt.savefig(topline_path, dpi=300, bbox_inches="tight")
    plt.show()

    for mode in df["mode"].unique():
        plotting_df = df[df["mode"] == mode]
        g = sns.FacetGrid(
            plotting_df[plotting_df["input_ablation"] != "None"],
            col="input_ablation",
            hue="input_ablation",
            col_wrap=3,
            height=4,
            aspect=1.2,
            sharey=True,
            sharex=True,
        )
        g.map_dataframe(
            sns.barplot,
            x="layer",
            y="r^2_score_decrease",
            order=sorted(plotting_df["layer"].unique()),
        )
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Layer", "R² Score Decrease")
        g.fig.suptitle(
            f"R² Score Decrease on {mode} (Probe={probe_name_lookup.get(probe, probe)})",
            fontsize=16,
            fontweight="bold",
            y=1.05,
        )
        plt.tight_layout()
        if save:
            fig_path = os.path.join(
                FIGURES_ROOT,
                f"{librispeech_split}_{plotting_df['pred_representation'].unique()[0]}_{probe}_{mode}_results.png",
            )
            g.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()


def run_feature_removal(split: str) -> None:
    all_results = read_all_results(split)
    mode = "top-down"
    plotting_configs = [
        "syntax_lexical",
        "syntax_lexical_2",
        "acoustics_speaker_id",
        "phonetic_speaker_id",
        "acoustics_speaker_id_2",
        "phonetic_speaker_id_2",
        "all_models_syntax_lexical",
        "all_models_acoustic_speaker",
        "all_models_phonetic_speaker",
        "syntax_lexical_wav2vec2",
        "acoustics_speaker_id_wav2vec2",
        "phonetic_speaker_id_wav2vec2",
        "syntax_lexicon_decomposition_wav2vec2",
    ]
    expected_files = [
        _build_main_figure_filename(
            mode=mode,
            featname=featname,
            librispeech_split=split,
        )
        for featname in plotting_configs
    ]
    logger.info("Expected parse_results naming for main figures: %s", expected_files)
    plot_main_figures(all_results, librispeech_split=split)


def run_random_seed_focus(split: str, modelname: str, focuses: Sequence[str]) -> None:
    focus_targets = list(focuses) or ["syntax_lexical"]
    random_seed_results = read_random_seed_results(
        librispeech_split=split, modelname=modelname
    )
    for focus in focus_targets:
        display_modelname = random_seed_results["modelname"].iloc[0].split(": ")[-1]
        expected_file = _build_random_seed_figure_filename(
            modelname=display_modelname,
            focus=focus,
            librispeech_split=split,
        )
        logger.info(
            "Expected parse_results naming for random-seed figure: %s",
            expected_file,
        )
        plot_focus_random_seed(
            random_seed_results,
            focus=focus,
            librispeech_split=split,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified visualization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    feature_parser = subparsers.add_parser(
        "feature-removal",
        help="Reproduce the top-down ablation plots from parse_results",
    )
    feature_parser.add_argument("--split", default="dev-clean")

    random_seed_parser = subparsers.add_parser(
        "random-seed", help="Plot random-seed variability for a specific model"
    )
    random_seed_parser.add_argument("--split", default="train-clean-100")
    random_seed_parser.add_argument("--model", required=True)
    random_seed_parser.add_argument(
        "--focus",
        nargs="*",
        default=["syntax_lexical"],
        choices=["syntax_lexical", "acoustic_speaker", "phonetic_speaker"],
    )

    bottom_up_parser = subparsers.add_parser(
        "bottom-up", help="Plot encoding probe results from CSV dumps"
    )
    bottom_up_parser.add_argument("--results-dir", default=RESULTS_ROOT)
    bottom_up_parser.add_argument("--save", action="store_true")
    bottom_up_parser.add_argument(
        "--manipulations",
        nargs="*",
        default=["zeroing", "permutation", "ablation"],
    )

    dim_parser = subparsers.add_parser(
        "dim-reduction", help="Plot dimensionality reduction sweeps"
    )
    dim_parser.add_argument("--model", default="wav2vec2-base")
    dim_parser.add_argument("--split", default="train-clean-100")
    dim_parser.add_argument("--save", action="store_true")

    encode_decode_parser = subparsers.add_parser(
        "encode-decode", help="Compare encoding and decoding probes"
    )
    encode_decode_parser.add_argument("--model", default="wav2vec2-base")
    encode_decode_parser.add_argument("--split", default="train-clean-100")
    encode_decode_parser.add_argument("--save", action="store_true")

    ppg_parser = subparsers.add_parser(
        "ppg", help="Plot PPG representation comparisons"
    )
    ppg_parser.add_argument("--model", default="wav2vec2-base")
    ppg_parser.add_argument("--split", default="train-clean-100")
    ppg_parser.add_argument("--save", action="store_true")

    permutation_parser = subparsers.add_parser(
        "permutation", help="Plot permutation-mode CSV summaries"
    )
    permutation_parser.add_argument("--probe", default="ridge")
    permutation_parser.add_argument("--split", default="train-clean-100")
    permutation_parser.add_argument("--save", action="store_true")

    sid_parser = subparsers.add_parser(
        "sid-decodability",
        help="Plot layerwise speaker-ID decodability from hidden states",
    )
    sid_parser.add_argument("--split", default="train-clean-100")
    sid_parser.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "f1_weighted", "f1_macro"],
    )
    sid_parser.add_argument("--save", action="store_true")

    manipulation_parser = subparsers.add_parser(
        "manipulation-comparison",
        help=(
            "Compare ablation vs shuffle/zero controls for a feature block "
            "across models. Loads separate manipulation pickles produced by "
            "experiment_pipeline --manipulation and overlays them with the "
            "AllFeatures topline."
        ),
    )
    manipulation_parser.add_argument("--split", default="train-clean-100")
    manipulation_parser.add_argument(
        "--models",
        nargs="+",
        default=["facebook/wav2vec2-base", "techsword/wav2vec2-ls100-sid"],
        help="Full HuggingFace model names to compare.",
    )
    manipulation_parser.add_argument("--probe", default="ridge")
    manipulation_parser.add_argument(
        "--experiment-subdir",
        default="speakerid_shuffle_comparison",
        help="Results subdirectory / experiment name.",
    )
    manipulation_parser.add_argument(
        "--manipulations",
        nargs="+",
        default=["drop", "shuffle"],
        choices=["drop", "shuffle", "zero"],
        help="Manipulation modes to load and compare.",
    )
    manipulation_parser.add_argument(
        "--target-configs",
        nargs="+",
        default=None,
        help=(
            "Raw config_name values to include (e.g. 'SpeakerID-OH+eGeMAPSv02'). "
            "When unset, all non-topline configs in the results are plotted."
        ),
    )
    manipulation_parser.add_argument(
        "--y-col",
        default="test_score",
        choices=["test_score", "unexplained_variance"],
        help="Y-axis column to plot.",
    )
    manipulation_parser.add_argument("--random-seed", type=int, default=42)
    manipulation_parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Override the results root directory. When unset, uses the "
            "auto-detected RESULTS_ROOT (results/ locally, "
            "/projects/prjs1586/experimental_results on Snellius). Use this "
            "when results have been copied to a non-default location."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "feature-removal":
        run_feature_removal(args.split)
    elif args.command == "random-seed":
        run_random_seed_focus(args.split, args.model, args.focus)
    elif args.command == "bottom-up":
        df = load_layer_csvs(results_dir=args.results_dir)
        plot_bottom_up_results(df, save=args.save, manipulations=args.manipulations)
    elif args.command == "dim-reduction":
        df = load_dim_reduction_results(
            modelname=args.model, librispeech_split=args.split
        )
        plot_dim_reduction_grid(df, save=args.save)
    elif args.command == "encode-decode":
        df = load_layer_csvs()
        plot_encode_decode_comparison(
            df, librispeech_split=args.split, modelname=args.model, save=args.save
        )
    elif args.command == "ppg":
        df = load_layer_csvs()
        plot_ppg_feature_representation(
            df, librispeech_split=args.split, modelname=args.model, save=args.save
        )
    elif args.command == "permutation":
        plot_permutation_results(
            probe=args.probe, librispeech_split=args.split, save=args.save
        )
    elif args.command == "sid-decodability":
        sid_df = load_sid_decodability_results(librispeech_split=args.split)
        plot_sid_decodability(sid_df, metric=args.metric, save=args.save)
    elif args.command == "manipulation-comparison":
        comparison_df = read_manipulation_comparison_results(
            librispeech_split=args.split,
            modelnames=args.models,
            probename=args.probe,
            experiment_subdir=args.experiment_subdir,
            manipulations=args.manipulations,
            random_seed=args.random_seed,
            results_root=args.results_root,
        )
        plot_manipulation_comparison(
            comparison_df,
            target_configs=args.target_configs,
            librispeech_split=args.split,
            y_col=args.y_col,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
