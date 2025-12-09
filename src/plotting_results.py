import glob
import logging
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

# Get the hostname of the machine running the code
hostname = os.uname().nodename
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


if "snellius" in hostname:
    # If running on Snellius, use the Snellius dataset root
    DATASET_ROOT = os.path.realpath("/projects/prjs1586/corpora/LibriSpeech")
    ALIGNMENT_ROOT = DATASET_ROOT.replace("LibriSpeech", "librispeech_textgrids")
    SAVEPATH = "/projects/prjs1586/experimental_data"
    RESULTS_ROOT = "/projects/prjs1586/experimental_results"
    FIGURES_ROOT = os.path.join(PROJECT_ROOT, "figures")

else:
    # If running on local machine, use the local dataset root
    DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENT_ROOT = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENT_ROOT = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
    FIGURES_ROOT = os.path.join(PROJECT_ROOT, "figures")

model_layer_dict = {
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

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_result_files(
    results_dir: str, results_file_pattern: str = "librispeech-*/**/layer_*.csv"
) -> pd.DataFrame:
    # logger.info(f"Loading result files from {results_dir}...")
    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} does not exist.")
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    # all_results_files = glob.glob(os.path.join(results_dir, "*.csv"))
    all_results_files = glob.glob(
        os.path.join(results_dir, results_file_pattern), recursive=True
    )
    all_results_files = [
        x
        for x in all_results_files
        if "all_layers" not in x and "_normalized" in x and "dimreduction" not in x
    ]

    all_results_df = pd.DataFrame()
    for result_file in all_results_files:
        # logger.info(f"Found results file: {result_file}")
        librispeech_split = result_file.split("/")[-4]
        modelname = result_file.split("/")[-3]
        probename, _, _, normalization = result_file.split("/")[-2].split("_")
        if "random" in probename:
            probename = "random_forest"

        # logger.info(
        #     f"Parsed - Librispeech Split: {librispeech_split}, Model Name: {modelname}, Probe Name: {probename}"
        # )
        df = pd.read_csv(result_file)
        df["modelname"] = (
            "Text: " + modelname
            if ("wav" not in modelname) and ("hubert" not in modelname)
            else "Audio: " + modelname
        )
        df["librispeech_split"] = librispeech_split
        df["probename"] = probename
        df["normalization"] = normalization

        df["norm_layer"] = df["layer"] / model_layer_dict.get(modelname, 1)
        all_results_df = pd.concat([all_results_df, df], ignore_index=True)

    return all_results_df


def load_dimreduction_files(results_dir: str):
    results_dir = RESULTS_ROOT
    # logger.info(f"Loading result files from {results_dir}...")
    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} does not exist.")
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    all_results_files = glob.glob(
        os.path.join(results_dir, "librispeech-*-dimreduction/**/*.csv"), recursive=True
    )
    modelname = "wav2vec2-base"
    # modelname = "bert-base-uncased"
    # modelname = "ModernBERT-base"

    all_results_files = [
        x
        for x in all_results_files
        if "all_layers" not in x and "_normalized" in x and modelname in x
    ]

    all_results_df = pd.DataFrame()
    for result_file in all_results_files:
        # logger.info(f"Found results file: {result_file}")
        librispeech_split = result_file.split("/")[-4].replace("-dimreduction", "")
        modelname = result_file.split("/")[-3]
        probename, _, _, normalization, dimension_reduction = result_file.split("/")[
            -2
        ].split("_")
        if "random" in probename:
            probename = "random-forest"

        # logger.info(
        #     f"Parsed - Librispeech Split: {librispeech_split}, Model Name: {modelname}, Probe Name: {probename}"
        # )
        df = pd.read_csv(result_file)
        df["modelname"] = modelname
        df["librispeech_split"] = librispeech_split
        df["probename"] = probename
        df["normalization"] = normalization
        df["dimensions"] = dimension_reduction.split("-")[-1]
        df["norm_layer"] = df["layer"] / model_layer_dict.get(modelname, 1)
        all_results_df = pd.concat([all_results_df, df], ignore_index=True)

    # Rename None to Untruncated PCA
    all_results_df["dimensions"] = all_results_df["dimensions"].replace(
        "None", "Untruncated PCA"
    )
    # Replace 768 with "original dimension"
    all_results_df["dimensions"] = all_results_df["dimensions"].replace(
        "768", "Original Dimension"
    )

    # Select only with probename == "ridge-transformed-target"
    all_results_df = all_results_df[
        all_results_df["probename"] == "ridge"  # -transformed-target"
    ].copy()

    # # change dimensions if probename == "ridge-transformed-target"
    # all_results_df.loc[
    #     all_results_df["probename"] == "ridge-transformed-target", "dimensions"
    # ] = "Original Dimension (Pipeline)"

    # Set manual order of dimensions so that numeric dimensions are in ascending order followed by Untruncated PCA and Original Dimension
    dimension_order = sorted(
        [int(dim) for dim in all_results_df["dimensions"].unique() if dim.isdigit()]
    ) + [
        "Untruncated PCA",
        "Original Dimension",
        "Original Dimension (StandardScaled)",
        "Original Dimension (Pipeline)",
    ]
    dimension_order = [str(dim) for dim in dimension_order]
    all_results_df["dimensions"] = pd.Categorical(
        all_results_df["dimensions"], categories=dimension_order, ordered=True
    )

    # Plot all of the results together with facet wrap on dimensions
    subset_df = all_results_df.copy()
    subset_df = subset_df[
        (subset_df["manipulated_feature_group"] != "none")
        & (subset_df["manipulation_mode"] == "ablation")
    ]
    all_feature_baseline = all_results_df[
        (all_results_df["manipulated_feature_group"] == "none")
        & (all_results_df["manipulation_mode"] == "none")
    ].copy()
    random_baseline = all_results_df[
        (all_results_df["manipulated_feature_group"] == "none")
        & (all_results_df["manipulation_mode"] == "random_baseline")
    ].copy()

    x_var, y_var = "layer", "test_score"

    figure = (
        p9.ggplot(subset_df)
        + p9.facet_wrap("~ dimensions", ncol=3)
        + p9.geom_line(
            p9.aes(
                x=x_var,
                y=y_var,
                color="manipulated_feature_group",
                shape="manipulated_feature_group",
                group="manipulated_feature_group",
            ),
            alpha=0.7,
        )
        + p9.geom_point(
            p9.aes(
                x=x_var,
                y=y_var,
                color="manipulated_feature_group",
                shape="manipulated_feature_group",
            ),
            data=subset_df,
        )
        # Add the baselines with distinct linetypes and colors for clarity
        + p9.geom_line(
            p9.aes(x=x_var, y=y_var),
            alpha=0.7,
            data=random_baseline,
            color="black",
            linetype="dashed",
        )
        + p9.geom_line(
            p9.aes(
                x=x_var,
                y=y_var,
            ),
            alpha=0.7,
            data=all_feature_baseline,
            color="black",
            linetype="dotted",
        )
        + p9.scale_color_discrete(name="Manip. Feat. Grp")
        + p9.scale_shape_discrete(name="Manip. Feat. Grp")
        # Put the x-axis ticks from 0 to max layer for every 3rd layer
        + p9.scale_x_continuous(breaks=range(0, subset_df["layer"].max() + 1, 3))
        + p9.theme(
            figure_size=(10, 10),
            dpi=300,
            plot_caption=p9.element_text(ha="left", margin={"t": 1, "units": "lines"}),
        )
        + p9.labs(
            x="Layer (from shallow to deep)",
            y="Test Score (R²)",
            title=f"Encoding Probe Performance Across Model Layers with Dimensionality Reduction on {modelname}",
        )
    )
    figure.show()


caption = """\
Baselines:
- Black dashed line: Random Baseline (random_baseline)
- Black dotted line: All Feature Baseline (all_feat_baseline)
All feature baseline is the best case results with all features intact. The feature group manipulations should be compared against these baselines.
"""


bottom_up_caption = """\
Baselines:
- Black dotted line: All Feature Baseline
All feature baseline is the best case results with all features groups present. 
The other curves are the results with JUST one feature group present (bottom-up).
"""


def plot_results(
    subset_df,
    random_baseline,
    all_feature_baseline,
    probename,
    librispeech_split,
    manipulation,
) -> p9.ggplot:
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
        # Add the baselines with distinct linetypes and colors for clarity
        + p9.geom_line(
            p9.aes(x="norm_layer", y="test_score"),
            alpha=0.7,
            data=random_baseline,
            color="black",
            linetype="dashed",
        )
        + p9.geom_line(
            p9.aes(
                x="norm_layer",
                y="test_score",
            ),
            alpha=0.7,
            data=all_feature_baseline,
            color="black",
            linetype="dotted",
        )
        + p9.theme(
            figure_size=(10, 10),
            dpi=300,
            plot_caption=p9.element_text(ha="left", margin={"t": 1, "units": "lines"}),
        )
        + p9.scale_color_discrete(name=f"{manipulation.capitalize()} Feature Group")
        + p9.scale_shape_discrete(name=f"{manipulation.capitalize()} Feature Group")
        + p9.labs(
            x="Layer (from shallow to deep, normalized)",
            y="Test Score (R²)",
            title="Encoding Probe Performance Across Model Layers",
            subtitle=f"Probe: {probename}, Librispeech Split: {librispeech_split} Manipulation: {manipulation}",
            caption=caption,
        )
    )

    return figure


def plot_all_results(all_results_df: pd.DataFrame, save=False) -> None:
    """Plot the results from the joined DataFrame."""

    logger.info("Plotting results...")

    # Split into subsets based on probe name
    probenames = all_results_df["probename"].unique()
    librispeech_splits = all_results_df["librispeech_split"].unique()
    manipulations = ["zeroing", "permutation", "ablation"]

    # # Normalize the layer numbers to start from 0 and end at max layer for each model
    # all_results_df["norm_layer"] = all_results_df.groupby("modelname")[
    #     "layer"
    # ].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    for probename, librispeech_split, manipulation in product(
        probenames, librispeech_splits, manipulations
    ):
        subset_df = all_results_df[
            (all_results_df["probename"] == probename)
            & (all_results_df["librispeech_split"] == librispeech_split)
            & (all_results_df["manipulation_mode"] == manipulation)
        ].copy()

        all_feature_baseline = all_results_df[
            (all_results_df["probename"] == probename)
            & (all_results_df["librispeech_split"] == librispeech_split)
            & (all_results_df["manipulation_mode"] == "none")
        ].copy()
        random_baseline = all_results_df[
            (all_results_df["probename"] == probename)
            & (all_results_df["librispeech_split"] == librispeech_split)
            & (all_results_df["manipulation_mode"] == "random_baseline")
        ].copy()

        random_baseline["manipulated_feature_group"] = "random_baseline"
        all_feature_baseline["manipulated_feature_group"] = "all_feat_baseline"
        random_baseline["baseline"] = True
        all_feature_baseline["baseline"] = True
        subset_df["baseline"] = False

        if subset_df.empty:
            logger.warning(
                f"No data for probe {probename} and split {librispeech_split} and manipulation {manipulation}. Skipping."
            )
            continue
        figure = plot_results(
            subset_df,
            random_baseline,
            all_feature_baseline,
            probename,
            librispeech_split,
            manipulation,
        )

        figure.show()

        if save:
            figure_filename = f"probe_{probename}_librispeech-{librispeech_split}_manipulation-{manipulation}_results.png"
            figure_path = os.path.join(FIGURES_ROOT, figure_filename)
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            figure.save(figure_path)
            logger.info(f"Saved figure to {figure_path}")
        logger.info("Finished plotting results.")

    # Also plot subset of wav2vec2-base results for quick inspection
    subset_df = all_results_df[
        (all_results_df["modelname"] == "Audio: wav2vec2-base")
        & (all_results_df["librispeech_split"] == "librispeech-train-clean-100")
        & (all_results_df["probename"] == "ridge")
        & (all_results_df["manipulation_mode"] == "ablation")
    ].copy()
    all_feature_baseline = all_results_df[
        (all_results_df["modelname"] == "Audio: wav2vec2-base")
        & (all_results_df["probename"] == "ridge")
        & (all_results_df["librispeech_split"] == "librispeech-train-clean-100")
        & (all_results_df["manipulation_mode"] == "none")
    ].copy()
    random_baseline = all_results_df[
        (all_results_df["modelname"] == "Audio: wav2vec2-base")
        & (all_results_df["probename"] == "ridge")
        & (all_results_df["librispeech_split"] == "librispeech-train-clean-100")
        & (all_results_df["manipulation_mode"] == "random_baseline")
    ].copy()
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
        # Add the baselines with distinct linetypes and colors for clarity
        + p9.geom_line(
            p9.aes(x="norm_layer", y="test_score"),
            alpha=0.7,
            data=random_baseline,
            color="black",
            linetype="dashed",
        )
        + p9.geom_line(
            p9.aes(
                x="norm_layer",
                y="test_score",
            ),
            alpha=0.7,
            data=all_feature_baseline,
            color="black",
            linetype="dotted",
        )
        + p9.theme(
            figure_size=(10, 6),
            dpi=300,
            plot_caption=p9.element_text(ha="left", margin={"t": 1, "units": "lines"}),
        )
        + p9.scale_color_discrete(name="Ablation Feature Group")
        + p9.scale_shape_discrete(name="Ablation Feature Group")
        + p9.labs(
            x="Layer (from shallow to deep, normalized)",
            y="Test Score (R²)",
            title="Encoding Probe Performance Across Model Layers",
            subtitle="Probe: ridge, Librispeech Split: librispeech-train-clean-100, Manipulation: ablation",
            caption=caption,
        )
    )
    figure.show()
    figure.save(
        os.path.join(
            FIGURES_ROOT,
            "probe_ridge_librispeech-train-clean-100_manipulation-ablation_wav2vec2-base_results.png",
        )
    )

    # Also plot subset of wav2vec2-base results for quick inspection
    subset_df = all_results_df[
        (all_results_df["modelname"] == "Audio: wav2vec2-base")
        & (all_results_df["librispeech_split"] == "librispeech-train-clean-100")
        & (all_results_df["probename"] == "random-forest")
        # & (all_results_df["manipulation_mode"] == "ablation")
    ].copy()
    if not subset_df.empty:
        all_feature_baseline = all_results_df[
            (all_results_df["modelname"] == "Audio: wav2vec2-base")
            & (all_results_df["probename"] == "random-forest")
            & (all_results_df["librispeech_split"] == "librispeech-train-clean-100")
            & (all_results_df["manipulation_mode"] == "none")
        ].copy()
        random_baseline = all_results_df[
            (all_results_df["modelname"] == "Audio: wav2vec2-base")
            & (all_results_df["probename"] == "random-forest")
            & (all_results_df["librispeech_split"] == "librispeech-train-clean-100")
            & (all_results_df["manipulation_mode"] == "random_baseline")
        ].copy()
        figure = plot_results(
            subset_df,
            random_baseline,
            all_feature_baseline,
            "random-forest",
            "librispeech-train-clean-100",
            "ablation",
        )
        figure + p9.theme(figure_size=(6, 6))
        figure.show()


def plot_coefficients(coefficients):
    pass
    # Regressor coefficients sanity check visualization
    # First we aggregate the coefficients for each feature group

    coefficient_dict = {}
    sections_shapes = (
        (0, 125, "acoustic"),
        (
            125,
            100 + 125,
            "word_embedding",
        ),
        (
            100 + 125,
            100 + 125 + 8,
            "syntax_features",
        ),
        (
            100 + 125 + 8,
            100 + 125 + 8 + 40,
            "ppgs_features",
        ),
        (
            100 + 125 + 8 + 40,
            100 + 125 + 8 + 40 + 100,
            "spk_embedding",
        ),
        (
            100 + 125 + 8 + 40 + 100,
            100 + 125 + 8 + 40 + 100 + 2,
            "metadata",
        ),
    )

    for range_start, range_end, name in sections_shapes:
        print(
            f"Feature group: {name}, Coefficient mean: {np.mean(np.abs(coefficients[:, range_start:range_end]))}"
        )
        coefficient_dict[name] = np.mean(np.abs(coefficients[:, range_start:range_end]))

    # Extract coefficients for each feature group using sections_shapes
    acoustic_features = coefficients[:, sections_shapes[0][0] : sections_shapes[0][1]]
    word_embedding_features = coefficients[
        :, sections_shapes[1][0] : sections_shapes[1][1]
    ]
    syntax_features = coefficients[:, sections_shapes[2][0] : sections_shapes[2][1]]
    ppgs_features = coefficients[:, sections_shapes[3][0] : sections_shapes[3][1]]
    spk_embedding_features = coefficients[
        :, sections_shapes[4][0] : sections_shapes[4][1]
    ]
    metadata_features = coefficients[:, sections_shapes[5][0] : sections_shapes[5][1]]
    # Sum the coefficients for each feature group
    acoustic_features = np.mean(acoustic_features, axis=1)
    word_embedding_features = np.mean(word_embedding_features, axis=1)
    # phone_embedding_features = np.mean(phone_embedding_features, axis=1)
    metadata_features = np.mean(metadata_features, axis=1)
    syntax_features = np.mean(syntax_features, axis=1)
    ppgs_features = np.mean(ppgs_features, axis=1)
    spk_embedding_features = np.mean(spk_embedding_features, axis=1)

    stacked_aggrgegated = np.stack(
        (
            acoustic_features,
            word_embedding_features,
            # phone_embedding_features,
            metadata_features,
            syntax_features,
            ppgs_features,
            spk_embedding_features,
        ),
        axis=0,
    )

    columns = [
        "Acoustic Features",
        "Word Embedding",
        "Metadata",
        "Syntax Features",
        "PPGs Features",
        "Speaker Embedding",
    ]
    # plot the coefficients in heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        stacked_aggrgegated.mean(-1).T, cmap="coolwarm", annot=True, xticklabels=columns
    )
    plt.title("Feature Correlation Matrix")
    plt.show()
    return coefficient_dict


def plot_encode_decode_comparison(
    librispeech_split="train-clean-100", modelname="wav2vec2-base"
):
    bottom_up_results_df = load_result_files(
        results_dir=RESULTS_ROOT,
        results_file_pattern=f"librispeech-{librispeech_split}/{modelname}/ridge_frame_probe_normalized/bottom-up-layer_*.csv",
    )
    rename_feature_group = {
        "SpectralInfo": "Spectral Features",
        "Formant Characteristics": "Formants",
        "syntax_feature": "Syntactic Features",
        "ppg_feature": "PPG",
        "spk_embedding": "Speaker Embedding",
        "metadata": "Metadata",
        "dnn_word_embedding": "DNN Word Embedding",
        "OtherAcoustic": "Other Acoustic Features",
        "Formants": "Formants",
        "eGeMAPSv02": "eGeMAPSv02",
    }
    decoding_probe_results = pd.read_csv(
        f"/home/gshen/work_dir/unprobe/results/decoding_frame_probe_results_{modelname}_{librispeech_split}.csv"
    )
    # rename r2_score to test_score
    decoding_probe_results = decoding_probe_results.rename(
        columns={"r2_score": "test_score"}
    )
    decoding_probe_results["Probing_Direction"] = "Decoding Probe"

    bottom_up_results_df["Probing_Direction"] = "Encoding Probe"
    bottom_up_baselines = bottom_up_results_df[
        (bottom_up_results_df["feature_group"] == "all")
        # | (bottom_up_results_df["feature_group"] == "random_baseline")
    ]
    bottom_up_results_df = bottom_up_results_df[
        (bottom_up_results_df["feature_group"] != "all")
        & (bottom_up_results_df["feature_group"] != "random_baseline")
    ]
    bottom_up_results_df["feature_group"] = bottom_up_results_df["feature_group"].map(
        lambda x: rename_feature_group.get(x, x)
    )
    # Join the decoding probe results with bottom-up results
    comparison_results_df = pd.concat(
        [bottom_up_results_df, decoding_probe_results]
    ).reset_index(drop=True)

    # Drop columns with any NaN values
    comparison_results_df = comparison_results_df.dropna(axis=1, how="any")

    # Plot the bottom-up results with a facet_wrap on Probing Direction
    plot = (
        p9.ggplot(comparison_results_df)
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
        # Add the baselines with distinct linetypes and colors for clarity
        + p9.geom_line(
            p9.aes(x="layer", y="test_score"),
            alpha=0.7,
            data=bottom_up_baselines[
                bottom_up_baselines["feature_group"] == "random_baseline"
            ],
            color="black",
            linetype="dashed",
        )
        + p9.geom_line(
            p9.aes(
                x="layer",
                y="test_score",
            ),
            alpha=0.7,
            data=bottom_up_baselines[bottom_up_baselines["feature_group"] == "all"],
            color="black",
            linetype="dotted",
        )
        + p9.labs(
            x="Layer (from shallow to deep)",
            y="Test Score (R²)",
            title=f"Direct Comparison between Encoding Probe and Decoding Probe Performance\nAcross {modelname} Model Layers ({librispeech_split})",
            caption=bottom_up_caption,
        )
        # Rename legend
        + p9.scale_color_discrete(name="Feature Group")
        + p9.scale_shape_discrete(name="Feature Group")
        # Break the x axis at intervals of 3
        + p9.scale_x_continuous(
            breaks=range(0, comparison_results_df["layer"].max() + 1, 3)
        )
        + p9.theme(
            figure_size=(10, 6),
            dpi=300,
            plot_caption=p9.element_text(ha="left", margin={"t": 1, "units": "lines"}),
        )
    )

    plot.show()


if __name__ == "__main__":
    all_results_df = load_result_files(results_dir=RESULTS_ROOT)
    # # Remove bert-base-uncased from the results
    # all_results_df = all_results_df[
    #     all_results_df["modelname"] != "Text: bert-base-uncased"
    # ]

    rename_feature_group = {
        "Prosodic & Voice Quality": "Prosodic Features",
        "Spectral Envelope": "Spectral Features",
        "Formant Characteristics": "Formants",
        "syntax_feature": "Syntactic Features",
        "ppg_feature": "PPG",
        "spk_embedding": "Speaker Embedding",
        "metadata": "Metadata",
        "dnn_word_embedding": "DNN Word Embedding",
        "eGeMAPSv02": "eGeMAPSv02",
    }
    # plot the results
    all_results_df["manipulated_feature_group"] = all_results_df[
        "manipulated_feature_group"
    ].map(lambda x: rename_feature_group.get(x, x))

    plot_all_results(all_results_df, save=True)

    plot_encode_decode_comparison(
        librispeech_split="train-clean-100", modelname="wav2vec2-base"
    )
    plot_encode_decode_comparison(
        librispeech_split="train-clean-100", modelname="bert-base-uncased"
    )
