import glob
import logging
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

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


def load_result_files(results_dir: str) -> pd.DataFrame:
    # logger.info(f"Loading result files from {results_dir}...")
    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} does not exist.")
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    # all_results_files = glob.glob(os.path.join(results_dir, "*.csv"))
    all_results_files = glob.glob(
        os.path.join(results_dir, "librispeech-*/**/*.csv"), recursive=True
    )
    all_results_files = [
        x for x in all_results_files if "all_layers" not in x and "_normalized" in x
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
        df["modelname"] = modelname
        df["librispeech_split"] = librispeech_split
        df["probename"] = probename
        df["normalization"] = normalization

        df["norm_layer"] = df["layer"] / model_layer_dict.get(modelname, 1)
        all_results_df = pd.concat([all_results_df, df], ignore_index=True)

    return all_results_df


caption = """\
Baselines:
- Black dashed line: Random Baseline (random_baseline)
- Black dotted line: All Feature Baseline (all_feat_baseline)
All feature baseline is the best case results with all features intact. The feature group manipulations should be compared against these baselines.
"""


def plot_results(all_results_df: pd.DataFrame) -> None:
    """Plot the results from the joined DataFrame.

    Args:
        all_results_df (pd.DataFrame): The DataFrame containing the results to plot.
    """
    logger.info("Plotting results...")

    # Split into subsets based on probe name
    probes = all_results_df["probename"].unique()
    librispeech_splits = all_results_df["librispeech_split"].unique()
    manipulations = ["zeroing", "permutation", "ablation"]

    # # Normalize the layer numbers to start from 0 and end at max layer for each model
    # all_results_df["norm_layer"] = all_results_df.groupby("modelname")[
    #     "layer"
    # ].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    for probe, librispeech_split, manipulation in product(
        probes, librispeech_splits, manipulations
    ):
        subset_df = all_results_df[
            (all_results_df["probename"] == probe)
            & (all_results_df["librispeech_split"] == librispeech_split)
            & (all_results_df["manipulation_mode"] == manipulation)
        ].copy()

        all_feature_baseline = all_results_df[
            (all_results_df["probename"] == probe)
            & (all_results_df["librispeech_split"] == librispeech_split)
            & (all_results_df["manipulation_mode"] == "none")
        ].copy()
        random_baseline = all_results_df[
            (all_results_df["probename"] == probe)
            & (all_results_df["librispeech_split"] == librispeech_split)
            & (all_results_df["manipulation_mode"] == "random_baseline")
        ].copy()

        random_baseline["manipulated_feature_group"] = "random_baseline"
        all_feature_baseline["manipulated_feature_group"] = "all_feat_baseline"
        random_baseline["baseline"] = True
        all_feature_baseline["baseline"] = True
        subset_df["baseline"] = False

        # subset_df = pd.concat(
        #     [subset_df, all_feature_baseline, random_baseline], ignore_index=True
        # ).reset_index()

        if subset_df.empty:
            logger.warning(
                f"No data for probe {probe} and split {librispeech_split} and manipulation {manipulation}. Skipping."
            )
            continue

        figure = (
            p9.ggplot()
            + p9.facet_wrap("~ modelname", ncol=2)
            + p9.geom_line(
                p9.aes(
                    x="norm_layer",
                    y="test_score",
                    color="manipulated_feature_group",
                    group="manipulated_feature_group",
                ),
                alpha=0.7,
                data=subset_df,
            )
            + p9.geom_point(
                p9.aes(
                    x="norm_layer", y="test_score", color="manipulated_feature_group"
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
                plot_caption=p9.element_text(
                    ha="left", margin={"t": 1, "units": "lines"}
                ),
            )
            + p9.scale_color_discrete(name="Manipulated Feature Group")
            + p9.labs(
                x="Layer (from shallow to deep, normalized)",
                y="Test Score (R²)",
                title="Encoding Probe Performance Across Model Layers",
                subtitle=f"Probe: {probe}, Librispeech Split: {librispeech_split} Manipulation: {manipulation}",
                caption=caption,
            )
        )
        figure.show()

        base_output_dir = os.path.join(RESULTS_DIR, "figures")
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, probe), exist_ok=True)
        os.makedirs(
            os.path.join(base_output_dir, probe, librispeech_split), exist_ok=True
        )

        output_filepath = os.path.join(
            base_output_dir,
            probe,
            librispeech_split,
            f"{probe}_{librispeech_split}_{manipulation}_results.png",
        )
        figure.save(output_filepath)
        logger.info(f"Saved figure to {output_filepath}")
    logger.info("Plotting complete.")


if __name__ == "__main__":
    all_results_df = load_result_files(results_dir=RESULTS_DIR)
    plot_results(all_results_df)
