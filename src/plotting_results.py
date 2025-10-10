import glob
import logging
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
from cv2 import line

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_result_files() -> pd.DataFrame:
    logger.info(f"Loading result files from {RESULTS_DIR}...")
    if not os.path.exists(RESULTS_DIR):
        logger.error(f"Results directory {RESULTS_DIR} does not exist.")
        raise FileNotFoundError(f"Results directory {RESULTS_DIR} does not exist.")

    all_results_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    joined_df = pd.DataFrame()
    for result_file in all_results_files:
        logger.info(f"Found results file: {result_file}")
        name_parse = os.path.basename(result_file).replace(".csv", "").split("_")
        librispeech_split = name_parse[0].replace("librispeech-", "")
        modelname = name_parse[1]
        probename = name_parse[4]
        if "random" in probename:
            probename = "random_forest"

        logger.info(
            f"Parsed - Librispeech Split: {librispeech_split}, Model Name: {modelname}, Probe Name: {probename}"
        )
        df = pd.read_csv(result_file)
        df["modelname"] = modelname
        df["librispeech_split"] = librispeech_split
        df["probename"] = probename
        joined_df = pd.concat([joined_df, df], ignore_index=True)
    return joined_df


def plot_results(joined_df: pd.DataFrame) -> None:
    """Plot the results from the joined DataFrame.

    Args:
        joined_df (pd.DataFrame): The DataFrame containing the results to plot.
    """
    logger.info("Plotting results...")

    # Split into subsets based on probe name
    probes = joined_df["probename"].unique()
    librispeech_splits = joined_df["librispeech_split"].unique()
    manipulations = ["zeroing", "permutation", "ablation"]

    # Normalize the layer numbers to start from 0 and end at max layer for each model
    joined_df["norm_layer"] = joined_df.groupby("modelname")["layer"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    for probe, librispeech_split, manipulation in product(
        probes, librispeech_splits, manipulations
    ):
        subset_df = joined_df[
            (joined_df["probename"] == probe)
            & (joined_df["librispeech_split"] == librispeech_split)
            & (joined_df["manipulation_mode"] == manipulation)
        ].copy()

        all_feature_baseline = joined_df[
            (joined_df["probename"] == probe)
            & (joined_df["librispeech_split"] == librispeech_split)
            & (joined_df["manipulation_mode"] == "none")
        ].copy()
        random_baseline = joined_df[
            (joined_df["probename"] == probe)
            & (joined_df["librispeech_split"] == librispeech_split)
            & (joined_df["manipulation_mode"] == "random_baseline")
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
            + p9.ggtitle(
                f"Probe: {probe}, Librispeech Split: {librispeech_split} Manipulation: {manipulation}"
            )
            + p9.xlab("Layer")
            + p9.ylab("Test Score")
            + p9.theme(figure_size=(10, 6), dpi=300)
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
    results_df = load_result_files()
    plot_results(results_df)
