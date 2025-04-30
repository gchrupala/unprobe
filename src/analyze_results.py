import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def plot_results(probe = "ridge", librispeech_split = "train-clean-100"):
    csv_file = os.path.join(
        PROJECT_ROOT,
        "results",
        f"{probe}_{librispeech_split}_permutations.csv",
    )

    df = pd.read_csv(csv_file, sep=";")
    # fill na in mode column with "None"
    df["mode"] = df["mode"].fillna("None")
    # Filter out the rows where mode is "None"
    df = df[df["mode"] != "None"]

    df = df.sort_values(
        by=["probe", "layer", "mode","r^2_score_decrease"], ascending=True
    ).reset_index(drop=True)

    probe_name_lookup = {
        "ridge": "Ridge",
        "rf": "Random Forest",
        "kernel_ridge": "Kernel Ridge",
        "MLPRegressor": "MLP",
        "LinearRegression": "Linear Regression",
        "Lasso": "Lasso",
    }
    for mode in df["mode"].unique():
        plotting_df = df[df["mode"] == mode]
        # Make a bar plot of the results with layer on the x axis and r^2_score_decrease on the y axis and facet by input_ablation and color by probe
        # Set the style
        sns.set(style="whitegrid")
        # Set the figure size
        plt.figure(figsize=(10, 6))
        g = sns.FacetGrid(
            plotting_df[plotting_df["input_ablation"] != "None"],
            col="input_ablation",
            hue="input_ablation",
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharey=True,
            sharex=True,
        )
        g.map_dataframe(
            sns.barplot,
            x="layer",
            y="r^2_score_decrease",
            order=plotting_df["layer"].unique(),
            hue_order=plotting_df["probe"].unique(),
        )
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Layer", "R^2 Score Decrease")
        # g.set_xticklabels(g.get_xticklabels(), rotation=45)
        # plt.legend(title="Probe")

        # Set the title
        g.fig.suptitle(
            f"R^2 Score Decrease for {plotting_df['pred_representation'].unique()[0].capitalize()} Representation on {probe_name_lookup[plotting_df['probe'].unique()[0]]} with {mode.capitalize()} Mode",
            fontsize=16,
            fontweight="bold",
            y=1.05,
        )

        plt.tight_layout()
        plt.show()

        fig_savepath = os.path.join(
            PROJECT_ROOT,
            "results",
            f"{plotting_df['pred_representation'].unique()[0]}_{plotting_df['probe'].unique()[0]}_{mode}_results.png",
        )

        # Save the figure
        plt.savefig(
            fig_savepath,
            dpi=300,
            bbox_inches="tight",
        )


def check_params():
    modelname = "rf"
    modelname = "ridge"
    cv_results_file = os.path.join(
        PROJECT_ROOT, "results", f"{modelname}_grid_search_results.pickle"
    )
    with open(cv_results_file, "rb") as f:
        cv_results = pickle.load(f)
    # Check the best parameters by sorting the mean_train_score - mean_test_score so less overfitting
    train_test_diff = cv_results["mean_train_score"] - cv_results["mean_test_score"]
    overfit_order = np.argsort(-1 * train_test_diff)
    # Check the test score order
    order = np.argsort(-1 * (cv_results["mean_test_score"]))
    # Get the best parameters that are not overfitting and the best test score
    best_params = cv_results["params"][overfit_order[0]]
    best_test_score = cv_results["mean_test_score"][order[0]]
    best_params = cv_results["params"][order[0]]
