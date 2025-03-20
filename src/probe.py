import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import opensmile
import pandas as pd
import torch
import re
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from comparator import RepresentationComparator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def load_data(datapath=os.path.join(PROJECT_ROOT, "data")):
    opensmile_data = pd.read_csv(os.path.join(datapath, "opensmile_features.csv"))
    audio_rep = torch.load(
        os.path.join(datapath, "audio_representation.pt"), weights_only=False
    )
    text_rep = torch.load(
        os.path.join(datapath, "text_representation.pt"), weights_only=False
    )
    return opensmile_data, audio_rep, text_rep

def choose_probe(probe_name="linear"):
    if probe_name == "linear":
        return LinearRegression()
    elif probe_name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
    elif probe_name == "ridge":
        return Ridge()
    elif probe_name == "lasso":
        return Lasso()

def main():
    opensmile_data, audio_rep, text_rep = load_data()

    opensmile_feature_names = opensmile_data.columns[3:]
    # get set of groups of features until last underscore
    feature_groups = set(["_".join(f.split("_")[:2]) for f in opensmile_feature_names])
    # Use regex to remove anything in []
    feature_groups = set([re.sub(r"\[.*\]", "", f) for f in feature_groups])

    results = []

    # Construct linear regression model
    probe = choose_probe("ridge")


    representations = {"audio": audio_rep, "text": text_rep}

    for representation in representations:
        # First run without any ablation
        X = opensmile_data[opensmile_feature_names].values
        y = representations[representation]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        score = probe.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        results.append(
            {
                "probe": "Ridge",
                "pred_representation": representation,
                "r^2_score": score,
                "mse": mse,
                "input_ablation": "None",
            }
        )

        for feature_group in tqdm(feature_groups):
        
            group_to_keep = [
                f for f in opensmile_feature_names if feature_group not in f
            ]
            X = opensmile_data[group_to_keep].values
            y = representations[representation]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)
            score = probe.score(X_test, y_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append(
                {
                    "probe": "Ridge",
                    "pred_representation": representation,
                    "r^2_score": score,
                    "mse": mse,
                    "input_ablation": feature_group,
                }
            )

    df = pd.DataFrame(results)
    df

if __name__ == "__main__":
    main()