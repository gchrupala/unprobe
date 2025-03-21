import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def load_data(datapath=os.path.join(PROJECT_ROOT, "data")):
    opensmile_data = pd.read_pickle(os.path.join(datapath, "opensmile_features.pickle"))
    opensmile_data["fileid"] = opensmile_data["file"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    with open(os.path.join(datapath, "audio_representation.pickle"), "rb") as f:
        audio_rep = pickle.load(f)
    with open(os.path.join(datapath, "text_representation.pickle"), "rb") as f:
        text_rep = pickle.load(f)

    with open(
        os.path.join(datapath, "librispeech_dev-clean_transcriptions.pickle"), "rb"
    ) as f:
        transcriptions = pickle.load(f)
    # Sort transcription based on the order of opensmile data fileid column

    transcriptions = [
        f for x in opensmile_data["fileid"] for f in transcriptions if x in f["fileid"]
    ]

    # Load opensmile grouping
    with open(os.path.join(datapath, "egemaps_feature_grouping.json"), "r") as f:
        egemaps_feature_grouping = json.load(f)

    grouped_input_data = {}
    for group in egemaps_feature_grouping:
        feature_names = egemaps_feature_grouping[group]
        grouped_input_data[group] = opensmile_data[feature_names].values

    vectorizer = CountVectorizer()
    sent_words = [" ".join(f["words"]) for f in transcriptions]
    bag_of_words = vectorizer.fit_transform(sent_words).toarray()
    sent_phones = [" ".join(f["phones"]) for f in transcriptions]
    bag_of_phones = vectorizer.fit_transform(sent_phones).toarray()
    grouped_input_data["words"] = bag_of_words
    grouped_input_data["phones"] = bag_of_phones
    grouped_input_data["non_acoustic"] = np.array(
        [f["non_acoustic"] for f in transcriptions]
    )

    return {
        "audio_rep": audio_rep,
        "text_rep": text_rep,
        "input_data": grouped_input_data,
    }


def run_probe():
    probe_data = load_data()

    regressors = [
        LinearRegression(n_jobs=-1),
        Ridge(),
        Lasso(),
        # MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000),
        MLPRegressor(),
        RandomForestRegressor(n_jobs=-1),
    ]

    results = []
    representations = {
        "audio": probe_data["audio_rep"]
        # "text": probe_data["text_rep"]
    }

    for representation in representations:
        # First test the probe with no ablation
        # We get the input data "X" by concatenating every value in the input_data dictionary
        input_data = list(probe_data["input_data"].values())
        X = np.concatenate(input_data, axis=1)
        y = representations[representation]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        for probe in tqdm(regressors, leave=False, desc="Probes"):
            # We fit every probe in the regressors list to the training data
            probe_type = probe.__class__.__name__
            tqdm.write(f"Running {probe_type}")
            probe.fit(X_train, y_train)

            # We predict the test data and calculate the r^2 score and mean squared error
            y_pred = probe.predict(X_test)
            score = probe.score(X_test, y_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append(
                {
                    "probe": probe_type,
                    "pred_representation": representation,
                    "r^2_score": score,
                    "mse": mse,
                    "input_ablation": "None",
                }
            )

        # Then ablate each feature group
        for feature_group in tqdm(
            probe_data["input_data"], leave=False, desc="Feature groups"
        ):
            # Concatenate all input data except the feature group we want to ablate
            X = np.concatenate(
                [
                    probe_data["input_data"][f]
                    for f in probe_data["input_data"]
                    if f != feature_group
                ],
                axis=1,
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            for probe in tqdm(regressors, leave=False, desc="Probes"):
                # We fit every probe in the regressors list to the training data
                probe_type = probe.__class__.__name__
                tqdm.write(f"Running {probe_type}")
                probe.fit(X_train, y_train)

                # We predict the test data and calculate the r^2 score and mean squared error
                y_pred = probe.predict(X_test)
                score = probe.score(X_test, y_test)
                mse = mean_squared_error(y_test, y_pred)
                results.append(
                    {
                        "probe": probe_type,
                        "pred_representation": representation,
                        "r^2_score": score,
                        "mse": mse,
                        "input_ablation": feature_group,
                    }
                )

    return results


def process_results(results):
    df = pd.DataFrame(results)

    # For each probe, pred_representation, and input_type, find
    # the decrease in r^2 score compared to no ablation

    no_ablation = df[df["input_ablation"] == "None"]
    for probe in df["probe"].unique():
        for pred_representation in df["pred_representation"].unique():
            no_ablation_score = no_ablation[
                (no_ablation["probe"] == probe)
                & (no_ablation["pred_representation"] == pred_representation)
            ]["r^2_score"].values[0]

            ablation_scores = df[
                (df["probe"] == probe)
                & (df["pred_representation"] == pred_representation)
                & (df["input_ablation"] != "None")
            ]["r^2_score"].values

            ablation_scores = np.array(ablation_scores)
            decrease = (no_ablation_score - ablation_scores) / no_ablation_score
            df.loc[
                (df["probe"] == probe)
                & (df["pred_representation"] == pred_representation)
                & (df["input_ablation"] != "None"),
                "r^2_score_decrease",
            ] = decrease

    # replace NaNs with 0
    df["r^2_score_decrease"] = df["r^2_score_decrease"].fillna(0)
    df = df.sort_values(
        by=["probe", "r^2_score_decrease"], ascending=False
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    results = run_probe()

    df = process_results(results)
