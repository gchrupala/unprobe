import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def load_data(
    datapath=os.path.join(PROJECT_ROOT, "data"), librispeech_split="dev-clean"
):
    opensmile_data = pd.read_pickle(
        os.path.join(
            datapath, f"librispeech-{librispeech_split}_opensmile_features.pickle"
        )
    )
    opensmile_data["fileid"] = opensmile_data["file"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    with open(
        os.path.join(
            datapath, f"librispeech-{librispeech_split}_audio_representation.pickle"
        ),
        "rb",
    ) as f:
        audio_rep = pickle.load(f)
    # with open(os.path.join(datapath, f"librispeech-{librispeech_split}_text_representation.pickle"), "rb") as f:
    #     text_rep = pickle.load(f)

    with open(
        os.path.join(
            datapath, f"librispeech-{librispeech_split}_transcriptions.pickle"
        ),
        "rb",
    ) as f:
        transcriptions = pickle.load(f)
    # # Embeddings for phone and word
    # with open(os.path.join(datapath, f"librispeech-{librispeech_split}_phones_embeddings.pickle"), "rb") as f:
    #     phone_embeddings = pickle.load(f)
    # with open(os.path.join(datapath, f"librispeech-{librispeech_split}_words_embeddings.pickle"), "rb") as f:
    #     word_embeddings = pickle.load(f)
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

    # vectorizer = CountVectorizer()
    # sent_words = [" ".join(f["words"]) for f in transcriptions]
    # bag_of_words = vectorizer.fit_transform(sent_words).toarray()
    # sent_phones = [" ".join(f["phones"]) for f in transcriptions]
    # bag_of_phones = vectorizer.fit_transform(sent_phones).toarray()
    # grouped_input_data["words"] = word_embeddings
    # grouped_input_data["phones"] = phone_embeddings
    grouped_input_data["non_acoustic"] = np.array(
        [f["non_acoustic"] for f in transcriptions]
    )

    return {
        "audio_rep": audio_rep,
        # "text_rep": text_rep,
        "input_data": grouped_input_data,
    }


def run_gridsearch(probe_name="ridge") -> None:
    probe_data = load_data(librispeech_split="dev-clean")

    if probe_name == "ridge":
        model = Ridge()
        # model = Lasso()
        param_grid = {
            "alpha": [10**x for x in range(-5, 5)],
            # "solver": ["auto", "sag", "saga", "lsqr"],
            "fit_intercept": [True, False],
            # "max_iter": 10000,
        }

    elif probe_name == "rf":
        model = RandomForestRegressor(n_jobs=-1)
        # random forest params
        param_grid = {
            "max_depth": [5, 7, 10, 15],
            "min_samples_split": [10, 20, 40, 80],
            "min_samples_leaf": [5, 10, 20, 40],
            "max_features": [
                "sqrt",
                0.7,
            ],
            "ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        }

    elif probe_name == "kernel_ridge":
        model = KernelRidge()
        param_grid = {
            "alpha": [10**x for x in range(-5, 5)],
            # "kernel": ["linear", "polynomial", "rbf"],
            # "gamma": [0.1, 1, 10],
            # "degree": [2, 3, 4],
            # "coef0": [0.1, 1, 10],
        }

    else:
        raise ValueError("Unknown probe name")

    # Create the gridsearch object
    GSregressor = GridSearchCV(
        model,
        param_grid,
        scoring="r2",
        # n_jobs=-1,
        cv=5,
        verbose=1,
        return_train_score=True,
    )

    exclude_feature_groups = ["non_acoustic"]

    input_data = [
        probe_data["input_data"][f]
        for f in probe_data["input_data"]
        if f not in exclude_feature_groups
    ]
    X = np.concatenate(input_data, axis=1)
    y = probe_data["audio_rep"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    GSregressor.fit(X_train, y_train)
    y_pred = GSregressor.predict(X_test)

    # Check the best parameters
    print(GSregressor.best_params_)
    # Check the best score
    print(GSregressor.best_score_)
    # Check the score on the test set
    score = GSregressor.score(X_test, y_test)
    print(score)

    results = []
    results.append(
        {
            "probe": probe_name,
            "pred_representation": "audio",
            "r^2_score": score,
            "input_ablation": "None",
            "best_scores": GSregressor.best_score_,
            "best_params": GSregressor.best_params_,
            "train_r^2_score": GSregressor.score(X_train, y_train),
            # "train_mse": mean_squared_error(y_train, GSregressor.predict(X_train)),
        }
    )

    # label the dimensions of X with groups
    dimension_for_group = {}
    start, end = 0, 0
    for group in probe_data["input_data"]:
        end += probe_data["input_data"][group].shape[1]
        dimension_for_group[group] = {
            "start": start,
            "end": end,
        }
        start = end

    for group in dimension_for_group:
        start = dimension_for_group[group]["start"]
        end = dimension_for_group[group]["end"]
        X_test_permuted = X_test.copy()
        X_test_permuted[:, start:end] = np.random.permutation(X_test[:, start:end])
        y_pred_permuted = GSregressor.predict(X_test_permuted)
        score_permuted = GSregressor.score(X_test_permuted, y_test)
        print(f"Score for {group} ablation: {score_permuted}")
        print(f"Score decreased by: {(score - score_permuted) / score}")
        results.append(
            {
                "probe": probe_name,
                "pred_representation": "audio",
                "r^2_score": score_permuted,
                "input_ablation": group,
                "best_scores": GSregressor.best_score_,
                "best_params": GSregressor.best_params_,
                "train_r^2_score": GSregressor.score(X_train, y_train),
                # "train_mse": mean_squared_error(y_train, GSregressor.predict(X_train)),
            }
        )

    df = pd.DataFrame(results)
    # Save the results to a csv file
    df.to_csv(
        os.path.join(
            PROJECT_ROOT, "results", f"{probe_name}_grid_search_permutations.csv"
        ),
        index=False,
        sep=";",
    )

    # Save the cv_results to a pickle file
    with open(
        os.path.join(
            PROJECT_ROOT, "results", f"{probe_name}_grid_search_results.pickle"
        ),
        "wb",
    ) as f:
        pickle.dump(GSregressor.cv_results_, f)


def run_probe():
    probe_data = load_data(librispeech_split="train-clean-100")

    regressors = [
        # LinearRegression(n_jobs=-1),
        Ridge(),
        # Lasso(),
        # MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000),
        # MLPRegressor(hidden_layer_sizes=(768, 768), max_iter=1000),
        # RandomForestRegressor(n_jobs=-1),
    ]

    results = []
    representations = {
        "audio": probe_data["audio_rep"]
        # "text": probe_data["text_rep"]
    }

    # exclude_feature_groups = ["words", "phones"]
    exclude_feature_groups = ["non_acoustic"]

    for representation in representations:
        # First test the probe with no ablation
        # We get the input data "X" by concatenating every value in the input_data dictionary
        input_data = [
            probe_data["input_data"][f]
            for f in probe_data["input_data"]
            if f not in exclude_feature_groups
        ]
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
            # Save the training performance to see if model overfit
            y_train_pred = probe.predict(X_train)
            train_score = probe.score(X_train, y_train)
            train_mse = mean_squared_error(y_train, y_train_pred)

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
                    "train_r^2_score": train_score,
                    "train_mse": train_mse,
                }
            )

        # Then ablate each feature group
        for feature_group in tqdm(
            probe_data["input_data"], leave=False, desc="Feature groups"
        ):
            if feature_group in exclude_feature_groups:
                continue
            # Concatenate all input data except the feature group we want to ablate
            X = np.concatenate(
                [
                    probe_data["input_data"][f]
                    for f in probe_data["input_data"]
                    if f != feature_group and f not in exclude_feature_groups
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

                # Save the training performance to see if model overfit
                y_train_pred = probe.predict(X_train)
                train_score = probe.score(X_train, y_train)
                train_mse = mean_squared_error(y_train, y_train_pred)

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
                        "train_r^2_score": train_score,
                        "train_mse": train_mse,
                    }
                )

    return results


def process_results(results):
    df = pd.DataFrame(results)
    df["input_ablation"] = df["input_ablation"].fillna("None")

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

    scores = pd.read_csv(
        os.path.join(
            PROJECT_ROOT, "results", f"{modelname}_grid_search_permutations.csv"
        ),
        sep=";",
    )
    scores = process_results(scores)
    scores = scores.drop(["best_params", "pred_representation"], axis=1).rename(
        {"best_scores": "cv_test_r^2_score"}, axis=1
    )
    scores


def submitit():
    # This function is used to submit the probe to a cluster
    # It is not used in the current implementation
    from submitit import AutoExecutor

    executor = AutoExecutor(folder=f"{PROJECT_ROOT}/logdir/%A")
    executor.update_parameters(
        timeout_min=1200,
        slurm_partition="CPU",
        name="gridsearch",
    )

    # executor.submit(run_gridsearch, "ridge")
    executor.submit(run_gridsearch, "rf")


if __name__ == "__main__":
    # results = run_probe()

    # df = process_results(results)

    submitit()
