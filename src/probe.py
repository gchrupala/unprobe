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
    # Embeddings for phone and word
    with open(
        os.path.join(
            datapath, f"librispeech-{librispeech_split}_phones_embeddings.pickle"
        ),
        "rb",
    ) as f:
        phone_embeddings = pickle.load(f)
    with open(
        os.path.join(
            datapath, f"librispeech-{librispeech_split}_words_embeddings.pickle"
        ),
        "rb",
    ) as f:
        word_embeddings = pickle.load(f)
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
    grouped_input_data["words"] = word_embeddings
    grouped_input_data["phones"] = phone_embeddings
    grouped_input_data["non_acoustic"] = np.array(
        [f["non_acoustic"] for f in transcriptions]
    )

    return {
        "audio_rep": audio_rep,
        # "text_rep": text_rep,
        "input_data": grouped_input_data,
    }


def run_probe(probe_name="ridge", librispeech_split="train-clean-100") -> None:
    probe_data = load_data(librispeech_split=librispeech_split)

    if probe_name == "ridge":
        model = Ridge
        n_jobs = -1
        # model = Lasso()
        param_grid = {
            "alpha": [10**x for x in range(-5, 3)],
            "solver": ["auto", "sag", "saga", "lsqr"],
            "fit_intercept": [True, False],
        }

    elif probe_name == "rf":
        model = RandomForestRegressor
        n_jobs = -1
        # random forest params
        param_grid = {
            "max_depth": [5, 7, 10, 15],
            # "min_samples_split": [10, 20, 40, 80],
            # "min_samples_leaf": [5, 10, 20, 40],
            "max_features": [
                "sqrt",
                0.7,
            ],
            # "ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        }

    elif probe_name == "kernel_ridge":
        model = KernelRidge
        n_jobs = None
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
        model(),
        param_grid,
        scoring="r2",
        n_jobs=n_jobs,
        cv=5,
        verbose=1,
        return_train_score=True,
    )

    exclude_feature_groups = []
    # exclude_feature_groups = ["non_acoustic", "words", "phones"]

    input_data = [
        probe_data["input_data"][f]
        for f in probe_data["input_data"]
        if f not in exclude_feature_groups
    ]
    X = np.concatenate(input_data, axis=1)
    y = probe_data["audio_rep"]
    # Check if we have multiple layers in dim 1
    if len(y.shape) == 2:
        y = y.unsqueeze(1)

    # make sure y has three dimensions (batch_size, num_layers, hidden_size)
    assert len(y.shape) == 3

    # If we have multiple layers, we need to run multiple probes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use the last layer to run gridsearch
    y_gs_train = y_train[:, -1, :]
    y_gs_test = y_test[:, -1, :]

    GSregressor.fit(X_train, y_gs_train)
    y_pred = GSregressor.predict(X_test)
    # Check the best parameters
    # print(GSregressor.best_params_)
    # Check the best score
    # print(GSregressor.best_score_)
    # Check the score on the test set
    score = GSregressor.score(X_test, y_gs_test)
    print(f"Score for gridsearch: {score}")
    print(f"Best params: {GSregressor.best_params_}")
    # Use the best parameters to run the probe for each layer
    results = []
    for num_layer in tqdm(range(y.shape[1])):
        print(f"Running probe for layer {num_layer}")
        regressor = model(**GSregressor.best_params_)
        regressor.fit(X_train, y_train[:, num_layer])
        # y_pred = regressor.predict(X_test)
        test_score = regressor.score(X_test, y_test[:, num_layer])
        train_score = regressor.score(X_train, y_train[:, num_layer])

        results.append(
            {
                "probe": probe_name,
                "pred_representation": "audio",
                "r^2_score": test_score,
                "input_ablation": "None",
                "train_r^2_score": train_score,
                "topline_score": test_score,
                "layer": num_layer,
                "mode": "None",
            }
        )

        # label the dimensions of X with groups
        dimension_for_group = {}
        start, end = 0, 0
        for group in probe_data["input_data"]:
            end += np.array(probe_data["input_data"][group]).shape[1]
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
            score_permuted = regressor.score(X_test_permuted, y_test[:, num_layer])
            # print(f"Score for {group} ablation: {score_permuted}")
            # print(f"Score decreased by: {(score - score_permuted) / score}%")
            results.append(
                {
                    "probe": probe_name,
                    "pred_representation": "audio",
                    "r^2_score": score_permuted,
                    "input_ablation": group,
                    "train_r^2_score": train_score,
                    "topline_score": test_score,
                    "layer": num_layer,
                    "mode": "permutation",
                }
            )
            # For each group, we also remove the group from the input data completely and refit the regressor
            # Concatenate all input data except the feature group we want to ablate
            X_train_ablated = np.delete(
                X_train,
                np.arange(dimension_for_group[group]["start"], end),
                axis=1,
            )
            X_test_ablated = np.delete(
                X_test,
                np.arange(dimension_for_group[group]["start"], end),
                axis=1,
            )
            ablated_regressor = model(**GSregressor.best_params_)
            ablated_regressor.fit(X_train_ablated, y_train[:, num_layer])
            test_score_ablated = ablated_regressor.score(
                X_test_ablated, y_test[:, num_layer]
            )
            train_score_ablated = ablated_regressor.score(
                X_train_ablated, y_train[:, num_layer]
            )
            results.append(
                {
                    "probe": probe_name,
                    "pred_representation": "audio",
                    "r^2_score": test_score_ablated,
                    "input_ablation": group,
                    "train_r^2_score": train_score_ablated,
                    "topline_score": test_score,
                    "layer": num_layer,
                    "mode": "ablation",
                }
            )

    df = pd.DataFrame(results)
    df = df[df["mode"] != "None"]
    df["r^2_score_decrease"] = (df["topline_score"] - df["r^2_score"]) / df[
        "topline_score"
    ]
    df = df.sort_values(
        by=["probe", "layer", "mode", "r^2_score_decrease"], ascending=True
    ).reset_index(drop=True)

    # Save the results to a csv file
    df.to_csv(
        os.path.join(
            PROJECT_ROOT,
            "results",
            f"{probe_name}_{librispeech_split}_permutations.csv",
        ),
        index=False,
        sep=";",
    )

    # Save the cv_results to a pickle file
    with open(
        os.path.join(
            PROJECT_ROOT,
            "results",
            f"{probe_name}_{librispeech_split}_grid_search_results.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(GSregressor.cv_results_, f)


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

    executor.submit(run_probe, probe_name="ridge", librispeech_split="dev-clean")
    executor.submit(run_probe, probe_name="rf", librispeech_split="dev-clean")

    # executor.submit(run_gridsearch, probe_name = "ridge", librispeech_split="train-clean-100")
    # executor.submit(run_gridsearch, probe_name = "rf", librispeech_split="train-clean-100")


if __name__ == "__main__":
    # results = run_probe()

    # df = process_results(results)

    # submitit()
    # run_probe(probe_name="ridge", librispeech_split="dev-clean")
    # run_probe(probe_name="rf", librispeech_split="dev-clean")

    run_probe(probe_name="ridge", librispeech_split="train-clean-100")
    run_probe(probe_name="rf", librispeech_split="train-clean-100")
