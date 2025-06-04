import json
import os
import pickle
from typing import Dict, List

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


def select_regression_model(probe_name):
    """
    Selects the regression model and its parameters based on the probe name.

    Args:
        probe_name (str): Name of the probe, can be "ridge", "rf", or "kernel_ridge".
    Returns:
        model: The regression model class to be used.
        n_jobs: Number of jobs to run in parallel for the model.
        param_grid: Dictionary containing the parameters for grid search.

    Raises:
        ValueError: If the probe name is not recognized.

    """
    if probe_name == "ridge":
        model = Ridge
        n_jobs = 16
        # model = Lasso()
        param_grid = {
            "alpha": [10**x for x in range(-5, 3)],
            # "solver": ["auto", "sag", "saga", "lsqr"],
            "solver": ["auto"],
            "fit_intercept": [True, False],
        }

    elif probe_name == "rf":
        model = RandomForestRegressor
        n_jobs = -1
        # random forest params
        param_grid = {
            "max_depth": [10, 15],  # 5, 7,
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
    return model, n_jobs, param_grid


def run_probe(
    probe_data, probe_name="ridge", librispeech_split="train-clean-100"
) -> None:
    model, n_jobs, param_grid = select_regression_model(probe_name)

    # Create the gridsearch object
    GSregressor = GridSearchCV(
        model(),
        param_grid,
        scoring="r2",
        n_jobs=n_jobs,
        cv=5,
        verbose=2,
        return_train_score=True,
    )

    X, y = probe_data
    # Check if we have multiple layers in dim 1
    if len(y.shape) == 2:
        # If we have only two dimensions, we need to add a third dimension to simulate multiple layers
        # This is a workaround to make sure we have three dimensions (batch_size, num_layers, hidden_size)
        y = y.unsqueeze(1)

    # make sure y has three dimensions (batch_size, num_layers, hidden_size)
    assert len(y.shape) == 3

    # If we have multiple layers, we need to run multiple probes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results: List[Dict] = []
    for num_layer in tqdm(range(y.shape[1])):
        tqdm.write(f"Running probe for layer {num_layer}")

        # Run gridsearch for each layer
        GSregressor.fit(X_train, y_train[:, num_layer, :])

        regressor = model(**GSregressor.best_params_)
        regressor.fit(X_train, y_train[:, num_layer])
        test_score = regressor.score(X_test, y_test[:, num_layer])
        train_score = regressor.score(X_train, y_train[:, num_layer])

        results.append(
            {
                "probe": probe_name,
                "layer": num_layer,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": GSregressor.best_params_,
                "best_score": GSregressor.best_score_,
                "coefficients": GSregressor.best_estimator_.coef_,
                "manipulation_mode": "none",
                "manipulated_feature_group": "none",
            }
        )

        # label the dimensions of X with groups

        for range_start, range_end, name in [
            (0, 125, "acoustic"),
            (125, 225, "word_embedding"),
            (225, 325, "phone_embedding"),
            (325, 327, "metadata"),
            (327, X_train.shape[1], "syntax_features"),
        ]:
            X_test_permuted = X_test.copy()
            X_test_permuted[:, range_start:range_end] = np.random.permutation(
                X_test[:, range_start:range_end]
            )
            test_score_permuted = regressor.score(X_test_permuted, y_test[:, num_layer])
            # print(f"Score for {group} ablation: {score_permuted}")
            # print(f"Score decreased by: {(score - score_permuted) / score}%")
            results.append(
                {
                    "probe": probe_name,
                    "layer": num_layer,
                    "train_score": train_score,
                    "test_score": test_score_permuted,
                    "best_params": GSregressor.best_params_,
                    "best_score": GSregressor.best_score_,
                    "coefficients": GSregressor.best_estimator_.coef_,
                    "manipulation_mode": "permutation",
                    "manipulated_feature_group": name,
                }
            )
            # For each group, we also remove the group from the input data completely and refit the regressor
            # Concatenate all input data except the feature group we want to ablate
            X_train_ablated = np.delete(
                X_train,
                np.s_[range_start:range_end],
                axis=1,
            )
            X_test_ablated = np.delete(
                X_test,
                np.s_[range_start:range_end],
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
                    "layer": num_layer,
                    "train_score": train_score_ablated,
                    "test_score": test_score_ablated,
                    "best_params": GSregressor.best_params_,
                    "best_score": GSregressor.best_score_,
                    "coefficients": GSregressor.best_estimator_.coef_,
                    "manipulation_mode": "ablation",
                    "manipulated_feature_group": name,
                }
            )
    return results


def postprocess_results(results, probe_name, librispeech_split):
    # This function processes the results of the probe and saves them to a csv file
    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)
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


def submitit():
    # This function is used to submit the probe to a cluster
    # It is not used in the current implementation
    from submitit import AutoExecutor

    executor = AutoExecutor(folder=f"{PROJECT_ROOT}/logdir/%A")
    executor.update_parameters(
        timeout_min=1200,
        name="gridsearch",
    )

    executor.submit(run_probe, probe_name="ridge", librispeech_split="dev-clean")
    executor.submit(run_probe, probe_name="rf", librispeech_split="dev-clean")

    # executor.submit(run_gridsearch, probe_name = "ridge", librispeech_split="train-clean-100")
    # executor.submit(run_gridsearch, probe_name = "rf", librispeech_split="train-clean-100")


# if __name__ == "__main__":
#     pass
    # results = run_probe(probe_name="ridge", librispeech_split="train-clean-100")
