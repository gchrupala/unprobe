import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm, trange

from load_probe_data import get_section_shapes, load_data

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Get the hostname of the machine running the code
hostname = os.uname().nodename
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# Setting up environmental variables depending on the cluster this code is running on

if "snellius" in hostname:
    # If running on Snellius, use the Snellius dataset root
    DATASET_ROOT = os.path.realpath("/projects/prjs1586/corpora/LibriSpeech")
    ALIGNMENT_ROOT = DATASET_ROOT.replace("LibriSpeech", "librispeech_textgrids")
    SAVEPATH = "/projects/prjs1586/experimental_data"
    RESULTS_ROOT = "/projects/prjs1586/experimental_results"

else:
    # If running on local machine, use the local dataset root
    DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENT_ROOT = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENT_ROOT = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def r2_score(
    y_true, y_pred, multioutput="variance_weighted", sklearn_=False, train_data=None
):
    if sklearn_:
        from sklearn.metrics import r2_score as r2_score_

        logger.warning(
            "Using sklearn's r2_score. Note that this uses test data mean for r2 calculation."
        )
        return r2_score_(y_true, y_pred, multioutput=multioutput)
    else:
        # Use custom implementation due to sklearn using test data mean in r2 calculation
        # We want to use train data mean for both train and test r2 calculation
        assert train_data is not None, (
            "train_data must be provided for custom r2_score calculation"
        )
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        y_train_mean = np.mean(train_data, axis=0)
        ss_tot = np.sum((y_true - y_train_mean) ** 2, axis=0)
        r2 = 1 - ss_res / ss_tot
        if multioutput == "variance_weighted":
            var = np.var(y_true, axis=0)
            weights = var / np.sum(var)
            r2 = np.sum(r2 * weights)
        elif multioutput == "uniform_average":
            r2 = np.mean(r2)
        return r2


def pick_probe(probe_name: str = "ridge", n_components: None | int = None):
    if probe_name == "ridge":
        model = Ridge()
        param_grid = {
            "alpha": [10**x for x in range(-3, 5)],
            # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "max_iter": [1000, 2000,  3000],
        }
    elif probe_name == "random-forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()  # n_jobs=-1, verbose=1)
        param_grid = {
            "max_depth": [10, 15, 20],  # 5, 7,
            # "min_samples_split": [10, 20, 40, 80],
            # "min_samples_leaf": [5, 10, 20, 40],
            "max_features": [
                "sqrt",
                0.7,
            ],
            # "ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        }
    elif probe_name == "ridge-transformed-target":
        from sklearn.compose import TransformedTargetRegressor

        # We scale the Y values using standardscaler and apply PCA as the y_transformer
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"Using Ridge regression with TransformedTargetRegressor with PCA n_components={n_components}"
        )

        pca = PCA(n_components=n_components, svd_solver="full")
        scaler = StandardScaler()
        y_transformer = Pipeline([("scaler", scaler), ("pca", pca)])
        base_model = Ridge()
        model = TransformedTargetRegressor(
            regressor=base_model, transformer=y_transformer
        )
        param_grid = {
            "regressor__alpha": [10**x for x in range(-3, 5)],
            # "regressor__solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "regressor__max_iter": [1000, 2000,  3000],
        }
    elif probe_name == "ridge_classifier":
        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier()
        param_grid = {
            "alpha": [10**x for x in range(-3, 5)],
            # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "max_iter": [1000, 2000,  3000],
        }

    else:
        raise ValueError(f"Probe {probe_name} not supported")

    return model, param_grid


def run_probe(
    feature_sets: np.ndarray,
    model_hidden_states: np.ndarray,
    data_shape: dict,
    filename_timestamp: list[tuple],
    feature_groups: list[tuple],
    probe_name: str = "ridge",
    select_layers: list[int] | None = None,
    zeroing: bool = True,
    ablation: bool = True,
    permutation: bool = True,
    results_path: str = RESULTS_ROOT,
    save_predictions: bool = False,
    dim_reduction: int | bool | None = False,
) -> list[dict]:
    """Runs the probing task on the given data.

    Args:
        feature_sets: The input data.
        model_hidden_states: The target data.
        data_shape: The shape of the data.
        filename_timestamp: The list of filename and timestamp tuples.
        probe_name: The name of the probe to use. Options are 'ridge' and 'random_forest'.
        feature_groups: The feature groups to use for probing.
        select_layers: The layers to use for probing. If None, all layers are used.
        zeroing: Whether to perform zeroing manipulation.
        ablation: Whether to perform ablation manipulation.
        permutation: Whether to perform permutation manipulation.
        results_path: The directory to save the results to.
    Returns:
        A list of dictionaries containing the results.
    """

    section_shapes = np.array(get_section_shapes(data_shape=data_shape))

    if not any((zeroing, ablation, permutation)):
        logger.warning(
            "At least one manipulation mode must be True, setting ablation to True"
        )
        ablation = True

    results = []

    for layer in trange(model_hidden_states.shape[1], desc="Layers in selected layers"):
        layer_results = []
        current_layer = select_layers[layer] if select_layers is not None else layer
        logger.info(
            f"Probing with {probe_name} on selected DNN model layer {current_layer}..."
        )
        if dim_reduction is not False:
            regressor, param_grid = pick_probe(probe_name, n_components=dim_reduction)
        else:
            regressor, param_grid = pick_probe(probe_name)
        GS = GridSearchCV(
            estimator=regressor,
            param_grid=param_grid,
            n_jobs=-1,
            cv=5,
            verbose=1,
        )
        (
            X_train,
            X_test,
            y_train,
            y_test,
            filename_timestamp_train,
            filename_timestamp_test,
        ) = train_test_split(
            feature_sets,
            model_hidden_states[:, layer, :],
            filename_timestamp,
            test_size=0.2,
            random_state=42,
        )

        if isinstance(dim_reduction, int) and dim_reduction >= y_train.shape[1]:
            logger.warning(
                f"dim_reduction {dim_reduction} is greater than or equal to output feature dimension {y_train.shape[1]}. Skipping dimensionality reduction."
            )

            dim_reduction = False
        if dim_reduction is not False:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=dim_reduction, svd_solver="full")
            y_train = pca.fit_transform(y_train)
            y_test = pca.transform(y_test)
            logger.info(
                f"Applied PCA with n_components={dim_reduction} for layer {current_layer}"
            )
            # Save PCA for future use
            with open(
                os.path.join(results_path, f"layer_{current_layer}_target_pca.pkl"),
                "wb",
            ) as f:
                pickle.dump(pca, f)
            logger.info(
                f"Saved target feature PCA for layer {current_layer} to {os.path.join(results_path, f'layer_{current_layer}_target_pca.pkl')}"
            )
            logger.info(f"New y_train shape: {y_train.shape}")

        GS.fit(X_train, y_train)
        # train_score = GS.score(X_train, y_train)
        # test_score = GS.score(X_test, y_test)
        train_score = r2_score(
            y_train,
            GS.predict(X_train),
            multioutput="variance_weighted",
            train_data=y_train,
        )
        test_score = r2_score(
            y_test,
            GS.predict(X_test),
            multioutput="variance_weighted",
            train_data=y_train,
        )

        result = {
            "layer": current_layer,
            "train_score": train_score,
            "test_score": test_score,
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            # "coefficients": GS.best_estimator_.coef_,
            # "intercept": GS.best_estimator_.intercept_,
            "manipulation_mode": "none",
            "manipulated_feature_group": "none",
        }
        layer_results.append(result)

        feature_pred_dict = {
            "layer": current_layer,
            "filename_timestamp_test": filename_timestamp_test,
            "features": X_test,
            "groundtruth": y_test,
            "full_feature_predictions": GS.predict(X_test),
        }

        # Add random baseline with shuffled x to predict y
        regressor, param_grid = pick_probe(probe_name)
        # We can skip the GridSearchCV here and just use the best_params from above
        regressor.set_params(**GS.best_params_)
        # Shuffle feature_sets
        shuffled_X = feature_sets.copy()
        np.random.shuffle(shuffled_X)
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
            shuffled_X, model_hidden_states[:, layer, :], test_size=0.2, random_state=42
        )
        regressor.fit(X_train_rand, y_train_rand)
        # random_train_score = regressor.score(X_train_rand, y_train_rand)
        # random_test_score = regressor.score(X_test_rand, y_test_rand)
        random_train_score = r2_score(
            y_train_rand,
            regressor.predict(X_train_rand),
            multioutput="variance_weighted",
            train_data=y_train_rand,
        )
        random_test_score = r2_score(
            y_test_rand,
            regressor.predict(X_test_rand),
            multioutput="variance_weighted",
            train_data=y_train_rand,
        )
        result = {
            "layer": current_layer,
            "train_score": random_train_score,
            "test_score": random_test_score,
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            # "coefficients": GS.best_estimator_.coef_,
            # "intercept": GS.best_estimator_.intercept_,
            "manipulation_mode": "random_baseline",
            "manipulated_feature_group": "none",
        }
        layer_results.append(result)

        for group in feature_groups:
            # Create a mask and use the section shapes to mask the areas of input feature that we want to manipulate
            mask_array = np.ones(X_train.shape[1], dtype=bool)
            feature_names = []
            for name in group:
                range_start, range_end, feature_name = section_shapes[
                    section_shapes[:, 2] == name
                ][0]
                mask_array[int(range_start) : int(range_end)] = (
                    False  # Mask out the group features
                )
                feature_names.append(feature_name)
            group_name = (
                "+".join(feature_names) if len(feature_names) > 1 else feature_names[0]
            )

            if permutation:
                # Permutation of features based on mask_array
                permuted_x_train = X_train.copy()
                permuted_x_test = X_test.copy()
                # Permute only the features in the mask_array
                features_to_permute_train = permuted_x_train[:, ~mask_array]
                features_to_permute_test = permuted_x_test[:, ~mask_array]
                # Create random generator with fixed seed = 42
                rng = np.random.default_rng(seed=42)
                # Use permuted to shuffle along the 1st axis (features)
                features_to_permute_train = rng.permuted(
                    features_to_permute_train, axis=1
                )
                features_to_permute_test = rng.permuted(
                    features_to_permute_test, axis=1
                )
                permuted_x_train[:, ~mask_array] = features_to_permute_train
                permuted_x_test[:, ~mask_array] = features_to_permute_test

                # Reinitialize the regressor here
                regressor, param_grid = pick_probe(probe_name)
                GS_permute = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    n_jobs=-1,
                    cv=5,
                    verbose=1,
                )
                GS_permute.fit(permuted_x_train, y_train)

                # train_score_permuted = GS_permute.score(permuted_x_train, y_train)
                # test_score_permuted = GS_permute.score(permuted_x_test, y_test)
                train_score_permuted = r2_score(
                    y_train,
                    GS_permute.predict(permuted_x_train),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                test_score_permuted = r2_score(
                    y_test,
                    GS_permute.predict(permuted_x_test),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                result = {
                    "layer": current_layer,
                    "train_score": train_score_permuted,
                    "test_score": test_score_permuted,
                    "best_params": GS.best_params_,
                    "best_score": GS.best_score_,
                    # "coefficients": GS.best_estimator_.coef_,
                    # "intercept": GS.best_estimator_.intercept_,
                    # "permutation": f"{range_start}-{range_end}"
                    "manipulation_mode": "permutation",
                    "manipulated_feature_group": group_name,
                }
                layer_results.append(result)

                feature_pred_dict["permutation_" + group_name] = GS_permute.predict(
                    permuted_x_test
                )

            if zeroing:
                # Zeroing out features
                zeroed_x_train = X_train.copy()
                zeroed_x_train[:, mask_array] = 0
                zeroed_x_test = X_test.copy()
                zeroed_x_test[:, mask_array] = 0

                # Reinitialize the regressor again
                regressor, param_grid = pick_probe(probe_name)
                GS_zero = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    n_jobs=-1,
                    cv=5,
                    verbose=1,
                )

                GS_zero.fit(zeroed_x_train, y_train)

                # zeroed_train_score = GS_zero.score(zeroed_x_train, y_train)
                # zeroed_test_score = GS_zero.score(zeroed_x_test, y_test)
                zeroed_train_score = r2_score(
                    y_train,
                    GS_zero.predict(zeroed_x_train),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                zeroed_test_score = r2_score(
                    y_test,
                    GS_zero.predict(zeroed_x_test),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                result = {
                    "layer": current_layer,
                    "train_score": zeroed_train_score,
                    "test_score": zeroed_test_score,
                    "best_params": GS.best_params_,
                    "best_score": GS.best_score_,
                    # "coefficients": GS_zero.best_estimator_.coef_,
                    # "intercept": GS_zero.best_estimator_.intercept_,
                    "manipulation_mode": "zeroing",
                    "manipulated_feature_group": group_name,
                }
                layer_results.append(result)

                feature_pred_dict["zeroing_" + group_name] = GS_zero.predict(
                    zeroed_x_test
                )

            if ablation:
                # Ablation of features
                ablated_x_train = X_train.copy()[:, mask_array]
                ablated_x_test = X_test.copy()[:, mask_array]
                # Reinitialize the regressor again
                regressor, param_grid = pick_probe(probe_name)
                GS_ablate = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    n_jobs=-1,
                    cv=5,
                    verbose=1,
                )
                GS_ablate.fit(ablated_x_train, y_train)
                # ablated_train_score = GS_ablate.score(ablated_x_train, y_train)
                # ablated_test_score = GS_ablate.score(ablated_x_test, y_test)
                ablated_train_score = r2_score(
                    y_train,
                    GS_ablate.predict(ablated_x_train),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                ablated_test_score = r2_score(
                    y_test,
                    GS_ablate.predict(ablated_x_test),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                result = {
                    "layer": current_layer,
                    "train_score": ablated_train_score,
                    "test_score": ablated_test_score,
                    "best_params": GS_ablate.best_params_,
                    "best_score": GS.best_score_,
                    # "coefficients": GS_ablate.best_estimator_.coef_,
                    "manipulation_mode": "ablation",
                    "manipulated_feature_group": group_name,
                }
                layer_results.append(result)

                feature_pred_dict["ablation_" + group_name] = GS_ablate.predict(
                    ablated_x_test
                )

        # Write the layer results to a csv file after each layer is done
        df = pd.DataFrame(layer_results)
        df.to_csv(
            os.path.join(results_path, f"layer_{current_layer}_results.csv"),
            index=False,
        )
        # Append the layer results to the overall results
        results.extend(layer_results)

        if save_predictions:
            # Save the feature_pred_dict to a pickle file
            with open(
                os.path.join(results_path, f"layer_{current_layer}_predictions.pkl"),
                "wb",
            ) as f:
                pickle.dump(feature_pred_dict, f)
            logger.info(
                f"Saved predictions for layer {current_layer} to {os.path.join(results_path, f'layer_{current_layer}_predictions.pkl')}"
            )
            # Show the keys of feature_pred_dict
            logger.info(f"Keys in feature_pred_dict: {list(feature_pred_dict.keys())}")
    return results


def run_probe_bottom_up(
    feature_sets: np.ndarray,
    model_hidden_states: np.ndarray,
    data_shape: dict,
    filename_timestamp: list[tuple],
    feature_groups: list[tuple],
    probe_name: str = "ridge",
    select_layers: list[int] | None = None,
    zeroing: bool = True,
    ablation: bool = True,
    permutation: bool = True,
    results_path: str = RESULTS_ROOT,
    save_predictions: bool = False,
    dim_reduction: int | bool | None = False,
) -> list[dict]:
    """Runs the probing task on the given data in a bottom-up manner.

    Args:
        feature_sets: The input data.
        model_hidden_states: The target data.
        data_shape: The shape of the data.
        filename_timestamp: The list of filename and timestamp tuples.
        probe_name: The name of the probe to use. Options are 'ridge' and 'random_forest'.
        feature_groups: The feature groups to use for probing.
        select_layers: The layers to use for probing. If None, all layers are used.
        zeroing: Whether to perform zeroing manipulation.
        ablation: Whether to perform ablation manipulation.
        permutation: Whether to perform permutation manipulation.
        results_path: The directory to save the results to.
    Returns:
        A list of dictionaries containing the results.
    """

    section_shapes = np.array(get_section_shapes(data_shape=data_shape))

    if not any((zeroing, ablation, permutation)):
        logger.warning(
            "At least one manipulation mode must be True, setting ablation to True"
        )
        ablation = True

    results = []

    for layer in trange(model_hidden_states.shape[1], desc="Layers in selected layers"):
        layer_results = []
        current_layer = select_layers[layer] if select_layers is not None else layer
        logger.info(
            f"Probing with {probe_name} on selected DNN model layer {current_layer}..."
        )
        if dim_reduction is not False:
            regressor, param_grid = pick_probe(probe_name, n_components=dim_reduction)
        else:
            regressor, param_grid = pick_probe(probe_name)
        GS = GridSearchCV(
            estimator=regressor,
            param_grid=param_grid,
            n_jobs=-1,
            cv=5,
            verbose=1,
        )
        (
            X_train,
            X_test,
            y_train,
            y_test,
            filename_timestamp_train,
            filename_timestamp_test,
        ) = train_test_split(
            feature_sets,
            model_hidden_states[:, layer, :],
            filename_timestamp,
            test_size=0.2,
            random_state=42,
        )

        if isinstance(dim_reduction, int) and dim_reduction >= y_train.shape[1]:
            logger.warning(
                f"dim_reduction {dim_reduction} is greater than or equal to output feature dimension {y_train.shape[1]}. Skipping dimensionality reduction."
            )

            dim_reduction = False
        if dim_reduction is not False:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=dim_reduction, svd_solver="full")
            y_train = pca.fit_transform(y_train)
            y_test = pca.transform(y_test)
            logger.info(
                f"Applied PCA with n_components={dim_reduction} for layer {current_layer}"
            )
            # Save PCA for future use
            with open(
                os.path.join(results_path, f"layer_{current_layer}_target_pca.pkl"),
                "wb",
            ) as f:
                pickle.dump(pca, f)
            logger.info(
                f"Saved target feature PCA for layer {current_layer} to {os.path.join(results_path, f'layer_{current_layer}_target_pca.pkl')}"
            )
            logger.info(f"New y_train shape: {y_train.shape}")

        GS.fit(X_train, y_train)
        # train_score = GS.score(X_train, y_train)
        # test_score = GS.score(X_test, y_test)
        train_score = r2_score(
            y_train,
            GS.predict(X_train),
            multioutput="variance_weighted",
            train_data=y_train,
        )
        test_score = r2_score(
            y_test,
            GS.predict(X_test),
            multioutput="variance_weighted",
            train_data=y_train,
        )

        result = {
            "layer": current_layer,
            "train_score": train_score,
            "test_score": test_score,
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            "feature_group": "all",
        }
        layer_results.append(result)

        feature_pred_dict = {
            "layer": current_layer,
            "filename_timestamp_test": filename_timestamp_test,
            "features": X_test,
            "groundtruth": y_test,
            "full_feature_predictions": GS.predict(X_test),
        }

        # Add random baseline with shuffled x to predict y
        regressor, param_grid = pick_probe(probe_name)
        # We can skip the GridSearchCV here and just use the best_params from above
        regressor.set_params(**GS.best_params_)
        # Shuffle feature_sets
        shuffled_X = feature_sets.copy()
        np.random.shuffle(shuffled_X)
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
            shuffled_X, model_hidden_states[:, layer, :], test_size=0.2, random_state=42
        )
        regressor.fit(X_train_rand, y_train_rand)
        # random_train_score = regressor.score(X_train_rand, y_train_rand)
        # random_test_score = regressor.score(X_test_rand, y_test_rand)
        random_train_score = r2_score(
            y_train_rand,
            regressor.predict(X_train_rand),
            multioutput="variance_weighted",
            train_data=y_train_rand,
        )
        random_test_score = r2_score(
            y_test_rand,
            regressor.predict(X_test_rand),
            multioutput="variance_weighted",
            train_data=y_train_rand,
        )
        result = {
            "layer": current_layer,
            "train_score": random_train_score,
            "test_score": random_test_score,
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            "feature_group": "random_baseline",
        }
        layer_results.append(result)
        for group in feature_groups:
            # Create a mask and use the section shapes to mask the areas of input feature that we want to manipulate
            mask_array = np.zeros(X_train.shape[1], dtype=bool)
            feature_names = []
            for name in group:
                range_start, range_end, feature_name = section_shapes[
                    section_shapes[:, 2] == name
                ][0]
                mask_array[int(range_start) : int(range_end)] = (
                    True  # Mark the group features to keep
                )
                feature_names.append(feature_name)
            group_name = (
                "+".join(feature_names) if len(feature_names) > 1 else feature_names[0]
            )
            # Bottom-up probing by using only the features in the current group
            selected_x_train = X_train.copy()[:, mask_array]
            selected_x_test = X_test.copy()[:, mask_array]
            # Reinitialize the regressor again
            regressor, param_grid = pick_probe(probe_name)
            GS_bottom_up = GridSearchCV(
                estimator=regressor,
                param_grid=param_grid,
                n_jobs=-1,
            )
            GS_bottom_up.fit(selected_x_train, y_train)
            bottom_up_train_score = r2_score(
                y_train,
                GS_bottom_up.predict(selected_x_train),
                multioutput="variance_weighted",
                train_data=y_train,
            )
            bottom_up_test_score = r2_score(
                y_test,
                GS_bottom_up.predict(selected_x_test),
                multioutput="variance_weighted",
                train_data=y_train,
            )
            result = {
                "layer": current_layer,
                "train_score": bottom_up_train_score,
                "test_score": bottom_up_test_score,
                "best_params": GS_bottom_up.best_params_,
                "best_score": GS_bottom_up.best_score_,
                "feature_group": group_name,
            }
            layer_results.append(result)
            feature_pred_dict["bottom_up_" + group_name] = GS_bottom_up.predict(
                selected_x_test
            )

            if group_name == "ppg_feature":
                # Convert ppg_feature from probability into different kind of representation and see if result changes
                selected_x_train_top = np.argmax(selected_x_train, axis=1)
                selected_x_test_top = np.argmax(selected_x_test, axis=1)

                regressor, param_grid = pick_probe(probe_name)
                GS_bottom_up_phone_ID = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    n_jobs=-1,
                )
                GS_bottom_up_phone_ID.fit(selected_x_train_top.reshape(-1, 1), y_train)
                bottom_up_phone_ID_train_score = r2_score(
                    y_train,
                    GS_bottom_up_phone_ID.predict(selected_x_train_top.reshape(-1, 1)),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                bottom_up_phone_ID_test_score = r2_score(
                    y_test,
                    GS_bottom_up_phone_ID.predict(selected_x_test_top.reshape(-1, 1)),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                result = {
                    "layer": current_layer,
                    "train_score": bottom_up_phone_ID_train_score,
                    "test_score": bottom_up_phone_ID_test_score,
                    "best_params": GS_bottom_up_phone_ID.best_params_,
                    "best_score": GS_bottom_up_phone_ID.best_score_,
                    "feature_group": "ppg_ID",
                }
                layer_results.append(result)
                feature_pred_dict["bottom_up_" + "ppg_ID"] = (
                    GS_bottom_up_phone_ID.predict(selected_x_test_top.reshape(-1, 1))
                )
                # Now do onehot encoding
                from sklearn.preprocessing import OneHotEncoder

                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                selected_x_train_ohe = ohe.fit_transform(
                    selected_x_train_top.reshape(-1, 1)
                )
                selected_x_test_ohe = ohe.transform(selected_x_test_top.reshape(-1, 1))
                # Reinitialize the regressor again
                regressor, param_grid = pick_probe(probe_name)
                GS_bottom_up_ohe = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    n_jobs=-1,
                )
                GS_bottom_up_ohe.fit(selected_x_train_ohe, y_train)
                bottom_up_ohe_train_score = r2_score(
                    y_train,
                    GS_bottom_up_ohe.predict(selected_x_train_ohe),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                bottom_up_ohe_test_score = r2_score(
                    y_test,
                    GS_bottom_up_ohe.predict(selected_x_test_ohe),
                    multioutput="variance_weighted",
                    train_data=y_train,
                )
                result = {
                    "layer": current_layer,
                    "train_score": bottom_up_ohe_train_score,
                    "test_score": bottom_up_ohe_test_score,
                    "best_params": GS_bottom_up_ohe.best_params_,
                    "best_score": GS_bottom_up_ohe.best_score_,
                    "feature_group": group_name + "_onehot",
                }
                layer_results.append(result)
                feature_pred_dict["bottom_up_" + group_name + "_onehot"] = (
                    GS_bottom_up_ohe.predict(selected_x_test_ohe)
                )

        # Write the layer results to a csv file after each layer is done
        df = pd.DataFrame(layer_results)
        df.to_csv(
            f"{results_path}/bottom-up-layer_results_{current_layer}.csv", index=False
        )

        # Append the layer results to the overall results
        results.extend(layer_results)
    return results


def bert_check():
    librispeech_split = "dev-clean"
    modelname = "google-bert/bert-base-uncased"
    probe_name = "ridge"
    select_layers = None
    normalize_features = True
    all_results = []

    # Do a check on how many word embeddings are exactly the same
    feature_sets, model_hidden_state, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        select_layers=select_layers,
        normalize_features=normalize_features,
        reduce_dnn_word_embedding=False,
        one_hot_encode_syntax=False,
    )
    sections_shape = get_section_shapes(data_shape=data_shape)
    group = "dnn_word_embedding"
    range_start, range_end, feature_name = np.array(sections_shape)[
        np.array(sections_shape)[:, 2] == group
    ][0]
    input_features = feature_sets[:, int(range_start) : int(range_end)]
    for layer in range(model_hidden_state.shape[1]):
        num_same = np.sum(
            np.all(input_features == model_hidden_state[:, layer, :], axis=1)
        )
        logger.info(
            f"Layer {layer}: Number of identical samples between input and output: {num_same} out of {input_features.shape[0]}"
        )

    feature_sets, model_hidden_state, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        select_layers=select_layers,
        normalize_features=normalize_features,
        reduce_dnn_word_embedding=False,
        one_hot_encode_syntax=True,
    )

    # Pick just the DNN word embedding from feature sets
    sections_shape = get_section_shapes(data_shape=data_shape)
    for group in ["dnn_word_embedding", "syntax_feature"]:
        range_start, range_end, feature_name = np.array(sections_shape)[
            np.array(sections_shape)[:, 2] == group
        ][0]
        input_features = feature_sets[:, int(range_start) : int(range_end)]

        # Run a probe for basic sanity check
        regressor, param_grid = pick_probe(probe_name)

        GS = GridSearchCV(
            estimator=regressor,
            param_grid=param_grid,
            n_jobs=-1,
            cv=5,
            verbose=1,
        )
        for layer in range(model_hidden_state.shape[1]):
            (
                X_train,
                X_test,
                y_train,
                y_test,
                filename_timestamp_train,
                filename_timestamp_test,
            ) = train_test_split(
                input_features,
                model_hidden_state[:, layer, :],
                filename_timestamp,
                test_size=0.2,
                random_state=42,
            )

            GS.fit(X_train, y_train)
            train_score = r2_score(
                y_train,
                GS.predict(X_train),
                multioutput="variance_weighted",
                train_data=y_train,
            )
            test_score = r2_score(
                y_test,
                GS.predict(X_test),
                multioutput="variance_weighted",
                train_data=y_train,
            )

            results = {
                "layer": layer,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": GS.best_params_,
                "best_score": GS.best_score_,
                "feature_group": group,
                "probe_direction": "encoding",
            }
            all_results.append(results)

            regressor, param_grid = pick_probe(probe_name)

            GS = GridSearchCV(
                estimator=regressor,
                param_grid=param_grid,
                n_jobs=-1,
                cv=5,
                verbose=1,
            )
            GS.fit(y_train, X_train)
            train_score = r2_score(
                X_train,
                GS.predict(y_train),
                multioutput="variance_weighted",
                train_data=X_train,
            )
            test_score = r2_score(
                X_test,
                GS.predict(y_test),
                multioutput="variance_weighted",
                train_data=X_train,
            )

            results = {
                "layer": layer,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": GS.best_params_,
                "best_score": GS.best_score_,
                "feature_group": group,
                "probe_direction": "decoding",
            }
            all_results.append(results)

    # Plot the results and use facet wrap on probe_direction
    results_df = pd.DataFrame(all_results)
    import plotnine as p9

    plot = (
        p9.ggplot(results_df, p9.aes(x="layer", y="test_score", color="feature_group"))
        + p9.geom_line()
        + p9.geom_point()
        + p9.facet_grid("probe_direction ~ ")
        + p9.ggtitle(
            f"BERT encoding probing results on LibriSpeech {librispeech_split}"
        )
    )

    plot.show()


def parse_args(use_default=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="dev-clean",
        help="The LibriSpeech split to use.",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="facebook/wav2vec2-base",
        help="The name of the model to use. Choose from 'facebook/wav2vec2-base', 'facebook/wav2vec2-large-960h', 'answerdotai/ModernBERT-base'",
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        default="ridge",
        help="The name of the probe to use. Options are 'ridge' and 'random_forest'.",
    )
    parser.add_argument(
        "--select_layers",
        type=int,
        nargs="+",
        default=None,
        help="The layers to use for probing.",
    )
    parser.add_argument(
        "--zeroing",
        action="store_true",
        help="Whether to perform zeroing manipulation.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Whether to perform ablation manipulation.",
    )
    parser.add_argument(
        "--permutation",
        action="store_true",
        help="Whether to perform permutation manipulation.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save the test set predictions.",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        help="Whether to normalize the input features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing extracted features.",
    )
    parser.add_argument(
        "--dim_reduction",
        type=str,
        default="False",
        help="The number of dimensions to reduce the target features to using PCA.",
    )
    parser.add_argument(
        "--feature_groups_config",
        type=str,
        default=f"{SAVEPATH}/default_feature_groups.json",
        help="The feature groups to use for probing.",
    )
    parser.add_argument(
        "--syntax_deep_dive",
        action="store_true",
        help="Whether to run the syntax deep dive analysis.",
    )
    parser.add_argument(
        "--bottom_up_probe",
        action="store_true",
        help="Whether to run the bottom-up probe analysis.",
    )

    if use_default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname
    probe_name = args.probe_name
    select_layers = args.select_layers if args.select_layers is not None else None
    zeroing = args.zeroing
    ablation = args.ablation
    permutation = args.permutation
    save_predictions = args.save_predictions
    normalize_features = args.normalize_features
    normalize_string = "normalized" if normalize_features else "unnormalized"
    bottom_up_probe = args.bottom_up_probe

    try:
        dim_reduction = int(args.dim_reduction)
    except ValueError:
        if "none" in args.dim_reduction.lower():
            dim_reduction = None
        elif "false" in args.dim_reduction.lower():
            dim_reduction = False
        # elif "normalize" in args.dim_reduction.lower():
        #     dim_reduction = "normalize"
        else:
            raise ValueError(
                f"dim_reduction argument {args.dim_reduction} not understood. Please provide an integer, 'None', or 'False'."
            )

    # Check select_layer against model size to make sure layers are valid
    model_layer_dict = {
        "facebook/wav2vec2-base": 12,
        "facebook/wav2vec2-base-960h": 12,
        "facebook/wav2vec2-large": 24,
        "facebook/wav2vec2-large-960h": 24,
        "facebook/wav2vec2-large-xlsr-53": 24,
        "facebook/hubert-base-ls960": 12,
        "facebook/hubert-large-ll60k": 24,
        "facebook/hubert-large-ls960-ft": 24,
        "microsoft/wavlm-base": 12,
        "FacebookAI/roberta-base": 12,
        "google-bert/bert-base-uncased": 12,
        "answerdotai/ModernBERT-base": 22,
    }
    if select_layers is not None:
        if modelname in model_layer_dict:
            max_layers = (
                model_layer_dict[modelname] + 1
            )  # +1 because 0th layer is embedding layer

            select_layers = np.array(select_layers)
            # Select only the layers that are valid
            select_layers = select_layers[select_layers < max_layers]
            if len(select_layers) == 0:
                raise ValueError(
                    f"All selected layers [{select_layers}] are invalid for model {modelname} with max layers {max_layers - 1}"
                )
            # Select only positive layers
            select_layers = select_layers[select_layers >= 0]
            # Convert back to list
            select_layers = select_layers.tolist()

            logger.info(
                f"Validated selected layers for model {modelname}: {select_layers}"
            )
        else:
            logger.warning(
                f"Model {modelname} not found in model_layer_dict. Skipping layer validation."
            )

    if any((zeroing, ablation, permutation)) is False:
        logger.warning(
            "At least one manipulation mode (zeroing, ablation, permutation) should be True."
        )
        logger.warning("Setting ablation to True by default.")
        ablation = True

    logger.info(f"Using LibriSpeech split: {librispeech_split}")
    logger.info(f"Using model: {modelname}")
    logger.info(f"Using probe: {probe_name}")
    logger.info(f"Normalize features: {normalize_features}")
    logger.info(f"Dimensionality reduction: {dim_reduction}")
    logger.info(
        f"Using selected layers: {select_layers if select_layers is not None else 'all layers'}"
    )
    logger.info("-" * 30)

    results_path = os.path.realpath(RESULTS_ROOT)
    # Make sure the parent path of results_path exists
    if not os.path.exists(os.path.dirname(results_path)):
        raise ValueError(f"Results path {results_path} does not exist.")
    # Create a subdirectory for the current experiment with separate librispeech split and modelname
    if dim_reduction is False:
        results_dir = f"librispeech-{librispeech_split}/{modelname.split('/')[-1]}/{probe_name}_frame_probe_{normalize_string}"
    else:
        results_dir = f"librispeech-{librispeech_split}-dimreduction/{modelname.split('/')[-1]}/{probe_name}_frame_probe_{normalize_string}_dimreduction-{dim_reduction}"

    if "default" not in args.feature_groups_config:
        feature_groups_config_name = os.path.basename(
            args.feature_groups_config
        ).replace(".json", "")
        feature_groups_config_name = feature_groups_config_name.replace("_", "-")
        # Replace the first directory with feature_groups_config_name
        first_dir = results_dir.split("/")[0]
        results_dir = results_dir.replace(
            first_dir, f"{first_dir}-{feature_groups_config_name}"
        )

    results_path = os.path.join(results_path, results_dir)
    # Create the results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)

    logger.info(f"Results will be saved to {results_path}")
    logger.info("-" * 30)

    logger.info("Formatting data for probe...")

    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        normalize_features=normalize_features,
        overwrite=args.overwrite,
    )

    feature_groups_config = os.path.join(SAVEPATH, args.feature_groups_config)
    if not os.path.exists(feature_groups_config):
        logger.warning(
            f"Feature groups config {feature_groups_config} not found. Generating and using default groups."
        )
        feature_groups = [(x,) for x in list(data_shape.keys())]
        with open(os.path.join(SAVEPATH, "default_feature_groups.json"), "w") as f:
            json.dump(feature_groups, f)
    else:
        with open(feature_groups_config, "r") as f:
            feature_groups = json.load(f)
        logger.info(
            f"Using feature groups from {feature_groups_config}: {feature_groups}"
        )
    # Save data_shape to results_path as json file for future reference

    with open(os.path.join(results_path, "data_shape.json"), "w") as f:
        json.dump(data_shape, f)
    logger.info("Data formatted.")
    logger.info("Running probe...")
    top_down_results = run_probe(
        feature_sets=feature_sets,
        model_hidden_states=model_hidden_states,
        data_shape=data_shape,
        feature_groups=feature_groups,
        filename_timestamp=filename_timestamp,
        probe_name=probe_name,
        select_layers=select_layers,
        zeroing=zeroing,
        ablation=ablation,
        permutation=permutation,
        results_path=results_path,
        save_predictions=save_predictions,
        dim_reduction=dim_reduction,
    )

    if bottom_up_probe:
        logger.info("Running bottom-up probe...")
        bottom_up_results = run_probe_bottom_up(
            feature_sets=feature_sets,
            model_hidden_states=model_hidden_states,
            data_shape=data_shape,
            feature_groups=feature_groups,
            filename_timestamp=filename_timestamp,
            probe_name=probe_name,
            select_layers=select_layers,
            zeroing=zeroing,
            ablation=ablation,
            permutation=permutation,
            results_path=results_path,
            save_predictions=save_predictions,
            dim_reduction=dim_reduction,
        )

    # logger.info("Saving all results...")
    # df = pd.DataFrame(results)
    # # df.drop(columns=["coefficients"], inplace=True)
    # df.to_csv(
    #     f"{results_path}/all_layers_results_{normalize_string}.csv",
    #     index=False,
    # )


if __name__ == "__main__":
    main()
