import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm, trange

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


sections_shapes = (
    (0, 125, "acoustic"),
    (
        125,
        100 + 125,
        "word_embedding",
    ),
    (
        100 + 125,
        100 + 125 + 34,
        "syntax_features",
    ),
    (
        100 + 125 + 34,
        100 + 125 + 34 + 40,
        "ppgs_features",
    ),
    (
        100 + 125 + 34 + 40,
        100 + 125 + 34 + 40 + 100,
        "spk_embedding",
    ),
    (
        100 + 125 + 34 + 40 + 100,
        100 + 125 + 34 + 40 + 100 + 2,
        "metadata",
    ),
)

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


def pick_probe(probe_name: str = "ridge"):
    if probe_name == "ridge":
        model = Ridge()
        param_grid = {
            "alpha": [10**x for x in range(-5, 3)],
            # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "max_iter": [1000, 2000,  3000],
        }
    elif probe_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_jobs=-1, verbose=5)
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
    else:
        raise ValueError(f"Probe {probe_name} not supported")

    return model, param_grid


def run_probe(
    processed_X: np.ndarray,
    processed_y: np.ndarray,
    filename_timestamp: list[tuple],
    probe_name: str = "ridge",
    select_layers: list[int] | None = None,
    zeroing: bool = True,
    ablation: bool = True,
    permutation: bool = True,
    results_path: str = RESULTS_ROOT,
    save_predictions: bool = False,
) -> list[dict]:
    """Runs the probing task on the given data.

    Args:
        processed_X: The input data.
        processed_y: The target data.
        probe_name: The name of the probe to use. Options are 'ridge' and 'random_forest'.
        select_layers: The layers to use for probing. If None, all layers are used.
        zeroing: Whether to perform zeroing manipulation.
        ablation: Whether to perform ablation manipulation.
        permutation: Whether to perform permutation manipulation.
        results_path: The directory to save the results to.
    Returns:
        A list of dictionaries containing the results.
    """

    results = []
    for layer in trange(processed_y.shape[1], desc="Layers in selected layers"):
        layer_results = []
        current_layer = select_layers[layer] if select_layers is not None else layer
        logger.info(
            f"Probing with {probe_name} on selected DNN model layer {current_layer}..."
        )
        regressor, param_grid = pick_probe(probe_name)
        GS = GridSearchCV(
            estimator=regressor,
            param_grid=param_grid,
            # n_jobs=-1,
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
            processed_X,
            processed_y[:, layer, :],
            filename_timestamp,
            test_size=0.2,
            random_state=42,
        )

        GS.fit(X_train, y_train)
        train_score = GS.score(X_train, y_train)
        test_score = GS.score(X_test, y_test)
        # print(f"Train score: {train_score}")
        # print(f"Test score: {test_score}")
        # print(f"Best parameters: {GS.best_params_}")
        # print(f"Best score: {GS.best_score_}")

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
        # Shuffle processed_X
        shuffled_X = processed_X.copy()
        np.random.shuffle(shuffled_X)
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
            shuffled_X, processed_y[:, layer, :], test_size=0.2, random_state=42
        )
        regressor.fit(X_train_rand, y_train_rand)
        random_train_score = regressor.score(X_train_rand, y_train_rand)
        random_test_score = regressor.score(X_test_rand, y_test_rand)
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

        for range_start, range_end, name in tqdm(
            sections_shapes, desc="Feature Groups", leave=False
        ):
            assert any((zeroing, ablation, permutation)), (
                "At least one manipulation mode must be True"
            )
            if permutation:
                # Permutation of features
                permuted_x_train = X_train.copy()
                permuted_x_train[:, range_start:range_end] = np.random.permutation(
                    permuted_x_train[:, range_start:range_end]
                )
                permuted_x_test = X_test.copy()
                permuted_x_test[:, range_start:range_end] = np.random.permutation(
                    permuted_x_test[:, range_start:range_end]
                )
                # Reinitialize the regressor here
                regressor, param_grid = pick_probe(probe_name)
                GS_permute = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    # n_jobs=-1,
                    cv=5,
                    verbose=1,
                )
                GS_permute.fit(permuted_x_train, y_train)
                train_score_permuted = GS_permute.score(permuted_x_train, y_train)
                test_score_permuted = GS_permute.score(permuted_x_test, y_test)
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
                    "manipulated_feature_group": name,
                }
                layer_results.append(result)

                feature_pred_dict["permutation_" + name] = GS_permute.predict(
                    permuted_x_test
                )

            if zeroing:
                # Zeroing out features
                zeroed_x_train = X_train.copy()
                zeroed_x_train[:, range_start:range_end] = 0
                zeroed_x_test = X_test.copy()
                zeroed_x_test[:, range_start:range_end] = 0

                # Reinitialize the regressor again
                regressor, param_grid = pick_probe(probe_name)
                GS_zero = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    # n_jobs=-1,
                    cv=5,
                    verbose=1,
                )

                GS_zero.fit(zeroed_x_train, y_train)
                zeroed_train_score = GS_zero.score(zeroed_x_train, y_train)
                zeroed_test_score = GS_zero.score(zeroed_x_test, y_test)
                result = {
                    "layer": current_layer,
                    "train_score": zeroed_train_score,
                    "test_score": zeroed_test_score,
                    "best_params": GS.best_params_,
                    "best_score": GS.best_score_,
                    # "coefficients": GS_zero.best_estimator_.coef_,
                    # "intercept": GS_zero.best_estimator_.intercept_,
                    "manipulation_mode": "zeroing",
                    "manipulated_feature_group": name,
                }
                layer_results.append(result)

                feature_pred_dict["zeroing_" + name] = GS_zero.predict(zeroed_x_test)

            if ablation:
                # Ablation of features
                ablated_x_train = X_train.copy()
                ablated_x_train = np.delete(
                    ablated_x_train, np.s_[range_start:range_end], axis=1
                )
                ablated_x_test = X_test.copy()
                ablated_x_test = np.delete(
                    ablated_x_test, np.s_[range_start:range_end], axis=1
                )
                # Reinitialize the regressor again
                regressor, param_grid = pick_probe(probe_name)
                GS_ablate = GridSearchCV(
                    estimator=regressor,
                    param_grid=param_grid,
                    # n_jobs=-1,
                    cv=5,
                    verbose=1,
                )
                GS_ablate.fit(ablated_x_train, y_train)
                ablated_train_score = GS_ablate.score(ablated_x_train, y_train)
                ablated_test_score = GS_ablate.score(ablated_x_test, y_test)
                result = {
                    "layer": current_layer,
                    "train_score": ablated_train_score,
                    "test_score": ablated_test_score,
                    "best_params": GS_ablate.best_params_,
                    "best_score": GS.best_score_,
                    # "coefficients": GS_ablate.best_estimator_.coef_,
                    "manipulation_mode": "ablation",
                    "manipulated_feature_group": name,
                }
                layer_results.append(result)

                feature_pred_dict["ablation_" + name] = GS_ablate.predict(
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


def parse_args():
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
                    f"All selected layers are invalid for model {modelname} with max layers {max_layers - 1}"
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
    logger.info(
        f"Using selected layers: {select_layers if select_layers is not None else 'all layers'}"
    )
    logger.info("-" * 30)

    results_path = os.path.realpath(RESULTS_ROOT)
    # Make sure the parent path of results_path exists
    if not os.path.exists(os.path.dirname(results_path)):
        raise ValueError(f"Results path {results_path} does not exist.")
    # Create a subdirectory for the current experiment with separate librispeech split and modelname
    results_path = os.path.join(
        results_path,
        f"librispeech-{librispeech_split}/{modelname.split('/')[-1]}/{probe_name}_frame_probe_{normalize_string}",
    )
    os.makedirs(results_path, exist_ok=True)

    logger.info(f"Results will be saved to {results_path}")
    logger.info("-" * 30)

    logger.info("Formatting data for probe...")
    from load_probe_data import load_data

    processed_X, processed_Y, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling="random_frames",
        select_layers=select_layers,
        normalize_features=normalize_features,
        overwrite=args.overwrite,
    )

    logger.info("Running probe...")
    results = run_probe(
        processed_X=processed_X,
        processed_y=processed_Y,
        filename_timestamp=filename_timestamp,
        probe_name=probe_name,
        select_layers=select_layers,
        zeroing=zeroing,
        ablation=ablation,
        permutation=permutation,
        results_path=results_path,
        save_predictions=save_predictions,
    )

    logger.info("Saving all results...")
    df = pd.DataFrame(results)
    # df.drop(columns=["coefficients"], inplace=True)
    df.to_csv(
        f"{results_path}/all_layers_results_{normalize_string}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
