# Use Acoustic features as the baseline for EncoderProbe
# Add additional features on top of acoustic features to show how much extra information each feature contributes on top of

import logging
import os
import pickle
import sys
from itertools import combinations

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm, trange

from load_probe_data import get_section_shapes, load_data
from utils import RESULTS_ROOT, SAVEPATH, parse_args, pick_probe, r2_score

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def run_acoustic_base_probe(
    feature_sets: np.ndarray,
    model_hidden_states: np.ndarray,
    section_shapes: np.ndarray,
    speaker_ID: list[str],
    feature_group_config: list[list[str]],
    librispeech_split,
    modelname,
    estimator,
    param_grid,
):
    results = []

    acoustic_feature_name = "eGeMAPSv02"
    acoustic_start = int(section_shapes[0][0])
    acoustic_end = int(section_shapes[0][1])

    # Turn section_shapes into a dict for easy access
    start_end_idx = {}
    for start, end, name in section_shapes:
        start_end_idx[str(name)] = (int(start), int(end))

    acoustic_features = feature_sets[:, acoustic_start:acoustic_end]

    logger.info(
        "Running experiments with acoustic features as baseline. Adding additional features on top of acoustic features."
    )
    for feature_group in tqdm(
        feature_group_config, desc="Feature Groups", position=0, leave=True
    ):
        tqdm.write(f"Running feature group: {feature_group}")

        # Determine the indices for the current feature group
        selected_feature_indices = []
        for feature_name in feature_group:
            start, end = start_end_idx[feature_name]
            selected_feature_indices.extend(list(range(start, end)))
        # Combine acoustic features with the selected feature group
        selected_features = np.concatenate(
            (acoustic_features, feature_sets[:, selected_feature_indices]), axis=1
        )
        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=1, leave=False
        ):
            config_name = "+".join(feature_group)
            tqdm.write(f"Running layer {layer} with feature group {config_name}")

            # Proceed with training and evaluation using selected_features
            x_train, x_test, y_train, y_test, _, _ = train_test_split(
                selected_features,
                model_hidden_states[:, layer, :],
                speaker_ID,
                test_size=0.2,
                random_state=42,
                stratify=speaker_ID,
            )
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="r2",
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_

            y_test_pred = best_model.predict(x_test)
            test_score = r2_score(
                y_test,
                y_test_pred,
                multioutput="variance_weighted",
                train_data=y_train,
            )
            train_score = r2_score(
                y_train,
                best_model.predict(x_train),
                multioutput="variance_weighted",
                train_data=y_train,
            )

            result = {
                "librispeech_split": librispeech_split,
                "modelname": modelname,
                "layer": layer,
                "config_name": config_name,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": grid_search.best_params_,
                "mode": "acoustic-baseline-addition",
            }
            results.append(result)

    # Do a baseline with only acoustic features
    logger.info("Running baseline with only acoustic features")
    for layer in trange(
        model_hidden_states.shape[1], desc="Layers", position=1, leave=False
    ):
        config_name = "AcousticOnly"
        # Proceed with training and evaluation using selected_features
        x_train, x_test, y_train, y_test = train_test_split(
            acoustic_features,
            model_hidden_states[:, layer, :],
            test_size=0.2,
            random_state=42,
        )
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="r2",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_

        y_test_pred = best_model.predict(x_test)
        test_score = r2_score(
            y_test,
            y_test_pred,
            multioutput="variance_weighted",
            train_data=y_train,
        )
        train_score = r2_score(
            y_train,
            best_model.predict(x_train),
            multioutput="variance_weighted",
            train_data=y_train,
        )

        result = {
            "librispeech_split": librispeech_split,
            "modelname": modelname,
            "layer": layer,
            "config_name": config_name,
            "train_score": train_score,
            "test_score": test_score,
            "best_params": grid_search.best_params_,
            "mode": "acoustic-baseline-addition",
        }
        results.append(result)
    return results


def run_topdown_probe(
    feature_sets: np.ndarray,
    model_hidden_states: np.ndarray,
    section_shapes: np.ndarray,
    speaker_ID: list[str],
    feature_group_config: list[list[str]],
    librispeech_split,
    modelname,
    estimator,
    param_grid,
    do_topline: bool = True,
):
    results = []

    # Turn section_shapes into a dict for easy access
    start_end_idx = {}
    for start, end, name in section_shapes:
        start_end_idx[str(name)] = (int(start), int(end))
    logger.info(
        "Running experiments with top-down approach. Removing features from the full feature set to see how much information is lost by removing each feature group."
    )
    for feature_group in tqdm(
        feature_group_config, desc="Feature Groups", position=0, leave=True
    ):
        tqdm.write(f"Running feature group: {feature_group}")

        # Determine the indices for the current feature group
        selected_feature_indices = []
        for feature_name in feature_group:
            start, end = start_end_idx[feature_name]
            selected_feature_indices.extend(list(range(start, end)))
        # We use a mask that marks selected_features as 0 so these features are not used
        selection_mask = np.ones(feature_sets.shape[1], dtype=bool)
        selection_mask[selected_feature_indices] = False
        # Select features that are NOT in selected_feature_indices
        selected_features = feature_sets[:, selection_mask]
        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=1, leave=False
        ):
            config_name = "+".join(feature_group)
            tqdm.write(f"Running layer {layer} with feature group {config_name}")

            # Proceed with training and evaluation using selected_features
            x_train, x_test, y_train, y_test, _, _ = train_test_split(
                selected_features,
                model_hidden_states[:, layer, :],
                speaker_ID,
                test_size=0.2,
                random_state=42,
                stratify=speaker_ID,
            )
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="r2",
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_

            y_test_pred = best_model.predict(x_test)
            test_score = r2_score(
                y_test,
                y_test_pred,
                multioutput="variance_weighted",
                train_data=y_train,
            )
            train_score = r2_score(
                y_train,
                best_model.predict(x_train),
                multioutput="variance_weighted",
                train_data=y_train,
            )

            result = {
                "librispeech_split": librispeech_split,
                "modelname": modelname,
                "layer": layer,
                "config_name": config_name,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": grid_search.best_params_,
                "mode": "top-down",
            }
            results.append(result)

    # Do a topline with All features if do_topline is True
    if do_topline:
        logger.info("Running topline with all features")
        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=1, leave=False
        ):
            config_name = "AllFeatures"

            # Proceed with training and evaluation using all availale features
            x_train, x_test, y_train, y_test = train_test_split(
                feature_sets,
                model_hidden_states[:, layer, :],
                test_size=0.2,
                random_state=42,
            )
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="r2",
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_

            y_test_pred = best_model.predict(x_test)
            test_score = r2_score(
                y_test,
                y_test_pred,
                multioutput="variance_weighted",
                train_data=y_train,
            )
            train_score = r2_score(
                y_train,
                best_model.predict(x_train),
                multioutput="variance_weighted",
                train_data=y_train,
            )

            result = {
                "librispeech_split": librispeech_split,
                "modelname": modelname,
                "layer": layer,
                "config_name": config_name,
                "train_score": train_score,
                "test_score": test_score,
                "best_params": grid_search.best_params_,
                "mode": "top-down",
            }
            results.append(result)
    return results


def syntax_decoder_probe_baseline():
    # We load the data and see how much syntax we can decode with just
    # the word embedding itself

    input_feature_select_components = ["word_embedding", "syntax_feature"]

    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split="train-clean-100",
        selected_input_components=input_feature_select_components,
        one_hot_encode_syntax_separate=True,
        one_hot_encode_metadata=False,
        one_hot_encode_syntax=False,
        argmax_ppg=False,
        normalize_features=False,
        reduce_dnn_word_embedding=False,
    )

    # Turn syntax_features into class labels by taking the argmax across the syntax feature dimensions
    section_shapes = get_section_shapes(data_shape=data_shape)
    lookup_dict = {}
    for start, end, name in section_shapes:
        if "syntax_" in name:
            syntax_features = feature_sets[:, int(start) : int(end)]
            class_labels = np.argmax(syntax_features, axis=1)
            lookup_dict[name] = class_labels

    # Use a probe classifier to see how well we can decode the syntax features using just the word embedding features
    results = []
    for name in lookup_dict:
        estimator, param_grid = pick_probe(probe_name="ridge_classifier")

        X_train, X_test, y_train, y_test = train_test_split(
            feature_sets[
                :, : int(section_shapes[0][1])
            ],  # Only use word embedding features
            lookup_dict[name],
            test_size=0.2,
            random_state=42,
        )
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        accuracy = best_model.score(X_test, y_test)

        _, count = np.unique(y_test, return_counts=1)  # type: ignore
        majority_baseline = float(max(count) / sum(count))

        results.append(
            {
                "feature_name": name,
                "best_params": grid_search.best_params_,
                "test_score": accuracy,
                "baseline": majority_baseline,
            }
        )
    # Save the results to a text file
    output_file = os.path.join(
        RESULTS_ROOT, "syntax_feat_removal", "syntax_decoder_probe_baseline_results.txt"
    )
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Feature Name: {result['feature_name']}\n")
            f.write(f"Best Params: {result['best_params']}\n")
            f.write(f"Test Accuracy: {result['test_score']}\n")
            f.write(f"Majority Baseline: {result['baseline']}\n")
            f.write("\n")


def main():
    args = parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname
    probe_name = args.probe_name
    select_layers = args.select_layers
    normalize_features = args.normalize_features
    overwrite = args.overwrite

    input_feature_select_components = [
        "eGeMAPSv02",
        "syntax_feature",
        "ppg_feature",
        "metadata",
        "word_embedding",
    ]

    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        selected_input_components=input_feature_select_components,
        seq_sampling="random_frames",
        select_layers=select_layers,
        overwrite=overwrite,
        one_hot_encode_syntax=True,
        one_hot_encode_syntax_separate=False,
        one_hot_encode_metadata=True,
        argmax_ppg=False,
        normalize_features=normalize_features,
    )

    speaker_ID = [x[0].split("-")[0] for x in filename_timestamp]

    feature_groups = [(x,) for x in list(data_shape.keys())]
    section_shapes = np.array(get_section_shapes(data_shape=data_shape))

    estimator, param_grid = pick_probe(probe_name=probe_name, n_components=None)

    possible_feature_group_names = [
        x for x in list(data_shape.keys()) if x != "eGeMAPSv02"
    ]

    feature_group_config = [
        list(x) for x in combinations(possible_feature_group_names, 1)
    ]

    # acoustic_baseline_results = run_acoustic_base_probe(
    #     feature_sets=feature_sets,
    #     model_hidden_states=model_hidden_states,
    #     section_shapes=section_shapes,
    #     speaker_ID=speaker_ID,
    #     feature_group_config=feature_group_config,
    #     librispeech_split=librispeech_split,
    #     modelname=modelname,
    #     estimator=estimator,
    #     param_grid=param_grid,
    # )

    single_feat_removal_results = run_topdown_probe(
        feature_sets=feature_sets,
        model_hidden_states=model_hidden_states,
        section_shapes=section_shapes,
        do_topline=True,
        speaker_ID=speaker_ID,
        feature_group_config=feature_group_config
        + [
            [
                "eGeMAPSv02",
            ],
        ],
        librispeech_split=librispeech_split,
        modelname=modelname,
        estimator=estimator,
        param_grid=param_grid,
    )

    savepath = os.path.join(
        RESULTS_ROOT,
        "single_feat_removal",
        f"{librispeech_split}_{modelname.replace('/', '-')}_{probe_name}_results.pkl",
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "wb") as f:
        pickle.dump(
            single_feat_removal_results,
            f,
        )
    logger.info(f"Results saved to {savepath}")

    syntax_feature_removal_config = [
        ["syntax_feature"],
        ["syntax_feature", "word_embedding"],
    ]

    syntax_feat_removal_results = run_topdown_probe(
        feature_sets=feature_sets,
        model_hidden_states=model_hidden_states,
        section_shapes=section_shapes,
        do_topline=False,
        speaker_ID=speaker_ID,
        feature_group_config=syntax_feature_removal_config,
        librispeech_split=librispeech_split,
        modelname=modelname,
        estimator=estimator,
        param_grid=param_grid,
    )

    syntax_save_path = os.path.join(
        RESULTS_ROOT,
        "syntax_feat_removal",
        f"{librispeech_split}_{modelname.replace('/', '-')}_{probe_name}_results.pkl",
    )
    os.makedirs(os.path.dirname(syntax_save_path), exist_ok=True)
    with open(syntax_save_path, "wb") as f:
        pickle.dump(
            syntax_feat_removal_results,
            f,
        )
    logger.info(f"Syntax feature removal results saved to {syntax_save_path}")

    speakerid_phonetic_acoustic_removal_config = [
        ["ppg_feature", "eGeMAPSv02"],
        ["ppg_feature", "SpeakerID-OH"],
        ["SpeakerID-OH", "eGeMAPSv02"],
        ["SpeakerID-OH", "eGeMAPSv02", "ppg_feature"],
    ]

    speakerid_phonetic_acoustic_removal_results = run_topdown_probe(
        feature_sets=feature_sets,
        model_hidden_states=model_hidden_states,
        section_shapes=section_shapes,
        do_topline=False,
        speaker_ID=speaker_ID,
        feature_group_config=speakerid_phonetic_acoustic_removal_config,
        librispeech_split=librispeech_split,
        modelname=modelname,
        estimator=estimator,
        param_grid=param_grid,
    )
    speakerid_phonetic_acoustic_save_path = os.path.join(
        RESULTS_ROOT,
        "speakerid_phonetic_acoustic_removal",
        f"{librispeech_split}_{modelname.replace('/', '-')}_{probe_name}_results.pkl",
    )
    os.makedirs(os.path.dirname(speakerid_phonetic_acoustic_save_path), exist_ok=True)
    with open(speakerid_phonetic_acoustic_save_path, "wb") as f:
        pickle.dump(
            speakerid_phonetic_acoustic_removal_results,
            f,
        )
    logger.info(
        f"SpeakerID, Phonetic, Acoustic feature removal results saved to {speakerid_phonetic_acoustic_save_path}"
    )


if __name__ == "__main__":
    main()
