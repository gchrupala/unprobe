import logging
import os
import pickle
import sys
from itertools import combinations

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm, trange

from load_probe_data import get_section_shapes, load_data
from probe_runner import (
    build_feature_lookup,
    drop_feature_groups,
    run_standard_probe,
    select_feature_groups,
)
from utils import RESULTS_ROOT, parse_args, pick_probe

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
    speaker_ids: list[str],
    feature_group_config: list[list[str]],
    librispeech_split: str,
    modelname: str,
    estimator,
    param_grid,
):
    lookup = build_feature_lookup(section_shapes)
    acoustic_features = select_feature_groups(feature_sets, lookup, ["eGeMAPSv02"])
    results: list[dict] = []

    logger.info(
        "Running experiments with acoustic features as baseline. Adding additional features on top of acoustic features."
    )
    for feature_group in tqdm(
        feature_group_config, desc="Feature Groups", position=0, leave=True
    ):
        config_name = "+".join(feature_group)
        extra_features = select_feature_groups(feature_sets, lookup, feature_group)
        selected_features = np.concatenate((acoustic_features, extra_features), axis=1)

        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=1, leave=False
        ):
            probe_result = run_standard_probe(
                estimator,
                param_grid,
                selected_features,
                model_hidden_states[:, layer, :],
                stratify_labels=speaker_ids,
            )
            results.append(
                {
                    "librispeech_split": librispeech_split,
                    "modelname": modelname,
                    "layer": layer,
                    "config_name": config_name,
                    "train_score": probe_result["train_score"],
                    "test_score": probe_result["test_score"],
                    "best_params": probe_result["best_params"],
                    "mode": "acoustic-baseline-addition",
                }
            )

    logger.info("Running baseline with only acoustic features")
    for layer in trange(
        model_hidden_states.shape[1], desc="Layers", position=1, leave=False
    ):
        probe_result = run_standard_probe(
            estimator,
            param_grid,
            acoustic_features,
            model_hidden_states[:, layer, :],
            stratify_labels=speaker_ids,
        )
        results.append(
            {
                "librispeech_split": librispeech_split,
                "modelname": modelname,
                "layer": layer,
                "config_name": "AcousticOnly",
                "train_score": probe_result["train_score"],
                "test_score": probe_result["test_score"],
                "best_params": probe_result["best_params"],
                "mode": "acoustic-baseline-addition",
            }
        )
    return results


def run_topdown_probe(
    feature_sets: np.ndarray,
    model_hidden_states: np.ndarray,
    section_shapes: np.ndarray,
    speaker_ids: list[str],
    feature_group_config: list[list[str]],
    librispeech_split: str,
    modelname: str,
    estimator,
    param_grid,
    do_topline: bool = True,
):
    lookup = build_feature_lookup(section_shapes)
    results: list[dict] = []

    logger.info(
        "Running experiments with top-down approach. Removing features from the full feature set to see how much information is lost by removing each feature group."
    )
    for feature_group in tqdm(
        feature_group_config, desc="Feature Groups", position=0, leave=True
    ):
        config_name = "+".join(feature_group)
        reduced_features = drop_feature_groups(feature_sets, lookup, feature_group)
        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=1, leave=False
        ):
            probe_result = run_standard_probe(
                estimator,
                param_grid,
                reduced_features,
                model_hidden_states[:, layer, :],
                stratify_labels=speaker_ids,
            )
            results.append(
                {
                    "librispeech_split": librispeech_split,
                    "modelname": modelname,
                    "layer": layer,
                    "config_name": config_name,
                    "train_score": probe_result["train_score"],
                    "test_score": probe_result["test_score"],
                    "best_params": probe_result["best_params"],
                    "mode": "top-down",
                }
            )

    if do_topline:
        logger.info("Running topline with all features")
        for layer in trange(
            model_hidden_states.shape[1], desc="Layers", position=1, leave=False
        ):
            probe_result = run_standard_probe(
                estimator,
                param_grid,
                feature_sets,
                model_hidden_states[:, layer, :],
                stratify_labels=speaker_ids,
            )
            results.append(
                {
                    "librispeech_split": librispeech_split,
                    "modelname": modelname,
                    "layer": layer,
                    "config_name": "AllFeatures",
                    "train_score": probe_result["train_score"],
                    "test_score": probe_result["test_score"],
                    "best_params": probe_result["best_params"],
                    "mode": "top-down",
                }
            )
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


def run_lexicon_syntax_decomposition_experiment(
    librispeech_split: str,
    modelname: str,
    probe_name: str,
    select_layers: list[int] | None,
    overwrite: bool,
    random_seed: int,
    normalize_features: bool,
) -> None:
    """Run focused syntax ablations on top of the -Lexicon condition.

    This experiment encodes the restricted syntax dimensions separately and then
    removes each syntax subfeature in addition to lexical removal.
    """

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
        one_hot_encode_syntax=False,
        one_hot_encode_syntax_separate=True,
        one_hot_encode_metadata=True,
        argmax_ppg=False,
        normalize_features=normalize_features,
        random_seed=random_seed,
    )

    speaker_ID = [x[0].split("-")[0] for x in filename_timestamp]
    section_shapes = np.array(get_section_shapes(data_shape=data_shape))
    estimator, param_grid = pick_probe(probe_name=probe_name, n_components=None)

    syntax_components = [
        component
        for component in data_shape
        if component
        in {
            "syntax_POS_OH",
            "syntax_Dependency_Label_OH",
            "syntax_Tree_Depth",
            "syntax_Word_Position",
            "syntax_Total_Tree_Depth",
            "syntax_Total_Word_Count",
        }
    ]

    if len(syntax_components) == 0:
        raise ValueError(
            "No separated syntax components found. Expected one_hot_encode_syntax_separate output."
        )

    feature_group_config = [["word_embedding"]] + [
        ["word_embedding", syntax_component] for syntax_component in syntax_components
    ]

    results = run_topdown_probe(
        feature_sets=feature_sets,
        model_hidden_states=model_hidden_states,
        section_shapes=section_shapes,
        do_topline=False,
        speaker_ids=speaker_ID,
        feature_group_config=feature_group_config,
        librispeech_split=librispeech_split,
        modelname=modelname,
        estimator=estimator,
        param_grid=param_grid,
    )

    savepath = os.path.join(
        RESULTS_ROOT,
        "syntax_lexicon_decomposition",
        f"{librispeech_split}_{modelname.replace('/', '-')}_{probe_name}_results.pkl",
    )
    if random_seed != 42:
        savepath = savepath.replace("results.pkl", f"results-seed{random_seed}.pkl")

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "wb") as f:
        pickle.dump(results, f)
    logger.info("Syntax/lexicon decomposition results saved to %s", savepath)


def main():
    args = parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname
    probe_name = args.probe_name
    select_layers = args.select_layers
    normalize_features = args.normalize_features
    overwrite = args.overwrite
    random_seed = args.random_seed

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
        random_seed=random_seed,
    )

    speaker_ID = [x[0].split("-")[0] for x in filename_timestamp]

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
        speaker_ids=speaker_ID,
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

    # Rename savepath in case random_seed is not default
    if random_seed != 42:
        savepath = savepath.replace("results.pkl", f"results-seed{random_seed}.pkl")

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
        speaker_ids=speaker_ID,
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

    # Rename savepath in case random_seed is not default
    if random_seed != 42:
        syntax_save_path = syntax_save_path.replace(
            "results.pkl", f"results-seed{random_seed}.pkl"
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
        speaker_ids=speaker_ID,
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
    # Rename savepath in case random_seed is not default
    if random_seed != 42:
        speakerid_phonetic_acoustic_save_path = (
            speakerid_phonetic_acoustic_save_path.replace(
                "results.pkl", f"results-seed{random_seed}.pkl"
            )
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

    run_lexicon_syntax_decomposition_experiment(
        librispeech_split=librispeech_split,
        modelname=modelname,
        probe_name=probe_name,
        select_layers=select_layers,
        overwrite=overwrite,
        random_seed=random_seed,
        normalize_features=normalize_features,
    )


if __name__ == "__main__":
    main()
