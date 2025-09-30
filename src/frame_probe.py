import argparse
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
        100 + 125 + 8,
        "syntax_features",
    ),
    (
        100 + 125 + 8,
        100 + 125 + 8 + 40,
        "ppgs_features",
    ),
    (
        100 + 125 + 8 + 40,
        100 + 125 + 8 + 40 + 100,
        "spk_embedding",
    ),
    (
        100 + 125 + 8 + 40 + 100,
        100 + 125 + 8 + 40 + 100 + 2,
        "metadata",
    ),
)

# Setting up environmental variables depending on the cluster this code is running on

if "snellius" in hostname:
    # If running on Snellius, use the Snellius dataset root
    DATASET_ROOT = os.path.realpath("/projects/prjs1586/corpora/LibriSpeech")
    ALIGNMENT_ROOT = DATASET_ROOT.replace("LibriSpeech", "librispeech_textgrids")
    SAVEPATH = "/projects/prjs1586/experimental_data"

else:
    # If running on local machine, use the local dataset root
    DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENT_ROOT = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENT_ROOT = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")


def find_alignment_interval(
    alignment: pd.DataFrame, timestamp_sec: float
) -> str | None:
    """
    Finds the alignment interval that contains the given timestamp.

    Args:
        alignment: A list of dictionaries containing alignment information.
        timestamp_sec: The timestamp in seconds.

    Returns:
        The matching string for the alignment interval, or None if no interval contains the timestamp.
    """
    for i, entry in alignment.iterrows():
        if entry["start"] <= timestamp_sec < entry["end"]:
            if (
                entry["text"] == ""
                or entry["text"] == "sil"
                or entry["text"] == "sp"
                or entry["text"] == "<unk>"
                or str(entry["text"]) == "<p:>"
                or str(entry["text"]) == "nan"
            ):
                # If the interval is empty or silent, we return padding token
                return "<pad>"
            else:
                return entry["text"]
    return None


def format_data_for_probe(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
) -> tuple[np.ndarray, np.ndarray]:
    """Formats the data for the probing task.
    Args:
        librispeech_split: The LibriSpeech split to use.
        modelname: The name of the model to use.
    Returns:
        A tuple containing the formatted input and target data.

    """
    modelname = modelname.split("/")[-1]

    # Load the raw data
    with open(
        f"{SAVEPATH}/librispeech-{librispeech_split}_opensmile_features_lld.pickle",
        "rb",
    ) as f:
        lld = pickle.load(f)

    with open(
        f"{SAVEPATH}/librispeech-{librispeech_split}_{modelname}_representation_full.pickle",
        "rb",
    ) as f:
        dnn_hidden_states = pickle.load(f)

    with open(
        f"{SAVEPATH}/librispeech-{librispeech_split}_special_features.pickle",
        "rb",
    ) as f:
        special_features = pickle.load(f)

    transcription_file = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_transcriptions.pickle"
    )
    with open(transcription_file, "rb") as f:
        transcription_raw = pickle.load(f)

    # Sort the list of dictionary by the fileid key
    transcription = sorted(
        transcription_raw,
        key=lambda x: x["fileID"],
    )

    # Remove entries where the length of words and syntax_feats are not equal
    transcription = [
        x for x in transcription if len(x["words"]) == len(x["syntax_feats"])
    ]
    transcription = pd.DataFrame(transcription)
    valid_fileids = transcription["fileID"].unique().tolist()

    # Sort the lld by fileid
    lld = lld.sort_values(["file", "start"])
    fileids = lld.index.get_level_values("file").unique().tolist()

    processed_X = []
    processed_y = []
    for i in tqdm(range(len(fileids))):
        fileid = fileids[i]
        bare_fileid = os.path.split(fileid)[-1].split(".")[0]
        if bare_fileid not in valid_fileids:
            continue
        # Remove start and end columns from x
        utt_lld = lld.loc[fileid].reset_index()
        y = np.moveaxis(dnn_hidden_states[bare_fileid], 0, 1)

        phone_alignment: pd.DataFrame = transcription.loc[
            (transcription["fileID"] == bare_fileid), "phone_alignment"
        ].item()  # type: ignore
        ort_alignment: pd.DataFrame = transcription.loc[
            (transcription["fileID"] == bare_fileid), "ort_alignment"
        ].item()  # type: ignore

        metadata: list = transcription.loc[
            (transcription["fileID"] == bare_fileid), "non_acoustic"
        ].item()  # type: ignore
        syntax_feats: np.ndarray = transcription.loc[
            (transcription["fileID"] == bare_fileid), "syntax_feats"
        ].item()  # type: ignore

        # Convert opensmile frame to ms in integer
        utt_lld["start_ms"] = (utt_lld["start"] / np.timedelta64(1, "ns") / 1e6).astype(
            int
        )

        if "wav2vec2" in modelname:
            # Convert wav2vec2 frame to ms
            w2v2_time = np.array(np.asarray(list(range(y.shape[0]))) * 20, dtype=int)
            # Find the union of the two time intervals
            viable_timestamp = np.intersect1d(utt_lld["start_ms"].to_numpy(), w2v2_time)
            # Randomly sample 5 frames from the x
            random_choice = np.random.choice(viable_timestamp, size=5, replace=False)
            # sort the random_choice
            random_choice = np.sort(random_choice)
        else:
            # Text only models only have one embedding per word
            # We sample 5 random words from the ort_alignment
            if len(ort_alignment) < 5:
                num_choices = len(ort_alignment)
            else:
                num_choices = 5
            random_choice = np.random.choice(
                ort_alignment["start"] * 1000, size=num_choices, replace=False
            )
            random_choice = np.sort(random_choice)

        spk_embedding = special_features[bare_fileid]["spk_emb"]
        ppgs_features = special_features[bare_fileid]["ppgs"]
        fasttext_embedding = special_features[bare_fileid]["fasttext"]

        for start_ms in random_choice:
            start = int((start_ms - 20) / 10)
            end = int((start_ms + 20) / 10 + 1)
            if start < 0 or end > utt_lld.shape[0]:
                continue
            utt_lld_frames = utt_lld.iloc[start:end]
            all_utt_lld_frames = utt_lld_frames.drop(
                columns=["start", "end", "start_ms"]
            ).to_numpy()
            if all_utt_lld_frames.shape[0] != 5:
                continue
            # concatenate the previous, current, and next frame together
            concat_utt_lld = np.concat(all_utt_lld_frames)
            # Make sure start_ms is in an interval of ort_alignment
            if not any(
                (ort_alignment["start"] <= start_ms / 1000)
                & (ort_alignment["end"] > start_ms / 1000)
            ):
                continue
            # find the corresponding word in the textgrid
            word_str = find_alignment_interval(ort_alignment, start_ms / 1000)
            phone_str = find_alignment_interval(phone_alignment, start_ms / 1000)
            # Skip if word_str is "<pad>" or None
            if word_str == "<pad>" or word_str is None:
                continue

            tokens = ort_alignment["text"].fillna("<pad>").tolist()
            tokens_nopad = [str(token) for token in tokens if token != "<pad>"]
            # Find the index of word_str in sent
            word_idx = tokens.index(word_str)
            word_idx_nopad = tokens_nopad.index(word_str)
            word_embedding = fasttext_embedding[word_idx]
            # If word is <pad>, there's no syntax features
            if word_str not in ort_alignment["text"].values:
                # If the word is a padding token, we use a zero vector for syntax features
                syntax_feat = np.zeros((syntax_feats.shape[1]))
            else:
                syntax_feat = syntax_feats[word_idx_nopad]
                # phone_embedding = phone_embedding_weights[phone_idx]

            # Embed the word and phone
            # word_embedding_tensor = word_embedding(word)
            # phone_embedding_tensor = phone_embedding(phone)
            ppgs_tensor = ppgs_features[:, int(start_ms / 10)]

            # Concatenate lld, word embd, syntax, ppg, and metadata together
            concatenated_x = np.concatenate(
                (
                    concat_utt_lld,
                    word_embedding,
                    # phone_embedding_tensor.numpy(),
                    syntax_feat,
                    ppgs_tensor,
                    spk_embedding,
                    np.array(metadata),
                )
            )

            processed_X.append(concatenated_x)
            if "wav2vec2" in modelname.lower():
                processed_y.append(y[w2v2_time == start_ms])  # type: ignore
            else:
                processed_y.append(y[word_idx])

    processed_X = np.array(processed_X)
    processed_y = np.array(processed_y).squeeze()

    return processed_X, processed_y


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

        model = RandomForestRegressor(n_jobs=-1)
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
    else:
        raise ValueError(f"Probe {probe_name} not supported")

    return model, param_grid


def run_probe(
    processed_X: np.ndarray,
    processed_y: np.ndarray,
    probe_name: str = "ridge",
) -> list[dict]:
    results = []
    for layer in trange(processed_y.shape[1], desc="Layers"):
        logger.info(f"Probing with {probe_name} on DNN model layer {layer}")
        regressor, param_grid = pick_probe(probe_name)
        GS = GridSearchCV(
            estimator=regressor,
            param_grid=param_grid,
            # n_jobs=-1,
            cv=5,
            verbose=1,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            processed_X, processed_y[:, layer, :], test_size=0.2, random_state=42
        )

        GS.fit(X_train, y_train)
        train_score = GS.score(X_train, y_train)
        test_score = GS.score(X_test, y_test)
        # print(f"Train score: {train_score}")
        # print(f"Test score: {test_score}")
        # print(f"Best parameters: {GS.best_params_}")
        # print(f"Best score: {GS.best_score_}")

        result = {
            "layer": layer,
            "train_score": train_score,
            "test_score": test_score,
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            # "coefficients": GS.best_estimator_.coef_,
            # "intercept": GS.best_estimator_.intercept_,
            "manipulation_mode": "none",
            "manipulated_feature_group": "none",
        }
        results.append(result)

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
            "layer": layer,
            "train_score": random_train_score,
            "test_score": random_test_score,
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            # "coefficients": GS.best_estimator_.coef_,
            # "intercept": GS.best_estimator_.intercept_,
            "manipulation_mode": "random_baseline",
            "manipulated_feature_group": "none",
        }
        results.append(result)

        for range_start, range_end, name in tqdm(
            sections_shapes, desc="Feature Groups", leave=False
        ):
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
                "layer": layer,
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
            results.append(result)

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
                "layer": layer,
                "train_score": zeroed_train_score,
                "test_score": zeroed_test_score,
                "best_params": GS.best_params_,
                "best_score": GS.best_score_,
                # "coefficients": GS_zero.best_estimator_.coef_,
                # "intercept": GS_zero.best_estimator_.intercept_,
                "manipulation_mode": "zeroing",
                "manipulated_feature_group": name,
            }
            results.append(result)

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
                "layer": layer,
                "train_score": ablated_train_score,
                "test_score": ablated_test_score,
                "best_params": GS_ablate.best_params_,
                "best_score": GS.best_score_,
                # "coefficients": GS_ablate.best_estimator_.coef_,
                "manipulation_mode": "ablation",
                "manipulated_feature_group": name,
            }
            results.append(result)
    return results


def save_results(
    results: list[dict],
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    probe_name: str = "ridge",
) -> None:
    modelname = modelname.split("/")[-1]
    df = pd.DataFrame(results)
    # df.drop(columns=["coefficients"], inplace=True)
    df.to_csv(
        f"{PROJECT_ROOT}/results/librispeech-{librispeech_split}_{modelname}_frame_probe_{probe_name}_results.csv",
        index=False,
    )


def visualize_results(
    df: pd.DataFrame,
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    probe_name: str = "ridge",
) -> None:
    modelname = modelname.split("/")[-1]
    # df.drop(columns=["coefficients"], inplace=True)

    # Make a new column in df that computes the difference between best_score and test_score
    df["score_diff"] = (df["best_score"] - df["test_score"]) / df["best_score"]

    # Plot the results, using layers as the x-axis and score_diff as the y-axis with permutation as hue
    def plot_results(df, manipulation_mode=None, y="test_score"):
        # Ignore the 'none' manipulation mode for the plot
        no_manip_df = df[df["manipulation_mode"] == "none"].copy()
        random_baseline_df = df[df["manipulation_mode"] == "random_baseline"].copy()

        if manipulation_mode is not None:
            df = df[df["manipulation_mode"] == manipulation_mode].copy()
        # Put the manipulation mode under a facet grid
        plt.figure(figsize=(10, 8))
        g = sns.FacetGrid(
            df,
            col="manipulation_mode",
            hue="manipulated_feature_group",
            height=4,
            aspect=1,
        )
        g.map(sns.lineplot, "layer", y, marker="o")
        # Add a baseline with a different color and linestyle based on the baseline
        sns.lineplot(
            data=no_manip_df,
            x="layer",
            y=y,
            color="black",
            linestyle="--",
            label="Full Feature Set",
        )
        # Add random baseline with a different color and linestyle based on the baseline
        sns.lineplot(
            data=random_baseline_df,
            x="layer",
            y=y,
            color="red",
            linestyle="--",
            label="Random Baseline",
        )
        # Move legend to the side instead of on the figure
        plt.legend(title="Feature Group", bbox_to_anchor=(1.05, 1), loc=2)
        if manipulation_mode:
            plt.title(
                f"Encoding probe test scores with {manipulation_mode.capitalize()} manipulation"
            )
        plt.grid(True)
        plt.show()

        return g

    ablation_plot = plot_results(df, "ablation")
    permutation_plot = plot_results(df, "permutation")
    zeroing_plot = plot_results(df, "zeroing")

    plot_results(df, "ablation", "score_diff")
    plot_results(df, "permutation", "score_diff")
    plot_results(df, "zeroing", "score_diff")

    # Save the plots to the results directory
    ablation_plot.savefig(
        f"{PROJECT_ROOT}/results/librispeech-{librispeech_split}_{modelname}_frame_probe_ablation_{probe_name}_plot.png",
        bbox_inches="tight",
    )
    permutation_plot.savefig(
        f"{PROJECT_ROOT}/results/librispeech-{librispeech_split}_{modelname}_frame_probe_permutation_{probe_name}_plot.png",
        bbox_inches="tight",
    )
    zeroing_plot.savefig(
        f"{PROJECT_ROOT}/results/librispeech-{librispeech_split}_{modelname}_frame_probe_zeroing_{probe_name}_plot.png",
        bbox_inches="tight",
    )


def plot_coefficients(coefficients):
    pass
    # Regressor coefficients sanity check visualization
    # First we aggregate the coefficients for each feature group

    coefficient_dict = {}

    for range_start, range_end, name in sections_shapes:
        print(
            f"Feature group: {name}, Coefficient mean: {np.mean(np.abs(coefficients[:, range_start:range_end]))}"
        )
        coefficient_dict[name] = np.mean(np.abs(coefficients[:, range_start:range_end]))

    # Extract coefficients for each feature group using sections_shapes
    acoustic_features = coefficients[:, sections_shapes[0][0] : sections_shapes[0][1]]
    word_embedding_features = coefficients[
        :, sections_shapes[1][0] : sections_shapes[1][1]
    ]
    syntax_features = coefficients[:, sections_shapes[2][0] : sections_shapes[2][1]]
    ppgs_features = coefficients[:, sections_shapes[3][0] : sections_shapes[3][1]]
    spk_embedding_features = coefficients[
        :, sections_shapes[4][0] : sections_shapes[4][1]
    ]
    metadata_features = coefficients[:, sections_shapes[5][0] : sections_shapes[5][1]]
    # Sum the coefficients for each feature group
    acoustic_features = np.mean(acoustic_features, axis=1)
    word_embedding_features = np.mean(word_embedding_features, axis=1)
    # phone_embedding_features = np.mean(phone_embedding_features, axis=1)
    metadata_features = np.mean(metadata_features, axis=1)
    syntax_features = np.mean(syntax_features, axis=1)
    ppgs_features = np.mean(ppgs_features, axis=1)
    spk_embedding_features = np.mean(spk_embedding_features, axis=1)

    stacked_aggrgegated = np.stack(
        (
            acoustic_features,
            word_embedding_features,
            # phone_embedding_features,
            metadata_features,
            syntax_features,
            ppgs_features,
            spk_embedding_features,
        ),
        axis=0,
    )

    columns = [
        "Acoustic Features",
        "Word Embedding",
        "Metadata",
        "Syntax Features",
        "PPGs Features",
        "Speaker Embedding",
    ]
    # plot the coefficients in heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        stacked_aggrgegated.mean(-1).T, cmap="coolwarm", annot=True, xticklabels=columns
    )
    plt.title("Feature Correlation Matrix")
    plt.show()
    return coefficient_dict


if __name__ == "__main__":
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
    args = parser.parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname
    probe_name = args.probe_name

    logger.info(f"Using LibriSpeech split: {librispeech_split}")
    logger.info(f"Using model: {modelname}")
    logger.info(f"Using probe: {probe_name}")
    logger.info("-" * 30)

    logger.info("Formatting data for probe...")
    processed_X, processed_y = format_data_for_probe(
        librispeech_split=librispeech_split, modelname=modelname
    )

    logger.info("Running probe...")
    results = run_probe(
        processed_X=processed_X,
        processed_y=processed_y,
        probe_name=probe_name,
    )
    logger.info("Saving and visualizing results...")
    save_results(
        results=results,
        librispeech_split=librispeech_split,
        modelname=modelname,
        probe_name=probe_name,
    )
    visualize_results(
        df=pd.DataFrame(results),
        librispeech_split=librispeech_split,
        modelname=modelname,
        probe_name=probe_name,
    )
