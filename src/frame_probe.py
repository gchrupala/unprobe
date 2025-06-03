import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm

from probe import PROJECT_ROOT, load_data

librispeech_split = "dev-clean"
probe_data = load_data(librispeech_split=librispeech_split)

with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_audio_representation_full.pickle",
    "rb",
) as f:
    audio_rep = pickle.load(f)
with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_opensmile_features_lld.pickle",
    "rb",
) as f:
    lld = pickle.load(f)


# Load the word and phone embedding weights along with the dictionaries
with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_words_embedding_dict.pickle",
    "rb",
) as f:
    word_dict = pickle.load(f)
with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_phones_embedding_dict.pickle",
    "rb",
) as f:
    phone_dict = pickle.load(f)
with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_words_embedding_weights.pickle",
    "rb",
) as f:
    word_embedding_weights = pickle.load(f)
word_embedding = torch.nn.Embedding.from_pretrained(
    torch.tensor(word_embedding_weights, dtype=torch.float32),
    freeze=True,
)

with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_phones_embedding_weights.pickle",
    "rb",
) as f:
    phone_embedding_weights = pickle.load(f)
phone_embedding = torch.nn.Embedding.from_pretrained(
    torch.tensor(phone_embedding_weights, dtype=torch.float32),
    freeze=True,
)


def find_interval_at_time(tier, timestamp_sec):
    """
    Finds the interval in a tier that contains the given timestamp.

    Args:
        tier: A textgrids.Tier object.
        timestamp_sec: The timestamp in seconds.

    Returns:
        The matching Interval object, or None if no interval contains the timestamp.
    """
    # Iterate through intervals and check if timestamp falls within [start, end)
    # Praat intervals are typically inclusive of the start, exclusive of the end
    for interval in tier:
        if interval.xmin <= timestamp_sec < interval.xmax:
            if (
                interval.text == ""
                or interval.text == "sil"
                or interval.text == "sp"
                or interval.text == "<unk>"
            ):
                # If the interval is empty or silent, we return padding token
                return "<pad>"
            # If the interval is not empty, we return the text
            # associated with the interval
            else:
                return interval.text
    return None  # No interval found at this timestamp


transcription_file = (
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_transcriptions.pickle"
)
with open(transcription_file, "rb") as f:
    transcription_raw = pickle.load(f)

# Sort the list of dictionary by the fileid key
transcription = sorted(
    transcription_raw,
    key=lambda x: x["fileid"],
)

# Remove entries where the length of words and syntax_feats are not equal
transcription = [x for x in transcription if len(x["words"]) == len(x["syntax_feats"])]

transcription = pd.DataFrame(transcription)

valid_fileids = transcription["fileid"].unique().tolist()


# sort the lld by fileid
lld = lld.sort_values(["file", "start"])
fileids = lld.index.get_level_values("file").unique().tolist()


processed_X = []
processed_y = []
for i in tqdm(range(len(fileids))):
    fileid = fileids[i]
    bare_fileid = os.path.split(fileid)[-1].split(".")[0]
    if bare_fileid not in valid_fileids:
        continue
    x = lld.loc[fileid].reset_index()
    # Remove start and end columns from x
    y = np.moveaxis(audio_rep[i], 0, 1)

    textgrid = transcription.loc[
        (transcription["fileid"] == bare_fileid), "textgrid"
    ].item()
    metadata = transcription.loc[
        (transcription["fileid"] == bare_fileid), "non_acoustic"
    ].item()
    syntax_feats = transcription.loc[
        (transcription["fileid"] == bare_fileid), "syntax_feats"
    ].item()

    ## convert wav2vec2 frame to ms
    w2v2_time = np.array(np.asarray(list(range(y.shape[0]))) * 20, dtype=int)
    # convert opensmile frame to ms in integer
    x["start_ms"] = (x["start"] / np.timedelta64(1, "ns") / 1e6).astype(int)

    # Find the union of the two time intervals
    viable_timestamp = np.intersect1d(x["start_ms"].to_numpy(), w2v2_time)

    # Randomly sample 5 frames from the x
    random_choice = np.random.choice(viable_timestamp, size=5, replace=False)
    # sort the random_choice
    random_choice = np.sort(random_choice)

    for start_ms in random_choice:
        start = int((start_ms - 20) / 10)
        end = int((start_ms + 20) / 10 + 1)
        if start < 0 or end > x.shape[0]:
            continue
        x_frame = x.iloc[start:end]
        all_frame = x_frame.drop(columns=["start", "end", "start_ms"]).to_numpy()
        if all_frame.shape[0] != 5:
            continue
        # concatenate the previous, current, and next frame together
        concatenated_x = np.concat(all_frame)

        # find the corresponding word in the textgrid
        word_str = find_interval_at_time(textgrid["words"], start_ms / 1000)
        word = torch.tensor(word_dict.index(word_str))
        phone_str = find_interval_at_time(textgrid["phones"], start_ms / 1000)
        phone = torch.tensor(phone_dict.index(phone_str))
        # If word is <pad>, there's no syntax features
        if word == 0:
            syntax_feat = np.zeros((syntax_feats.shape[1]))
        else:
            sent = [
                x.text for x in textgrid["words"] if x.text != "" and x.text != "<unk>"
            ]
            # Find the index of word_str in sent
            word_idx = sent.index(word_str)
            syntax_feat = syntax_feats[word_idx]

        # Embed the word and phone
        word_embedding_tensor = word_embedding(word)
        phone_embedding_tensor = phone_embedding(phone)

        # Concatenate word, phone, audio features and metadata together
        concatenated_x = np.concatenate(
            (
                concatenated_x,
                word_embedding_tensor.numpy(),
                phone_embedding_tensor.numpy(),
                np.array(metadata),
                syntax_feat,
            )
        )

        processed_X.append(concatenated_x)
        processed_y.append(y[w2v2_time == start_ms])


processed_X = np.array(processed_X)
processed_y = np.array(processed_y).squeeze()

results = []
for layer in range(processed_y.shape[1]):
    print(f"Layer {layer}")
    GS = GridSearchCV(
        estimator=Ridge(),
        param_grid={
            "alpha": [10**x for x in range(-5, 3)],
            # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "max_iter": [1000, 2000, 3000],
        },
        n_jobs=-1,
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
        "coefficients": GS.best_estimator_.coef_,
        # "intercept": GS.best_estimator_.intercept_,
        "manipulation_mode": "none",
        "manipulated_feature_group": "none",
    }
    # plot_coefficients(GS.best_estimator_.coef_)
    results.append(result)

    for range_start, range_end, name in [
        (0, 125, "acoustic"),
        (125, 225, "word_embedding"),
        (225, 325, "phone_embedding"),
        (325, 327, "metadata"),
        (327, processed_X.shape[1], "syntax_features"),
    ]:
        # Permutation of features
        permuted_x_train = X_train.copy()
        permuted_x_train[:, range_start:range_end] = np.random.permutation(
            permuted_x_train[:, range_start:range_end]
        )
        # print()
        result = {
            "layer": layer,
            "train_score": GS.score(X_train, y_train),
            "test_score": GS.score(permuted_x_train, y_train),
            "best_params": GS.best_params_,
            "best_score": GS.best_score_,
            "coefficients": GS.best_estimator_.coef_,
            # "intercept": GS.best_estimator_.intercept_,
            # "permutation": f"{range_start}-{range_end}"
            "manipulation_mode": "permutation",
            "manipulated_feature_group": name,
        }
        results.append(result)

        # Ablation of features
        ablated_x_train = X_train.copy()
        ablated_x_train = np.delete(
            ablated_x_train, np.s_[range_start:range_end], axis=1
        )
        ablated_x_test = X_test.copy()
        ablated_x_test = np.delete(ablated_x_test, np.s_[range_start:range_end], axis=1)
        GS_ablate = GridSearchCV(
            estimator=Ridge(),
            param_grid={
                "alpha": [10**x for x in range(-5, 3)],
                # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
                # "max_iter": [1000, 2000, 3000],
            },
            n_jobs=-1,
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
            "coefficients": GS_ablate.best_estimator_.coef_,
            "manipulation_mode": "ablation",
            "manipulated_feature_group": name
        }
        results.append(result)



df = pd.DataFrame(results)
df.drop(columns=["coefficients"], inplace=True)
df.to_csv(
    f"{PROJECT_ROOT}/results/librispeech-{librispeech_split}_frame_probe_results.csv",
    index=False,
)

# Make a new column in df that computes the difference between best_score and test_score
df["score_diff"] = (df["best_score"] - df["test_score"]) / df["best_score"]


# Plot the results, using layers as the x-axis and score_diff as the y-axis with permutation as hue
def plot_results(df, manipulation_mode=None):
    # Ignore the 'none' manipulation mode for the plot
    no_manip_df = df[df["manipulation_mode"] == "none"].copy()
    if manipulation_mode is not None:
        df = df[df["manipulation_mode"] == manipulation_mode].copy()
    # Put the manipulation mode under a facet grid
    plt.figure(figsize=(10, 8))
    g = sns.FacetGrid(df, col="manipulation_mode", hue="manipulated_feature_group", height=4, aspect=1)
    g.map(sns.lineplot, "layer", "test_score", marker="o")
    # Add a baseline with a different color and linestyle based on the baseline
    sns.lineplot(
        data=no_manip_df,
        x="layer",
        y="test_score",
        color="black",
        linestyle="--",
        label="Baseline (No Manip.)",
    )
    # Move legend to the side instead of on the figure
    plt.legend(title="Feature Group", bbox_to_anchor=(1.05, 1), loc=2)
    if manipulation_mode:
        plt.title(f"Encoding probe test scores with {manipulation_mode.capitalize()} manipulation")
    plt.grid(True)
    plt.show()

plot_results(df, "ablation")
plot_results(df, "permutation")


def plot_coefficients(coefficients):
    # Regressor coefficients sanity check visualization
    # First we aggregate the coefficients for each feature group
    acoustic_features = coefficients[:, : lld.shape[1] * 5]
    word_embedding_features = coefficients[
        :, lld.shape[1] * 5 : lld.shape[1] * 5 + word_embedding_weights.shape[1]
    ]
    phone_embedding_features = coefficients[
        :,
        lld.shape[1] * 5 + word_embedding_weights.shape[1] : lld.shape[1] * 5
        + word_embedding_weights.shape[1]
        + phone_embedding_weights.shape[1],
    ]
    metadata_features = coefficients[
        :,
        lld.shape[1] * 5
        + word_embedding_weights.shape[1]
        + phone_embedding_weights.shape[1] : -8,
    ]
    syntax_features = coefficients[:, -8:]
    # Sum the coefficients for each feature group
    acoustic_features = np.mean(acoustic_features, axis=1)
    word_embedding_features = np.mean(word_embedding_features, axis=1)
    phone_embedding_features = np.mean(phone_embedding_features, axis=1)
    metadata_features = np.mean(metadata_features, axis=1)
    syntax_features = np.mean(syntax_features, axis=1)

    stacked_aggrgegated = np.stack(
        (
            acoustic_features,
            word_embedding_features,
            phone_embedding_features,
            metadata_features,
            syntax_features,
        ),
        axis=0,
    )

    columns = [
        "Acoustic Features",
        "Word Embedding",
        "Phone Embedding",
        "Metadata",
        "Syntax Features",
    ]
    # plot the coefficients in heatmap
    plt.figure(figsize=(20, 16))
    # sns.heatmap(
    #     plot_df,
    #     cmap="coolwarm",
    #     # annot=True,
    #     xticklabels=plot_df.columns,
    # )
    sns.heatmap(
        stacked_aggrgegated.mean(-1).T, cmap="coolwarm", annot=True, xticklabels=columns
    )
    plt.title("Feature Correlation Matrix")
    plt.show()


results_with_all_features = [
    result for result in results if result["permutation"] == "none"
]
coefficients = [result["coefficients"] for result in results_with_all_features]
coefficients = np.stack(coefficients)
coefficients = np.moveaxis(coefficients, 1, -1)

# Plot the coefficients for each layer
plot_coefficients(coefficients)

# Feature check
feature_names = lld.columns.tolist() * 5
correlation_matrix = np.corrcoef(X_train.T)
plt.figure(figsize=(20, 16))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    xticklabels=feature_names,
    yticklabels=feature_names,
)
plt.title("Feature Correlation Matrix")
plt.show()
