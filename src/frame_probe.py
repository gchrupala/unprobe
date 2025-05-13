import pickle
import os
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
with open(
    f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_phones_embedding_weights.pickle",
    "rb",
) as f:
    phone_embedding_weights = pickle.load(f)

word_embedding = torch.nn.Embedding.from_pretrained(
    torch.tensor(word_embedding_weights, dtype=torch.float32),
    freeze=True,
)
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
            if interval.text == "" or interval.text == "sil" or interval.text == "sp":
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
    transcription = pickle.load(f)

# sort the transcription by fileid
transcription = pd.DataFrame(transcription)
transcription = transcription.sort_values(by=["fileid"])


# sort the lld by fileid
lld = lld.sort_values(["file", "start"])
fileids = lld.index.get_level_values("file").unique().tolist()


processed_X = []
processed_y = []
for i in tqdm(range(len(fileids))):
    fileid = fileids[i]
    bare_fileid = os.path.split(fileid)[-1].split(".")[0]
    x = lld.loc[fileid].reset_index()
    # Remove start and end columns from x
    y = np.moveaxis(audio_rep[i], 0, 1)

    textgrid = transcription.loc[
        (transcription["fileid"] == bare_fileid), "textgrid"
    ].item()
    metadata = transcription.loc[
        (transcription["fileid"] == bare_fileid), "non_acoustic"
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
        word = find_interval_at_time(textgrid["words"], start_ms / 1000)
        word = torch.tensor(word_dict.index(word))
        phone = find_interval_at_time(textgrid["phones"], start_ms / 1000)
        phone = torch.tensor(phone_dict.index(phone))

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
    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")
    print(f"Best parameters: {GS.best_params_}")
    print(f"Best score: {GS.best_score_}")

    result = {
        "layer": layer,
        "train_score": train_score,
        "test_score": test_score,
        "best_params": GS.best_params_,
        "best_score": GS.best_score_,
        "coefficients": GS.best_estimator_.coef_,
        # "intercept": GS.best_estimator_.intercept_,
        "permutation": 'none'
    }
    # plot_coefficients(GS.best_estimator_.coef_)
    results.append(result)

    for range_start, range_end, name in [
        (0, 125, "acoustic"),
        (125, 225, "word_embedding"),
        (225, 325, "phone_embedding"),
        (325, 327, "metadata"),
    ]:
        # Permutation test
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
            "permutation": name,
        }
        results.append(result)


df = pd.DataFrame(results)
df.drop(columns=["coefficients"], inplace=True)
df.to_csv(
    f"{PROJECT_ROOT}/results/librispeech-{librispeech_split}_frame_probe_results.csv",
    index=False,
)


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
        + phone_embedding_weights.shape[1] :,
    ]
    # Sum the coefficients for each feature group
    acoustic_features = np.mean(acoustic_features, axis=1)
    word_embedding_features = np.mean(word_embedding_features, axis=1)
    phone_embedding_features = np.mean(phone_embedding_features, axis=1)
    metadata_features = np.mean(metadata_features, axis=1)

    stacked_aggrgegated = np.stack(
        (
            acoustic_features,
            word_embedding_features,
            phone_embedding_features,
            metadata_features,
        ),
        axis=0,
    )

    plot_df = pd.DataFrame(
        stacked_aggrgegated.T,
        columns=["Acoustic Features", "Word Embedding", "Phone Embedding", "Metadata"],
    )
    # plot the coefficients in heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        plot_df,
        cmap="coolwarm",
        # annot=True,
        xticklabels=plot_df.columns,
    )
    plt.title("Feature Correlation Matrix")
    plt.show()
    

for layer in range(len(results)):
    print(f"Layer {layer}")
    plot_coefficients(results[layer]["coefficients"])

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

