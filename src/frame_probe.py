import pickle

import numpy as np
from sklearn.linear_model import Ridge
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


fileids = lld.index.get_level_values("file").unique().tolist()
processed_X = []
processed_y = []
for i, fileid in tqdm(enumerate(fileids)):
    x = lld.loc[fileid].reset_index()
    y = audio_rep[i][-1]

    w2v2_time = np.array(np.asarray(list(range(y.shape[0]))) * 20, dtype=np.float64)
    x["start_ms"] = x["start"] / np.timedelta64(1, "ns") / 1e6
    # Filter x so that it only contains the same time points as w2v2_time
    x = x[x["start_ms"].isin(w2v2_time)]
    # Filter y so that only the same time points as x
    y = y[np.isin(w2v2_time, x["start_ms"].values)]
    # Append the processed data to the lists
    clean_x = x.drop(columns=["start", "end", "start_ms"])
    processed_X.append(clean_x.to_numpy())
    processed_y.append(y)

# Concatenate all the processed data
processed_X = np.vstack(processed_X)
processed_y = np.concatenate(processed_y, axis=0)

GS = GridSearchCV(
    estimator=Ridge(),
    param_grid={
        "alpha": [0.1, 1, 10, 100],
        # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
        # "max_iter": [1000, 2000, 3000],
    },
    n_jobs=-1,
    cv=5,
    verbose=3,
)
X_train, X_test, y_train, y_test = train_test_split(
    processed_X, processed_y, test_size=0.2, random_state=42
)

GS.fit(X_train, y_train)
r2 = GS.score(X_test, y_test)
print(f"R2: {r2}")
