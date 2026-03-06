import os 
import glob

import torch
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


hidden_states = torch.load("/home/g.shen/unprobe/data/librispeech_dev-cleanextracted_pitch_data.pt", weights_only=False)
with open('/home/g.shen/unprobe/data/encoded_articulatory_features.pkl', 'rb') as f:
    articulatory_features = pickle.load(f)

with open('/home/g.shen/unprobe/data/librispeech-dev-clean_opensmile_features_lld.pickle', 'rb') as f:
    opensmile_features = pickle.load(f)
lld = opensmile_features.sort_values(["file", "start"])

y = []
X = []

for i in tqdm(range(len(hidden_states))):
    regression_output = hidden_states[i]['hidden_states']
    fileid = hidden_states[i]['file_path'].replace('/data', '')
    acoustic_features = lld.loc[fileid].reset_index().drop(columns=["start", "end",]).to_numpy()
    # Only keep every other frame of the acoustic features and keep the last frame just in case regression_output is longer
    last_frame_acoustic = acoustic_features[-1].copy()
    acoustic_features = acoustic_features[::2]
    if len(acoustic_features) < len(regression_output):
        # If the regression output is longer, pad the acoustic features
        acoustic_features = np.concatenate((acoustic_features, np.tile(last_frame_acoustic, (len(regression_output) - len(acoustic_features), 1))), axis=0)
    y.append(regression_output)

    ema =  articulatory_features[i]['ema']
    loudness = articulatory_features[i]['loudness']
    periodicity = articulatory_features[i]['periodicity']
    pitch = articulatory_features[i]['pitch']

    # Trim everything to the same length as ema
    min_length = min(len(ema), len(loudness), len(periodicity), len(pitch))
    ema = ema[:min_length]
    loudness = loudness[:min_length]
    periodicity = periodicity[:min_length]
    pitch = pitch[:min_length]
    # Stack the features
    X.append(np.column_stack((ema, loudness, periodicity, pitch, acoustic_features)))

X_raw = np.vstack(X)
y_raw = np.vstack(y)

# Dimensionality reduction for the target y

# from sklearn.decomposition import PCA
# pca = PCA(n_components=20)
# y = pca.fit_transform(y)
X = X_raw
y = y_raw

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor_GS = GridSearchCV(
    Ridge(),
    param_grid={
        'alpha': [0.1, 1.0, 10.0, 100.0],
    },
    cv=5,
    n_jobs=-1
)

# Fit the model
regressor_GS.fit(X_train, y_train)
# Make predictions
y_pred = regressor_GS.predict(X_test)
# Calculate the mean squared error and score
mse = mean_squared_error(y_test, y_pred)
score = regressor_GS.score(X_test, y_test)
print(f"Best parameters: {regressor_GS.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"Score: {score}")

regressor = Ridge(alpha=regressor_GS.best_params_['alpha'])
# Fit the model
regressor.fit(X_train[:,:25], y_train)
print(f"Articulatory only: {regressor.score(X_test[:,:25], y_test)}")

regressor = Ridge(alpha=regressor_GS.best_params_['alpha'])
# Fit the model
regressor.fit(X_train[:,25:], y_train)
print(f"Acoustic only: {regressor.score(X_test[:,25:], y_test)}")
regressor = Ridge(alpha=regressor_GS.best_params_['alpha'])
# Fit the model
regressor.fit(X_train, y_train)
print(f"Full model: {regressor.score(X_test, y_test)}")
# Save the model