import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import seaborn as sns
import torch
import torchaudio
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LS_ROOT = f"{PROJECT_ROOT}/LibriSpeech"
librispeech_split = "dev-clean"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def parselmouth_pitch_tracking(file_path):
    """
    Use Parselmouth to extract pitch from an audio file.
    Pitch array shape is (n_frames, 1) where frame size is 10ms.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        pitch_values (numpy.ndarray): Pitch values extracted from the audio file.
    """
    sound = parselmouth.Sound(file_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    return pitch_values


def extract_data():
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model.to(device)

    files = glob.glob(f"{LS_ROOT}/{librispeech_split}/**/*.flac", recursive=True)
    files.sort()

    extracted_data = []

    for file_path in tqdm(files):
        # print(file)
        pitch = parselmouth_pitch_tracking(file_path)

        waveform, sample_rate = torchaudio.load(file_path)
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            device=device,
        )
        with torch.no_grad():
            outputs = model(
                inputs["input_values"].to(device), output_hidden_states=True
            )
            hidden_states = outputs.hidden_states.squeeze().cpu().numpy()

        extracted_data.append(
            {"file_path": file_path, "pitch": pitch, "hidden_states": hidden_states}
        )

    return extracted_data


def process_raw_input():
    """
    Process raw input data to extract pitch and hidden states.
    Args:
        None
    Returns:
        hidden_states_arr (numpy.ndarray): Hidden states extracted from the audio files.
        pitch_arr (numpy.ndarray): Pitch values extracted from the audio files.
        pitch_binned (numpy.ndarray): Binned pitch values.

    """
    extracted_data = torch.load(
        f"{PROJECT_ROOT}/data/librispeech_{librispeech_split}extracted_pitch_data.pt",
        weights_only=False,
    )

    # We first put pitch and hidden states into separate lists
    pitch_list = []
    hidden_states_list = []
    for data in tqdm(extracted_data):
        pitch = data["pitch"]
        hidden_states = data["hidden_states"]

        for i in range(hidden_states.shape[0]):
            i_conform = i * 2
            frame_pitch = pitch[i_conform : i_conform + 2]
            if len(frame_pitch) < 2:
                continue
            if 0 in frame_pitch:
                frame_pitch = frame_pitch.max()
            else:
                frame_pitch = frame_pitch.mean()
            pitch_list.append(frame_pitch)
            hidden_states_list.append(hidden_states[i])

    pitch_arr = np.array(pitch_list)
    hidden_states_arr = np.array(hidden_states_list)

    # Bin the pitch values into 50 bins on a logarithmic scale
    pitch_bins = np.linspace(pitch_arr.min(), pitch_arr.max(), 50)
    pitch_binned = np.digitize(pitch_arr, pitch_bins) - 1
    pitch_binned = np.clip(
        pitch_binned, 0, 49
    )  # Ensure pitch values are within the bin range

    # Get rid of the bin that only has 1 value
    hidden_states_arr = np.delete(
        hidden_states_arr, np.where(pitch_binned == 49), axis=0
    )
    pitch_arr = np.delete(pitch_arr, np.where(pitch_binned == 49), axis=0)
    pitch_binned = np.delete(pitch_binned, np.where(pitch_binned == 49), axis=0)

    return hidden_states_arr, pitch_arr, pitch_binned


def main():
    extracted_pitch_data_path = (
        f"{PROJECT_ROOT}/data/librispeech_{librispeech_split}extracted_pitch_data.pt"
    )
    if os.path.exists(extracted_pitch_data_path):
        print(f"Extracted pitch data already exists at {extracted_pitch_data_path}.")
    else:
        print(f"Extracting pitch data from LibriSpeech dataset {librispeech_split}...")
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(extracted_pitch_data_path), exist_ok=True)
        # Extract data from LibriSpeech dataset
        extracted_data = extract_data()
        # Save the extracted data to a file
        torch.save(extracted_data, extracted_pitch_data_path)
        print(f"Extracted pitch data saved to {extracted_pitch_data_path}.")

    # Process the raw input data
    hidden_states_arr, pitch_arr, pitch_binned = process_raw_input()
    print(f"Hidden states shape: {hidden_states_arr.shape}")
    print(f"Pitch shape: {pitch_arr.shape}")

    # Assemble a probe with ridge regression
    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states_arr,
        pitch_arr,
        test_size=0.2,
        random_state=42,
        stratify=pitch_binned,
    )
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    GridSearchRegressor = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [10**i for i in range(-5, 5)]},
        # scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    GridSearchRegressor.fit(X_train, y_train)
    y_pred = GridSearchRegressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    score = GridSearchRegressor.score(X_test, y_test)
    print(f"R^2 Score: {score}")


if __name__ == "__main__":
    main()
