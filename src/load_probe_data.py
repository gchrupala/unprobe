import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

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

else:
    # If running on local machine, use the local dataset root
    DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENT_ROOT = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENT_ROOT = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")


def format_data(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    seq_sampling: str = "random_frames",
    normalize_features: bool = True,
):
    """
    Format the data for probing tasks.
    Args:
        librispeech_split: The LibriSpeech split to use. Options are "dev-clean", "train-clean-100"
        modelname: The name of the transformer model used to extract hidden states.
        seq_sampling: The sequence sampling method used. Options are "random_frames", "mean", "none"
    Returns:
        processed_X: The processed input features.
        processed_Y: The processed target hidden states.
        data_shape: A dictionary containing the shape of each feature component.

    """

    acoustic_feature_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_opensmile_features_lld.pickle"
    )
    special_features_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_special_features.pickle"
    )
    transcription_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_transcriptions.pickle"
    )
    dnn_hidden_states_path = f"{SAVEPATH}/librispeech-{librispeech_split}_{modelname.split('/')[-1]}_representation_{seq_sampling}.pickle"

    # Make sure all the required files exist
    for path in [
        acoustic_feature_path,
        special_features_path,
        transcription_path,
        dnn_hidden_states_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load the acoustic features low level descriptors (LLD)
    with open(
        acoustic_feature_path,
        "rb",
    ) as f:
        lld = pickle.load(f)

    # Change the file index column in lld to be the fileID only
    lld.index = lld.index.set_levels(
        lld.index.levels[0].str.split("/").str[-1].str.replace(".flac", ""),
        level=0,
    )

    # Load the special features including word embedding, phonetic posterior grams (PPG), and speaker embeddings
    with open(
        special_features_path,
        "rb",
    ) as f:
        special_features = pickle.load(f)

    # Load the transcriptions with syntax features
    with open(transcription_path, "rb") as f:
        transcription_raw = pickle.load(f)

    # Sort the list of dictionary by the fileID key
    transcription = sorted(
        transcription_raw,
        key=lambda x: x["fileID"],
    )
    logger.info("Loaded and sorted transcription data")

    # Sort the lld by fileID
    lld = lld.sort_values(["file", "start"])

    # Load the transformer model hidden states
    with open(
        dnn_hidden_states_path,
        "rb",
    ) as f:
        dnn_hidden_states = pickle.load(f)

    # Remove entries where the length of words and syntax_feats are not equal
    transcription = [
        x for x in transcription if len(x["words"]) == len(x["syntax_feats"])
    ]
    transcription = [
        x for x in transcription if x["fileID"] in dnn_hidden_states.keys()
    ]
    transcription = pd.DataFrame(transcription)
    valid_fileIDs = transcription["fileID"].unique().tolist()
    # Set the fileID column as the index for transcription
    transcription = transcription.set_index("fileID")

    logger.info("Loaded all data files")
    logger.info(f"Number of valid fileIDs: {len(valid_fileIDs)}")

    processed_X, processed_Y = [], []
    filename_timestamp = []

    for fileID in tqdm(valid_fileIDs):
        spk_embedding = np.array(special_features[fileID]["spk_emb"])
        ppg_features = np.array(special_features[fileID]["ppgs"])
        # Swap the dimensions in ppg_features to be (time, ppg_dim)
        ppg_features = ppg_features.transpose((1, 0))
        fasttext_embedding = np.array(special_features[fileID]["fasttext"])

        metadata = transcription.loc[fileID]["non_acoustic"]
        syntax_feats = transcription.loc[fileID]["syntax_feats"]
        # sent = transcription.loc[fileID]["words"]

        utt_lld = lld.loc[fileID].reset_index()
        # Convert opensmile frame to ms in integer
        utt_lld["start_ms"] = (utt_lld["start"] / np.timedelta64(1, "ns") / 1e6).astype(
            int
        )

        ort_alignment = (
            transcription.loc[fileID]["ort_alignment"].dropna().reset_index()
        )
        # phone_alignment = transcription.loc[fileID]["phone_alignment"]

        # Add new columns char_idx_start and char_idx_end to ort_alignment
        # by counting how many characters since the start of the sentence
        cumulative_char_count = 0
        char_idx_starts = []
        char_idx_ends = []
        for word in ort_alignment["text"]:
            char_idx_starts.append(cumulative_char_count)
            cumulative_char_count += len(word)
            char_idx_ends.append(cumulative_char_count)
            # Add 1 to cumulative_char_count to account for the space
            cumulative_char_count += 1
        ort_alignment["char_idx_start"] = char_idx_starts
        ort_alignment["char_idx_end"] = char_idx_ends

        utt_dnn_hidden_states = dnn_hidden_states[fileID]
        # Move the time dimension to the first dimension
        utt_dnn_hidden_states["hidden_states"] = np.moveaxis(
            utt_dnn_hidden_states["hidden_states"], 1, 0
        )

        if "offset_mapping" in utt_dnn_hidden_states["frame_token_indices"].keys():
            frame_indices = utt_dnn_hidden_states["frame_token_indices"][
                "frame_indices"
            ]
            offset_mappings = utt_dnn_hidden_states["frame_token_indices"][
                "offset_mapping"
            ]
            raw_frame_indices = None
        else:
            frame_indices = utt_dnn_hidden_states["frame_token_indices"][
                "frame_indices_in_ms"
            ]
            raw_frame_indices = utt_dnn_hidden_states["frame_token_indices"][
                "frame_indices"
            ]
            # Make a list filled with None for offset_mapping
            offset_mappings = None
        hidden_states = utt_dnn_hidden_states["hidden_states"]
        list_of_hidden_states_and_frame_indices = list(
            zip(hidden_states, frame_indices)
        )

        for i, (
            utt_dnn_hidden_state,
            frame_index,
        ) in enumerate(list_of_hidden_states_and_frame_indices):
            # Use if statement to separate text model from audio model
            if offset_mappings is not None:
                # For text models, the frame_index corresponds to the subword token index
                # So we need to look up the actual word index from offset_mapping
                if frame_index == 0:
                    # If the frame_index is 0, it means it's the [CLS] token
                    # We can skip this frame
                    logger.info(
                        f"Skipping [CLS] token for fileID {fileID} at frame_index {frame_index}"
                    )
                    continue
                # Use offset_mappings to get the word index
                start_char_idx = offset_mappings[frame_index][
                    0
                ]  # start char index of the token
                # Find the word index in ort_alignment that contains this char index
                word_idx = ort_alignment[
                    (ort_alignment["char_idx_start"] <= start_char_idx)
                    & (ort_alignment["char_idx_end"] > start_char_idx)
                ].index

                if word_idx.empty:
                    logger.info(
                        f"Empty word_idx for fileID {fileID} at frame_index {frame_index}"
                    )
                    continue

                token_start_time = (
                    ort_alignment.loc[word_idx, "start"] * 1000
                )  # Convert to ms
                token_end_time = (
                    ort_alignment.loc[word_idx, "end"] * 1000
                )  # Convert to ms
                # Get the middle time of the token in ms as integer and round to the nearest 10 ms
                token_time = (
                    int((token_start_time.iloc[0] + token_end_time.iloc[0]) / 2 / 10)
                    * 10
                )

                utt_lld_frames = (
                    utt_lld[
                        (utt_lld["start_ms"] >= token_time - 20)
                        & (utt_lld["start_ms"] <= token_time + 20)
                    ]
                    .drop(columns=["start", "end", "start_ms"])
                    .reset_index(drop=True)
                    .to_numpy()
                )

            else:
                token_time = frame_index
                # Get the corresponding word and syntax features for current frame index
                # To get the word embedding we need to find the index of the word in ort_alignment
                word_idx = ort_alignment[
                    (ort_alignment["start"] <= token_time / 1000)
                    & (ort_alignment["end"] > token_time / 1000)
                ].index
                if word_idx.empty:
                    # skip if no word found
                    continue

                # Get the corresponding lld rows for current frame index +- 2 frames
                utt_lld_frames = (
                    utt_lld[
                        (utt_lld["start_ms"] >= token_time - 20)
                        & (utt_lld["start_ms"] <= token_time + 20)
                    ]
                    .drop(columns=["start", "end", "start_ms"])
                    .reset_index(drop=True)
                    .to_numpy()
                )
            if utt_lld_frames.shape[0] != 5:
                continue
            utt_lld_frames = utt_lld_frames.flatten()
            utt_lld_names = utt_lld.columns.tolist()

            word_embedding = fasttext_embedding[word_idx].flatten()
            syntax_feature = np.array(syntax_feats)[word_idx].flatten()

            # Get the corresponding PPG features for current frame index
            ppg_feature = ppg_features[int(token_time // 10)].flatten()

            # Make sure all the dimensions are correct
            assert all(
                (
                    word_embedding.shape[0] == 100,
                    syntax_feature.shape[0] == 34,
                    ppg_feature.shape[0] == 40,
                    spk_embedding.shape[0] == 100,
                    metadata.shape[0] == 2,
                )
            )

            # Concatenate all the features to form the input feature vector
            input_feature = np.concatenate(
                [
                    utt_lld_frames,
                    word_embedding,
                    syntax_feature,
                    ppg_feature,
                    spk_embedding,
                    metadata,
                ],
                axis=0,
            )

            processed_X.append(input_feature)
            processed_Y.append(utt_dnn_hidden_state)
            filename_timestamp.append(
                (fileID, frame_index)
            ) if raw_frame_indices is None else filename_timestamp.append(
                (fileID, frame_index, raw_frame_indices[i])
            )

            data_shape = {
                "acoustic_features": utt_lld_frames.shape,
                "word_embedding": word_embedding.shape,
                "syntax_feature": syntax_feature.shape,
                "ppg_feature": ppg_feature.shape,
                "spk_embedding": spk_embedding.shape,
                "metadata": metadata.shape,
                "input_feature_all": input_feature.shape,
                "dnn_hidden_state": utt_dnn_hidden_state.shape,
            }

    processed_X = np.array(processed_X)
    if normalize_features:
        from sklearn.preprocessing import StandardScaler

        # We normalize features in a column-wise manner so that each feature has zero mean and unit variance
        scaler = StandardScaler()
        processed_X = scaler.fit_transform(processed_X)

        # start_idx = 0
        # end_idx = 0
        # # Normalize the features within each group
        # for name, shape in data_shape.items():  # type: ignore
        #     if name == "input_feature_all" or name == "dnn_hidden_state":
        #         continue
        #     start_idx = end_idx
        #     end_idx += shape[0]
        #     scaler = StandardScaler()
        #     processed_X[:, start_idx:end_idx] = scaler.fit_transform(
        #         processed_X[:, start_idx:end_idx]
        #     )
        logger.info("Normalized input features")
    processed_Y = np.array(processed_Y)

    # Processed_Y shape should be (num_frames, num_layers, hidden_size)
    logger.info(f"Processed X shape: {processed_X.shape}")
    logger.info(f"Processed Y shape: {processed_Y.shape}")
    return processed_X, processed_Y, filename_timestamp, data_shape  # type: ignore


def load_data(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    seq_sampling: str = "random_frames",
    select_layers: list | None = None,
    overwrite: bool = False,
    normalize_features: bool = True,
):
    """
    Load the formatted data for probing tasks.
    Args:
        librispeech_split: The LibriSpeech split to use. Options are "dev-clean", "train-clean-100"
        modelname: The name of the transformer model used to extract hidden states.
        seq_sampling: The sequence sampling method used. Options are "random_frames", "mean", "none"
        select_layers: List of layer indices to select from the transformer model. If None, use all layers.
    Returns:
        processed_X: The processed input features.
        processed_Y: The processed target hidden states.
        data_shape: A dictionary containing the shape of each feature component.

    """
    # Check if the data has already been formatted and saved

    formatted_data_path = f"{SAVEPATH}/processed_data/librispeech-{librispeech_split}_{modelname.split('/')[-1]}_representation_{seq_sampling}_formatted.pickle"
    if normalize_features:
        formatted_data_path = formatted_data_path.replace(
            ".pickle", "_normalized.pickle"
        )

    # Make sure the directory exists
    os.makedirs(os.path.dirname(formatted_data_path), exist_ok=True)
    if os.path.exists(formatted_data_path) and not overwrite:
        logger.info(f"Loading formatted data from {formatted_data_path}")
        with open(formatted_data_path, "rb") as f:
            return pickle.load(f)
    else:
        # If not, format the data and save it
        logger.info(
            "Formatted data not found or overwrite flag is set, formatting data..."
        )
        processed_X, processed_Y, filename_timestamp, data_shape = format_data(
            librispeech_split=librispeech_split,
            modelname=modelname,
            seq_sampling=seq_sampling,
            normalize_features=normalize_features,
        )
        with open(formatted_data_path, "wb") as f:
            pickle.dump((processed_X, processed_Y, filename_timestamp, data_shape), f)
        logger.info(f"Saved formatted data to {formatted_data_path}")

    if select_layers is not None:
        logger.info(f"Selecting layers: {select_layers}")
        processed_Y = processed_Y[:, select_layers, :]

        logger.info(
            f"Selected processed_Y shape after layer selection: {processed_Y.shape}"
        )

    return processed_X, processed_Y, filename_timestamp, data_shape


def load_sample_data():
    bert_pickle = "../experimental_data/librispeech-dev-clean_bert-base-uncased_representation_random_frames.pickle"

    with open(bert_pickle, "rb") as f:
        transformer_representation = pickle.load(f)

    hidden_states = [
        transformer_representation[x]["hidden_states"]
        for x in list(transformer_representation.keys())
    ]


def dimension_reduction(hidden_states: np.ndarray, n_components: int = 100):
    """Reduce the dimension of the hidden_states

    Args:
        hidden_states (np.ndarray): _hidden_states is a list of numpy arrays with shape (nunm_frames, num_layers, hidden_size)
        n_components (int, optional): _n_components is the number of components to keep. Defaults to 100.
    """
    # hidden_states is a list of numpy arrays with shape (num_frames, num_layers, hidden_size)

    logger.info(f"Stacked hidden_states shape: {hidden_states.shape}")

    # Use dimension reduction technique such as PCA or t-SNE to reduce the dimension of hidden states
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    # Since the hidden_states is 3D array (num_samples, num_layers, hidden_size), we need to do PCA on the last dimension. We need to do the PCA on each layer separately and then stack them back together.
    reduced_hidden_states = []
    for layer in range(hidden_states.shape[1]):
        layer_hidden_states = hidden_states[:, layer, :]
        reduced_layer_hidden_states = pca.fit_transform(layer_hidden_states)
        reduced_hidden_states.append(reduced_layer_hidden_states)

    # Stack the reduced hidden states back to a 3D array with shape (num_samples, num_layers, n_components)
    reduced_hidden_states = np.array(reduced_hidden_states)
    reduced_hidden_states = np.moveaxis(reduced_hidden_states, 0, 1)
    logger.info(f"Reduced hidden_states shape: {reduced_hidden_states.shape}")

    # Make sure the first two dimensions are the same as the original hidden_states
    assert reduced_hidden_states.shape[0] == hidden_states.shape[0]
    assert reduced_hidden_states.shape[1] == hidden_states.shape[1]
    return reduced_hidden_states


if __name__ == "__main__":
    librispeech_split = "dev-clean"
    modelname = "facebook/hubert-base-ls960"
    seq_sampling = "random_frames"
    select_layers = [0, 6, 12]
    processed_X, processed_Y, _, data_shape = format_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling=seq_sampling,
    )  # For testing purposes
    reduced_Y = dimension_reduction(
        processed_Y, n_components=100
    )  # Reduce to 100 dimensions
    print(data_shape)
    print(f"Processed X shape: {processed_X.shape}")
    print(f"Processed Y shape: {processed_Y.shape}")
    print(f"Reduced Y shape: {reduced_Y.shape}")
