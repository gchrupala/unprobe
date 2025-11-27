import json
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

ACOUSTIC_FEATURE_NAMES = []

INPUT_FEATURE_SELECT_COMPONENTS = [
    "OtherAcoustic",
    "SpectralInfo",
    "Formants",
    # "word_embedding",
    "syntax_feature",
    "ppg_feature",
    "spk_embedding",
    # "metadata",
    # "word_form_feature",
    "dnn_word_embedding",
]


def get_opensmile_feature_names():
    import opensmile

    feature_level = opensmile.FeatureLevel.LowLevelDescriptors
    smile = opensmile.Smile(
        feature_set="eGeMAPSv02",
        feature_level=feature_level,
        verbose=True,
        num_workers=8,
        sampling_rate=16000,
        resample=True,
    )

    feature_names = smile.feature_names

    feature_groups = {
        "OtherAcoustic": [
            "Loudness_sma3",
            "F0semitoneFrom27.5Hz_sma3nz",
            "jitterLocal_sma3nz",
            "shimmerLocaldB_sma3nz",
            "HNRdBACF_sma3nz",
            "logRelF0-H1-H2_sma3nz",
            "logRelF0-H1-A3_sma3nz",
        ],
        "SpectralInfo": [
            "alphaRatio_sma3",
            "hammarbergIndex_sma3",
            "slope0-500_sma3",
            "slope500-1500_sma3",
            "spectralFlux_sma3",
            "mfcc1_sma3",
            "mfcc2_sma3",
            "mfcc3_sma3",
            "mfcc4_sma3",
        ],
        "Formants": [
            "F1frequency_sma3nz",
            "F1bandwidth_sma3nz",
            "F1amplitudeLogRelF0_sma3nz",
            "F2frequency_sma3nz",
            "F2bandwidth_sma3nz",
            "F2amplitudeLogRelF0_sma3nz",
            "F3frequency_sma3nz",
            "F3bandwidth_sma3nz",
            "F3amplitudeLogRelF0_sma3nz",
        ],
    }

    # Grab the corresponding index for each feature within each group
    grouped_feature_indices = {}
    for group_name, features in feature_groups.items():
        indices = [
            feature_names.index(feat) for feat in features if feat in feature_names
        ]
        grouped_feature_indices[group_name] = {
            "feature_names": features,
            "indices": indices,
        }

    return grouped_feature_indices


opensmile_feature_names_json = f"{SAVEPATH}/opensmile_feature_names.json"
if os.path.exists(opensmile_feature_names_json):
    with open(opensmile_feature_names_json, "r") as f:
        ACOUSTIC_FEATURE_NAMES = json.load(f)
    logger.info("Loaded opensmile feature names from JSON file")
else:
    ACOUSTIC_FEATURE_NAMES = get_opensmile_feature_names()
    with open(opensmile_feature_names_json, "w") as f:
        json.dump(ACOUSTIC_FEATURE_NAMES, f, indent=4)
    logger.info("Saved opensmile feature names to JSON file")


def format_data(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    seq_sampling: str = "random_frames",
):
    """
    Format the data for probing tasks.
    Args:
        librispeech_split: The LibriSpeech split to use. Options are "dev-clean", "train-clean-100"
        modelname: The name of the transformer model used to extract hidden states.
        seq_sampling: The sequence sampling method used. Options are "random_frames", "mean", "none"
    Returns:
        feature_sets: The processed input features.
        model_hidden_states: The processed target hidden states.
        filename_timestamp: List of tuples containing (fileID, frame_index) for each data point.

    """

    acoustic_feature_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_opensmile_features_lld.pickle"
    )
    fasttext_word_embedding_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_fasttext_word_embeddings.pickle"
    )
    ppg_feature_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_phonetic_posteriorgram.pickle"
    )
    speaker_embedding_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_speaker_embedding.pickle"
    )
    transcription_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_transcriptions.pickle"
    )
    word_form_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_letter_unigram_embeddings.pickle"
    )
    syntax_feature_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_syntax_features.pickle"
    )
    dnn_word_embeddings_path = (
        f"{SAVEPATH}/librispeech-{librispeech_split}_dnn_word_embeddings.pickle"
    )

    dnn_hidden_states_path = f"{SAVEPATH}/librispeech-{librispeech_split}_{modelname.split('/')[-1]}_representation_{seq_sampling}.pickle"

    # Make sure all the required files exist
    for path in [
        acoustic_feature_path,
        fasttext_word_embedding_path,
        ppg_feature_path,
        speaker_embedding_path,
        transcription_path,
        dnn_hidden_states_path,
        word_form_path,
        syntax_feature_path,
        dnn_word_embeddings_path,
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

    # Load the speaker embeddings
    with open(speaker_embedding_path, "rb") as f:
        all_speaker_embeddings = pickle.load(f)
    # Load the PPG features
    with open(ppg_feature_path, "rb") as f:
        all_ppg_features = pickle.load(f)
    # Load the fasttext word embeddings
    with open(fasttext_word_embedding_path, "rb") as f:
        all_fasttext_word_embeddings = pickle.load(f)

    # Load the transcriptions with syntax features
    with open(transcription_path, "rb") as f:
        transcription_raw = pickle.load(f)

    # Load the syntax features
    with open(syntax_feature_path, "rb") as f:
        all_syntax_features = pickle.load(f)

    # Load the word form
    with open(word_form_path, "rb") as f:
        all_word_form_embeddings = pickle.load(f)

    # Load DNN word embeddings
    with open(dnn_word_embeddings_path, "rb") as f:
        all_dnn_word_embeddings = pickle.load(f)

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

    # # Remove entries where the length of words and syntax_feats are not equal
    # transcription = [
    #     x
    #     for i, x in enumerate(transcription)
    #     if len(x["words"]) == len(all_syntax_features[transcription[i]["fileID"]]['features'])
    # ]
    transcription = [
        x for x in transcription if x["fileID"] in dnn_hidden_states.keys()
    ]
    transcription = pd.DataFrame(transcription)
    valid_fileIDs = transcription["fileID"].unique().tolist()
    # Set the fileID column as the index for transcription
    transcription = transcription.set_index("fileID")

    logger.info("Loaded all data files")
    logger.info(f"Number of valid fileIDs: {len(valid_fileIDs)}")

    model_hidden_states = []
    filename_timestamp = []
    all_input_features = []

    for fileID in tqdm(valid_fileIDs):
        spk_embedding = np.array(all_speaker_embeddings[fileID])
        ppg_features = np.array(all_ppg_features[fileID])
        # Swap the dimensions in ppg_features to be (time, ppg_dim)
        ppg_features = ppg_features.transpose((1, 0))
        fasttext_embedding = np.array(all_fasttext_word_embeddings[fileID])
        word_form_embeddings = all_word_form_embeddings[fileID]
        dnn_word_embeddings = all_dnn_word_embeddings[fileID]

        metadata = transcription.loc[fileID]["non_acoustic"]
        # syntax_feats = transcription.loc[fileID]["syntax_feats"]
        syntax_feats = all_syntax_features[fileID]["features"]
        syntax_feats_offset_mapping = all_syntax_features[fileID]["offset_mapping"]
        # sent = transcription.loc[fileID]["words"]

        utt_lld = lld.loc[fileID].reset_index()
        # Convert opensmile frame to ms in integer
        utt_lld["start_ms"] = (utt_lld["start"] / np.timedelta64(1, "ns") / 1e6).astype(
            int
        )

        ort_alignment = (
            transcription.loc[fileID]["ort_alignment"].dropna().reset_index()
        )
        ort_alignment = ort_alignment[ort_alignment["text"] != "<pad>"].reset_index(
            drop=True
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
            word_idx = word_idx.values[0]
            if utt_lld_frames.shape[0] != 5:
                continue
            # utt_lld_names = utt_lld.columns.tolist()

            word_embedding = fasttext_embedding[word_idx].flatten()

            word_form_feature = word_form_embeddings[word_idx].flatten()

            # Find the corresponding DNN word embedding
            char_start_idx = ort_alignment.loc[word_idx, "char_idx_start"]
            char_end_idx = ort_alignment.loc[word_idx, "char_idx_end"]
            # Find at which interval in the numpy array dnn_word_embeddings['offset_mapping'] the char_start_idx falls into
            dnn_offset_mappings = dnn_word_embeddings["offset_mapping"]
            dnn_word_idx = np.where(
                (dnn_offset_mappings[:, 0] <= char_start_idx)
                & (dnn_offset_mappings[:, 1] > char_start_idx)
            )[0]
            if dnn_word_idx.size == 0:
                logger.info(
                    "Char start index not found in DNN embeddings: Skipped",
                )
                continue
            dnn_word_idx = dnn_word_idx[0]
            dnn_word_embedding = dnn_word_embeddings["word_embeddings"][
                dnn_word_idx
            ].flatten()
            dnn_token = dnn_word_embeddings["words"][dnn_word_idx]

            # Use offset mapping to look up syntax features as well
            syntax_word_idx = np.where(
                (np.array(syntax_feats_offset_mapping)[:, 0] <= char_start_idx)
                & (np.array(syntax_feats_offset_mapping)[:, 1] >= char_end_idx)
            )
            if syntax_word_idx[0].size == 0:
                logger.info(
                    "Char start index not found in syntax features: Skipped",
                )
                continue
            syntax_feature = np.array(syntax_feats)[syntax_word_idx].flatten()
            syntax_token = all_syntax_features[fileID]["words"][syntax_word_idx[0][0]]

            # Get the corresponding PPG features for current frame index
            ppg_feature = ppg_features[int(token_time // 10)].flatten()

            acoustic_features = {}
            for acoustic_group in list(ACOUSTIC_FEATURE_NAMES.keys()):
                if acoustic_group in INPUT_FEATURE_SELECT_COMPONENTS:
                    acoustic_feature_indices = ACOUSTIC_FEATURE_NAMES[acoustic_group][
                        "indices"
                    ]
                    acoustic_features[acoustic_group] = utt_lld_frames[
                        :, acoustic_feature_indices
                    ].flatten()

            # Store input feature components in a dictionary
            input_feature_components = acoustic_features | {
                "word_embedding": word_embedding,
                "syntax_feature": syntax_feature,
                "ppg_feature": ppg_feature,
                "spk_embedding": spk_embedding,
                "metadata": metadata,
                "word_form_feature": word_form_feature,
                "dnn_word_embedding": dnn_word_embedding,
                "dnn_token": dnn_token,
                "syntax_token": syntax_token,
            }

            all_input_features.append(input_feature_components)
            model_hidden_states.append(utt_dnn_hidden_state)
            filename_timestamp.append(
                (fileID, frame_index)
            ) if raw_frame_indices is None else filename_timestamp.append(
                (fileID, frame_index, raw_frame_indices[i])
            )

    # # Check that for each word, the dnn_token is a substring of syntax_token
    # word_pairs = [(x['dnn_token'], x['syntax_token']) for x in all_input_features]
    # for word_pair in word_pairs:
    #     assert word_pair[0] in word_pair[1].lower()

    # We restructure all_input_features to be a dictionary of numpy arrays for each feature component
    logger.info("Processing input features...")
    all_input_features_dict = {}
    for feature_name in INPUT_FEATURE_SELECT_COMPONENTS:
        all_input_features_dict[feature_name] = [
            input_feature_components[feature_name]
            for input_feature_components in all_input_features
        ]

    all_input_features_dict["dnn_word_embedding"] = np.array(
        all_input_features_dict["dnn_word_embedding"]
    )
    logger.info(
        f"Processed DNN word embeddings shape before PCA: {all_input_features_dict['dnn_word_embedding'].shape}"
    )

    return all_input_features_dict, model_hidden_states, filename_timestamp


def further_process(
    model_hidden_states,
    all_input_features_dict: dict,
    reduce_dnn_word_embedding: bool = True,
    reduce_speaker_embedding: bool = True,
    one_hot_encode_syntax: bool = True,
    normalize_features: bool = True,
    n_components: int | float | None = 0.9,
):
    if reduce_dnn_word_embedding:
        # Apply dimension reduction to processed_dnn_word_embeddings using PCA so that 90% variance is retained by default
        # Change n_components to an integer or float to specify the number of components or variance ratio to retain
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        all_input_features_dict["dnn_word_embedding"] = scaler.fit_transform(
            all_input_features_dict["dnn_word_embedding"]
        )

        pca = PCA(
            n_components=n_components
        )  # Use pca to retain 90% variance by default
        all_input_features_dict["dnn_word_embedding"] = pca.fit_transform(
            all_input_features_dict["dnn_word_embedding"]
        )

        logger.info(
            f"Processed DNN word embeddings shape after PCA: {all_input_features_dict['dnn_word_embedding'].shape}"
        )
    if reduce_speaker_embedding:
        # Apply dimension reduction to speaker_embedding using PCA so that 90% variance is retained by default
        # Change n_components to an integer or float to specify the number of components or variance ratio to retain
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        all_input_features_dict["spk_embedding"] = scaler.fit_transform(
            all_input_features_dict["spk_embedding"]
        )

        pca = PCA(
            n_components=n_components
        )  # Use pca to retain 90% variance by default
        all_input_features_dict["spk_embedding"] = pca.fit_transform(
            all_input_features_dict["spk_embedding"]
        )

        logger.info(
            f"Processed speaker embeddings shape after PCA: {all_input_features_dict['spk_embedding'].shape}"
        )

    if one_hot_encode_syntax:
        # One hot encode the individual columns within syntax_feature
        from sklearn.preprocessing import OneHotEncoder

        # mask out the normed features for one-hot encoding
        syntax_feature_array = np.array(all_input_features_dict["syntax_feature"])
        syntax_feature_mask = np.ones_like(syntax_feature_array, dtype=bool)
        # Normed features are -1 and -3rd columns in each group of 7 features
        syntax_feature_mask[:, -1] = False
        syntax_feature_mask[:, -3] = False
        syntax_feature_array = syntax_feature_array[:, syntax_feature_mask[0]]
        encoder = OneHotEncoder(sparse_output=False)
        onehot_encoded_columns = []
        for original_syntax_feat in syntax_feature_array.T:
            original_syntax_feat = original_syntax_feat.reshape(-1, 1)
            onehot_encoded_col = encoder.fit_transform(original_syntax_feat)
            onehot_encoded_columns.append(onehot_encoded_col)
        syntax_feature_onehot = np.concatenate(onehot_encoded_columns, axis=1)
        all_input_features_dict["syntax_feature"] = syntax_feature_onehot

    logger.info("Concatenating input features...")
    logger.info(f"Input features include: {INPUT_FEATURE_SELECT_COMPONENTS}")
    # Now concatenate all input features to form the final input feature matrix
    feature_sets = np.array(
        np.concatenate(
            [
                all_input_features_dict[feature_name]
                for feature_name in INPUT_FEATURE_SELECT_COMPONENTS
            ],
            axis=1,
        )
    )

    data_shape = {
        feature_name: all_input_features_dict[feature_name][0].shape
        for feature_name in INPUT_FEATURE_SELECT_COMPONENTS
    }

    # model_hidden_states shape should be (num_frames, num_layers, hidden_size)
    logger.info(f"Processed X shape: {feature_sets.shape}")
    model_hidden_states = np.array(
        model_hidden_states
    )  # Convert list to numpy array # type: ignore
    logger.info(f"Processed Y shape: {model_hidden_states.shape}")

    if normalize_features:
        normalize_groups = [
            "OtherAcoustic",
            "SpectralInfo",
            "Formants",
        ]
        from sklearn.preprocessing import StandardScaler

        # We normalize features in a column-wise manner so that each feature has zero mean and unit variance
        logger.info("Normalizing input features")
        scaler = StandardScaler()
        # We only want to normalize the features inside normalize_groups
        feature_indices_to_normalize = []
        start_idx = 0
        for feature_name in INPUT_FEATURE_SELECT_COMPONENTS:
            feature_dim = data_shape[feature_name][0]
            end_idx = start_idx + feature_dim
            if feature_name in normalize_groups:
                feature_indices_to_normalize.extend(list(range(start_idx, end_idx)))
            start_idx = end_idx
        feature_sets_to_normalize = feature_sets[:, feature_indices_to_normalize]
        feature_sets_normalized = scaler.fit_transform(feature_sets_to_normalize)
        # Replace the normalized features back to feature_sets
        feature_sets[:, feature_indices_to_normalize] = feature_sets_normalized
        logger.info("Normalized input features")

    return feature_sets, model_hidden_states, data_shape


def load_data(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    seq_sampling: str = "random_frames",
    select_layers: list | None = None,
    overwrite: bool = False,
    normalize_features: bool = True,
    one_hot_encode_syntax: bool = True,
    reduce_dnn_word_embedding: bool = True,
):
    """
    Load the formatted data for probing tasks.
    Args:
        librispeech_split: The LibriSpeech split to use. Options are "dev-clean", "train-clean-100"
        modelname: The name of the transformer model used to extract hidden states.
        seq_sampling: The sequence sampling method used. Options are "random_frames", "mean", "none"
        select_layers: List of layer indices to select from the transformer model. If None, use all layers.
    Returns:
        feature_sets: The processed input features.
        model_hidden_states: The processed target hidden states.
        data_shape: A dictionary containing the shape of each feature component.

    """
    # Check if the data has already been formatted and saved

    formatted_data_path = f"{SAVEPATH}/processed_data/librispeech-{librispeech_split}_{modelname.split('/')[-1]}_representation_{seq_sampling}_formatted.pickle"

    # Make sure the directory exists
    os.makedirs(os.path.dirname(formatted_data_path), exist_ok=True)
    if os.path.exists(formatted_data_path) and not overwrite:
        logger.info(f"Loading formatted data from {formatted_data_path}")
        with open(formatted_data_path, "rb") as f:
            feature_sets, model_hidden_states, filename_timestamp = pickle.load(f)

        logger.info("Loaded formatted data successfully")
    else:
        # If not, format the data and save it
        logger.info(
            "Formatted data not found or overwrite flag is set, formatting data..."
        )
        feature_sets, model_hidden_states, filename_timestamp = format_data(
            librispeech_split=librispeech_split,
            modelname=modelname,
            seq_sampling=seq_sampling,
        )
        with open(formatted_data_path, "wb") as f:
            pickle.dump((feature_sets, model_hidden_states, filename_timestamp), f)
        logger.info(f"Saved formatted data to {formatted_data_path}")

    if select_layers is not None:
        logger.info(f"Selecting layers: {select_layers}")
        model_hidden_states = model_hidden_states[:, select_layers, :]

        logger.info(
            f"Selected model_hidden_states shape after layer selection: {model_hidden_states.shape}"
        )

    feature_sets, model_hidden_states, data_shape = further_process(
        model_hidden_states,
        feature_sets,
        reduce_dnn_word_embedding=reduce_dnn_word_embedding,
        one_hot_encode_syntax=one_hot_encode_syntax,
        normalize_features=normalize_features,
    )

    return feature_sets, model_hidden_states, filename_timestamp, data_shape


def sanity_check_pca():
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import (
        GridSearchCV,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, Y, _, _ = load_data(
        librispeech_split="dev-clean",
        modelname="facebook/wav2vec2-base",
        seq_sampling="random_frames",
        # select_layers=[12],
        overwrite=False,
        normalize_features=True,
    )

    results = {}

    for layer in range(Y.shape[1]):
        logger.info(f"Layer {layer} hidden state shape: {Y[:, layer, :].shape}")
        y_layer = Y[:, layer, :]
        # This pipeline will scale the Y data, then apply PCA
        y_transformer = Pipeline(steps=[("scaler", StandardScaler()), ("pca", PCA())])
        # The regressor will be a simple Ridge model
        ridge = Ridge()

        # The full model applies the transformer to Y before fitting Ridge
        # and inverse_transforms the predictions.
        model = TransformedTargetRegressor(regressor=ridge, transformer=y_transformer)
        param_grid = {
            "regressor__alpha": np.logspace(-2, 2, 5),  # e.g., [0.01, 0.1, 1, 10, 100]
            "transformer__pca__n_components": [0.90, 0.95, 0.99, None],
            # + list(range(50, 301, 50)),
            # A good search space:
            # - Floats: Capture a certain % of variance. 'None' is the original (problematic) case.
            # - Integers: Test specific numbers of components.
        }

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y_layer, test_size=0.2, random_state=42
        )

        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        grid_search.fit(X_train, Y_train)

        print("\nBest parameters found from grid search:")
        print(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(Y_test, y_pred)
        print(f"\nR-squared score on the test set: {test_r2:.4f}")

        results[layer] = {
            "test_r2": test_r2,
            "best_train_score": grid_search.best_score_,
        } | grid_search.best_params_

    print(results)


def load_sample_data():
    bert_pickle = "../experimental_data/librispeech-dev-clean_bert-base-uncased_representation_random_frames.pickle"

    with open(bert_pickle, "rb") as f:
        transformer_representation = pickle.load(f)

    hidden_states = [
        transformer_representation[x]["hidden_states"]
        for x in list(transformer_representation.keys())
    ]


def dimension_reduction(hidden_states: np.ndarray, n_components: int | None = None):
    """Reduce the dimension of the hidden_states

    Args:
        hidden_states (np.ndarray): _hidden_states is a list of numpy arrays with shape (nunm_frames, num_layers, hidden_size)
        n_components (int, optional): _n_components is the number of components to keep. Defaults to 100.
    """
    # hidden_states is a list of numpy arrays with shape (num_frames, num_layers, hidden_size)

    logger.info(f"Stacked hidden_states shape: {hidden_states.shape}")
    if n_components is None:
        logger.info("n_components is None, PCA is untruncated")

    else:
        if n_components >= hidden_states.shape[2]:
            logger.warning(
                f"n_components {n_components} is greater than or equal to hidden size {hidden_states.shape[2]}. Skipping dimension reduction."
            )
            return hidden_states
        logger.info(f"Reducing hidden states to {n_components} dimensions")

    # Use PCA to reduce the dimension of hidden states
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, svd_solver="full")
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


def get_section_shapes(data_shape: dict) -> list:
    """
    Get the shape of a specific section from the data_shape dictionary.
    Args:
        data_shape: A dictionary containing the shape of each feature component.
    Returns:
        section_shapes: A numpy array containing the start and end indices as well as the name of each section.
    """
    section_names = list(data_shape.keys())
    # We don't need the input_feature_all and dnn_hidden_state keys to create the section shape

    section_names = [
        name
        for name in section_names
        if name not in ["input_feature_all", "dnn_hidden_state"]
    ]

    start_idx = 0
    end_idx = 0

    section_shapes = []

    for section_name in section_names:
        start_idx = end_idx
        end_idx += data_shape[section_name][0]
        section_shapes.append([start_idx, end_idx, section_name])

    return section_shapes


if __name__ == "__main__":
    librispeech_split = "dev-clean"
    modelname = "facebook/hubert-base-ls960"
    modelname = "facebook/wav2vec2-base"
    seq_sampling = "random_frames"
    overwrite = False
    select_layers = [0, 6, 12]
    all_input_features_dict, model_hidden_states, filename_timestamp = format_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling=seq_sampling,
    )  # For testing purposes
    reduced_Y = dimension_reduction(
        model_hidden_states, n_components=100
    )  # Reduce to 100 dimensions
    print(data_shape)
    print(f"Processed X shape: {feature_sets.shape}")
    print(f"Processed Y shape: {model_hidden_states.shape}")
    print(f"Reduced Y shape: {reduced_Y.shape}")
