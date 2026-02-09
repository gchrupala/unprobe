import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import SAVEPATH, get_opensmile_feature_names

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


INPUT_FEATURE_SELECT_COMPONENTS = [
    # "OtherAcoustic",
    # "SpectralInfo",
    # "Formants",
    # "word_embedding",
    "eGeMAPSv02",
    "syntax_feature",
    "ppg_feature",
    # "spk_embedding",
    "metadata",
    # "word_form_feature",
    "dnn_word_embedding",
]


opensmile_feature_names_json = f"{SAVEPATH}/opensmile_feature_names.json"
if os.path.exists(opensmile_feature_names_json):
    with open(opensmile_feature_names_json, "r") as f:
        ACOUSTIC_FEATURE_NAMES: dict = json.load(f)
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
    num_skip_cls_sep = 0

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
            # TODO: punctuations exists in the text but not in the alignment. this is a problem.
            # Also the Subword tokenization may split words into multiple tokens, need to handle that as well.
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
                if np.all(offset_mappings[frame_index] == np.array([0, 0])):
                    # If the offset_mapping is [0,0], it means it's the [CLS] or [SEP] token
                    # We can skip this frame
                    num_skip_cls_sep += 1

                    continue
                word_idx = utt_dnn_hidden_states["frame_token_indices"]["word_ids"][
                    frame_index
                ]

                if word_idx > ort_alignment.index.max():
                    # Skip if word_idx is out of bounds, this is probably due to punctuations
                    continue

                token_start_time = int(
                    ort_alignment.loc[word_idx, "start"] * 1000
                )  # Convert to ms
                token_end_time = int(
                    ort_alignment.loc[word_idx, "end"] * 1000
                )  # Convert to ms
                # Get the middle time of the token in ms as integer and round to the nearest 10 ms
                token_time = int((token_start_time + token_end_time) / 2 / 10) * 10

            else:
                # For audio models, the frame_index corresponds to the time in ms and there's no offset_mapping
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
                word_idx = word_idx.values[0]

            # Get the corresponding lld rows using the lower and upper bounds

            lowerbound = token_time  # - 20
            upperbound = token_time  # + 20
            # Get the number of 10ms frames between lowerbound and upperbound
            no_frames = (upperbound - lowerbound) // 10 + 1

            utt_lld_frames = (
                utt_lld[
                    (utt_lld["start_ms"] >= lowerbound)
                    & (utt_lld["start_ms"] <= upperbound)
                ]
                .drop(columns=["start", "end", "start_ms"])
                .reset_index(drop=True)
                .to_numpy()
            )
            # Look up the word start char index
            start_char_idx = ort_alignment.loc[word_idx, "char_idx_start"]

            if utt_lld_frames.shape[0] != no_frames:  # Check if no_frames match
                continue
            # utt_lld_names = utt_lld.columns.tolist()

            word_embedding = fasttext_embedding[word_idx].flatten()

            word_form_feature = word_form_embeddings[word_idx].flatten()

            # Find the corresponding DNN word embedding
            dnn_word_embedding = dnn_word_embeddings["word_embeddings"][word_idx]

            # Use offset mapping to look up syntax features as well
            syntax_word_idx = np.where(
                (np.array(syntax_feats_offset_mapping)[:, 0] <= start_char_idx)
                & (np.array(syntax_feats_offset_mapping)[:, 1] > start_char_idx)
            )
            if syntax_word_idx[0].size == 0:
                # logger.info(
                #     "Char start index not found in syntax features: Skipped",
                # )
                # This is mainly due to the sentencizer in spacy splitting the sentences up
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
    logger.info(f"Skipping [CLS] or [SEP] token count: {num_skip_cls_sep}")
    # We restructure all_input_features to be a dictionary of numpy arrays for each feature component
    logger.info("Processing input features...")
    # We turn the list of dictionaries into a dictionary of lists
    all_input_features_dict = {}
    for feature_name in all_input_features[0].keys():
        all_input_features_dict[feature_name] = [
            x[feature_name] for x in all_input_features
        ]
    all_input_features_dict["dnn_word_embedding"] = np.array(
        all_input_features_dict["dnn_word_embedding"]
    )
    syntax_feats_names = all_syntax_features[valid_fileIDs[-1]]["names"]
    all_input_features_dict["syntax_feature_names"] = syntax_feats_names
    logger.info(
        f"Processed DNN word embeddings shape before PCA: {all_input_features_dict['dnn_word_embedding'].shape}"
    )

    return all_input_features_dict, model_hidden_states, filename_timestamp


def further_process(
    model_hidden_states,
    all_input_features_dict: dict,
    selected_input_components: list,
    reduce_dnn_word_embedding: bool = True,
    reduce_speaker_embedding: bool = True,
    one_hot_encode_syntax: bool = True,
    one_hot_encode_syntax_separate: bool = False,
    one_hot_encode_metadata: bool = True,
    add_additional_syntax_features: bool = False,
    argmax_ppg: bool = False,
    normalize_features: bool = True,
    n_components: int | float | None = 0.95,
    select_layers: list | None = None,
):
    # Use selected_input_components to select which features in the dictionary we want to keep
    for feature_name in selected_input_components:
        if feature_name not in all_input_features_dict.keys():
            raise ValueError(f"Feature {feature_name} not found in input features.")
        all_input_features_dict[feature_name] = np.array(
            all_input_features_dict[feature_name]
        )

    if (
        reduce_dnn_word_embedding
        and "dnn_word_embedding" in all_input_features_dict.keys()
    ):
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
    if reduce_speaker_embedding and "spk_embedding" in all_input_features_dict.keys():
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

    if one_hot_encode_syntax and "syntax_feature" in all_input_features_dict.keys():
        # One hot encode the individual columns within syntax_feature
        from sklearn.preprocessing import OneHotEncoder

        # mask out the normed features for one-hot encoding
        syntax_feature_array = np.array(all_input_features_dict["syntax_feature"])
        syntax_feature_names = all_input_features_dict["syntax_feature_names"]
        # Get the index of the feature name where they contain any of the
        # component in the string content
        components = ["depth", "location", "idx"]
        syntax_to_onehot_idx = []
        for i, feature_name in enumerate(syntax_feature_names):
            if any(comp in feature_name.lower() for comp in components):
                syntax_to_onehot_idx.append(i)
        syntax_to_onehot_idx = np.array(syntax_to_onehot_idx)
        syntax_feature_mask = np.ones_like(syntax_feature_names, dtype=bool)
        # Floating point features to be excluded from one-hot encoding
        syntax_feature_mask[syntax_to_onehot_idx] = False
        syntax_feature_array = syntax_feature_array[:, syntax_feature_mask]
        encoder = OneHotEncoder(sparse_output=False)
        onehot_encoded_columns = []
        for original_syntax_feat in syntax_feature_array.T:
            original_syntax_feat = original_syntax_feat.reshape(-1, 1)
            onehot_encoded_col = encoder.fit_transform(original_syntax_feat)
            onehot_encoded_columns.append(onehot_encoded_col)
        syntax_feature_onehot = np.concatenate(onehot_encoded_columns, axis=1)
        all_input_features_dict["syntax_feature"] = syntax_feature_onehot
    if (
        add_additional_syntax_features
        and "syntax_feature" in all_input_features_dict.keys()
    ):
        # Use word_head_idx to look up the word embedding of the head
        syntax_feature_names = all_input_features_dict["syntax_feature_names"]

        word_head_idx_idx = syntax_feature_names.index("word_head_idx")
        word_head_indices = all_input_features_dict["syntax_feature"][
            :, word_head_idx_idx
        ].astype(int)

        # Look up the corresponding dnn_word_embedding for each head index
        dnn_word_embeddings = all_input_features_dict["dnn_word_embedding"]
        head_word_embeddings = dnn_word_embeddings[word_head_indices]
        all_input_features_dict["syntax_head_word_embedding"] = head_word_embeddings
        selected_input_components.append("syntax_head_word_embedding")

    if one_hot_encode_syntax_separate:
        # One hot encode individual columns within syntax feature like above
        # But save the one-hot encoded syntax features as separate components in the dictionary
        # Syntax feature indices mapping
        syntax_feature_idx = {
            "POS": 0,
            "Dependency_Label": 1,
            # "Constituent_Label": 2,
            "Tree_Depth": 3,
            # "Tree_Depth_Normed": 4,
            "Word_Position": 5,
            # "Word_Position_Normed": 6,
        }
        from sklearn.preprocessing import OneHotEncoder

        syntax_feature_array = np.array(all_input_features_dict["syntax_feature"])
        encoder = OneHotEncoder(sparse_output=False)
        for feature_name, idx in syntax_feature_idx.items():
            original_syntax_feat = syntax_feature_array[:, idx].reshape(-1, 1)
            onehot_encoded_col = encoder.fit_transform(original_syntax_feat)
            all_input_features_dict[f"syntax_{feature_name}_OH"] = onehot_encoded_col
            selected_input_components.append(f"syntax_{feature_name}_OH")
        # Remove the original syntax_feature from selected_input_components
        selected_input_components.remove("syntax_feature")
        del all_input_features_dict["syntax_feature"]

    if one_hot_encode_metadata and "metadata" in all_input_features_dict.keys():
        # First check if metadata is in the dictionary
        if "metadata" in all_input_features_dict.keys():
            from sklearn.preprocessing import OneHotEncoder

            # Onehot encode the first column "SpeakerID-OH" and second column "ChapterID-OH"
            speakerIDS = all_input_features_dict["metadata"][:, 0]
            chapterIDS = all_input_features_dict["metadata"][:, 1]
            encoder = OneHotEncoder(sparse_output=False)
            speakerIDS_onehot = encoder.fit_transform(speakerIDS.reshape(-1, 1))
            # chapterIDS_onehot = encoder.fit_transform(chapterIDS.reshape(-1, 1))
            all_input_features_dict["SpeakerID-OH"] = speakerIDS_onehot
            # all_input_features_dict["ChapterID-OH"] = chapterIDS_onehot
            # Delete the metadata key from the dictionary
            del all_input_features_dict["metadata"]
            selected_input_components += ["SpeakerID-OH"]  # , "ChapterID-OH"]
            selected_input_components.remove("metadata")

        else:
            logger.warning(
                "Metadata feature not found in input features, skipping one-hot encoding for metadata."
            )

    if argmax_ppg:
        # We convert the 40 dimensional PPG features into a Phoneme ID by taking the argmax
        ppg_features = all_input_features_dict["ppg_feature"]
        ppg_phoneme_ids = np.argmax(ppg_features, axis=1).reshape(-1, 1)
        all_input_features_dict["ppg_feature"] = ppg_phoneme_ids
    if normalize_features:
        normalize_groups = [
            "eGeMAPSv02",
        ]
        from sklearn.preprocessing import StandardScaler

        for group_name in normalize_groups:
            # We normalize features in a column-wise manner so that each feature has zero mean and unit variance
            scaler = StandardScaler()
            # We only want to normalize the features inside normalize_groups
            feature_sets_to_normalize = all_input_features_dict[group_name]
            feature_sets_normalized = scaler.fit_transform(feature_sets_to_normalize)
            # Replace the normalized features back to feature_sets
            all_input_features_dict[group_name] = feature_sets_normalized
            logger.info(f"Input features {group_name} normalized.")

    # Remove features not in selected_input_components
    for feature_name in list(all_input_features_dict.keys()):
        if feature_name not in selected_input_components:
            del all_input_features_dict[feature_name]

    logger.info("Concatenating input features...")
    logger.info(f"Input features include: {selected_input_components}")
    # Now concatenate all input features to form the final input feature matrix
    feature_sets = np.array(
        np.concatenate(
            [
                all_input_features_dict[feature_name]
                for feature_name in selected_input_components
            ],
            axis=1,
        )
    )

    data_shape = {
        feature_name: all_input_features_dict[feature_name][0].shape
        for feature_name in selected_input_components
    }

    # model_hidden_states shape should be (num_frames, num_layers, hidden_size)
    logger.info(f"Processed X shape: {feature_sets.shape}")
    model_hidden_states = np.array(model_hidden_states)  # Convert list to numpy array
    if select_layers is not None:
        logger.info(f"Selecting layers: {select_layers}")
        model_hidden_states = model_hidden_states[:, select_layers, :]

        logger.info(
            f"Selected model_hidden_states shape after layer selection: {model_hidden_states.shape}"
        )

    logger.info(f"Processed Y shape: {model_hidden_states.shape}")

    return feature_sets, model_hidden_states, data_shape


def load_data(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    selected_input_components: list = INPUT_FEATURE_SELECT_COMPONENTS,
    seq_sampling: str = "random_frames",
    select_layers: list | None = None,
    overwrite: bool = False,
    normalize_features: bool = True,
    one_hot_encode_syntax: bool = True,
    add_additional_syntax_features: bool = False,
    one_hot_encode_syntax_separate: bool = False,
    one_hot_encode_metadata: bool = True,
    argmax_ppg: bool = False,
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

    processed_feature_sets, processed_model_hidden_states, data_shape = further_process(
        model_hidden_states,
        feature_sets,
        selected_input_components=selected_input_components,
        reduce_dnn_word_embedding=reduce_dnn_word_embedding,
        one_hot_encode_syntax=one_hot_encode_syntax,
        add_additional_syntax_features=add_additional_syntax_features,
        one_hot_encode_syntax_separate=one_hot_encode_syntax_separate,
        one_hot_encode_metadata=one_hot_encode_metadata,
        argmax_ppg=argmax_ppg,
        normalize_features=normalize_features,
        select_layers=select_layers,
    )

    return (
        processed_feature_sets,
        processed_model_hidden_states,
        filename_timestamp,
        data_shape,
    )


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


def get_section_shapes(data_shape: dict) -> np.ndarray:
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

    return np.array(section_shapes)


if __name__ == "__main__":
    librispeech_split = "dev-clean"
    modelname = "facebook/hubert-base-ls960"
    modelname = "facebook/wav2vec2-base"
    # modelname = "google-bert/bert-base-uncased"
    seq_sampling = "random_frames"
    overwrite = False
    select_layers = [0, 6, 12]
    all_input_features_dict, model_hidden_states, filename_timestamp = format_data(
        librispeech_split=librispeech_split,
        modelname=modelname,
        seq_sampling=seq_sampling,
    )
