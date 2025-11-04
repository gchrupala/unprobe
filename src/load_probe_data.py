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
    processed_dnn_word_embeddings = []

    for fileID in tqdm(valid_fileIDs):
        spk_embedding = np.array(all_speaker_embeddings[fileID])
        ppg_features = np.array(all_ppg_features[fileID])
        # Swap the dimensions in ppg_features to be (time, ppg_dim)
        ppg_features = ppg_features.transpose((1, 0))
        fasttext_embedding = np.array(all_fasttext_word_embeddings[fileID])
        word_form_embeddings = all_word_form_embeddings[fileID]
        dnn_word_embeddings = all_dnn_word_embeddings[fileID]

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
            utt_lld_frames = utt_lld_frames.flatten()
            utt_lld_names = utt_lld.columns.tolist()

            # word_embedding = fasttext_embedding[word_idx].flatten()
            syntax_feature = np.array(syntax_feats)[word_idx].flatten()
            word_form_feature = word_form_embeddings[word_idx].flatten()

            # Find the corresponding DNN word embedding
            char_start_idx = ort_alignment.loc[word_idx, "char_idx_start"]
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

            processed_dnn_word_embeddings.append(dnn_word_embedding)

            # Get the corresponding PPG features for current frame index
            ppg_feature = ppg_features[int(token_time // 10)].flatten()

            # Make sure all the dimensions are correct
            assert all(
                (
                    # word_embedding.shape[0] == 100,
                    # syntax_feature.shape[0] == 34,
                    ppg_feature.shape[0] == 40,
                    spk_embedding.shape[0] == 100,
                    metadata.shape[0] == 2,
                    # word_form_feature.shape[0] == 29,
                )
            )

            # Concatenate all the features to form the input feature vector
            input_feature = np.concatenate(
                [
                    utt_lld_frames,
                    # word_embedding,
                    syntax_feature,
                    ppg_feature,
                    spk_embedding,
                    metadata,
                    # word_form_feature,
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
                # "word_embedding": word_embedding.shape,
                "syntax_feature": syntax_feature.shape,
                "ppg_feature": ppg_feature.shape,
                "spk_embedding": spk_embedding.shape,
                "metadata": metadata.shape,
                # "word_form_feature": word_form_feature.shape,
                "input_feature_all": input_feature.shape,
                "dnn_hidden_state": utt_dnn_hidden_state.shape,
            }

    processed_X = np.array(processed_X)
    processed_Y = np.array(processed_Y)

    # Apply dimension reduction to processed_dnn_word_embeddings using PCA to 100 dimensions
    processed_dnn_word_embeddings = np.array(processed_dnn_word_embeddings)
    logger.info(
        f"Processed DNN word embeddings shape before PCA: {processed_dnn_word_embeddings.shape}"
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    processed_dnn_word_embeddings = scaler.fit_transform(processed_dnn_word_embeddings)

    pca = PCA(n_components=100)
    processed_dnn_word_embeddings = pca.fit_transform(processed_dnn_word_embeddings)

    logger.info(
        f"Processed DNN word embeddings shape after PCA: {processed_dnn_word_embeddings.shape}"
    )
    data_shape["dnn_word_embedding"] = processed_dnn_word_embeddings.shape[1:]  # type: ignore
    # Then concatenate the processed_dnn_word_embeddings to processed_X along the last dimension
    processed_X = np.concatenate([processed_X, processed_dnn_word_embeddings], axis=1)

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

    # Make sure the directory exists
    os.makedirs(os.path.dirname(formatted_data_path), exist_ok=True)
    if os.path.exists(formatted_data_path) and not overwrite:
        logger.info(f"Loading formatted data from {formatted_data_path}")
        with open(formatted_data_path, "rb") as f:
            processed_X, processed_Y, filename_timestamp, data_shape = pickle.load(f)
    else:
        # If not, format the data and save it
        logger.info(
            "Formatted data not found or overwrite flag is set, formatting data..."
        )
        processed_X, processed_Y, filename_timestamp, data_shape = format_data(
            librispeech_split=librispeech_split,
            modelname=modelname,
            seq_sampling=seq_sampling,
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

    if normalize_features:
        from sklearn.preprocessing import StandardScaler

        # We normalize features in a column-wise manner so that each feature has zero mean and unit variance
        logger.info("Normalizing input features")
        scaler = StandardScaler()
        processed_X = scaler.fit_transform(processed_X)
        logger.info("Normalized input features")

    return processed_X, processed_Y, filename_timestamp, data_shape


def sanity_check_pca():
    from scipy.stats import loguniform
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
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
