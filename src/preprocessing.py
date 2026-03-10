import argparse
import glob
import logging
import os
import pickle
import sys

import benepar
import fasttext
import fasttext.util
import nltk
import numpy as np
import opensmile
import pandas as pd
import spacy
import textgrids
import torch
from datasets import Dataset
from tqdm.auto import tqdm

from utils import ALIGNMENT_ROOT, DATASET_ROOT, PROJECT_ROOT, SAVEPATH

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


_SPACY_NLP = None
_FASTTEXT_MODEL = None

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
    else torch.device("cpu")
)

DATASETPATH = DATASET_ROOT
ALIGNMENTPATH = ALIGNMENT_ROOT


def _get_spacy_benepar_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    nlp = spacy.load("en_core_web_sm")
    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    _SPACY_NLP = nlp
    return _SPACY_NLP


def _get_fasttext_model(target_dim: int = 100):
    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is None:
        fasttext.util.download_model("en", if_exists="ignore")
        _FASTTEXT_MODEL = fasttext.load_model("cc.en.300.bin")
        fasttext.util.reduce_model(_FASTTEXT_MODEL, target_dim)
    return _FASTTEXT_MODEL


def save_librispeech_tg_to_single_file(
    librispeech_split: str = "dev-clean",
    alignment_single_file_savepath: str = f"{PROJECT_ROOT}/data",
):
    """
    Extracting the textgrid files from the Librispeech dataset and save them to a single file for each tier.
    Args:
        librispeech_split (str, optional): split of librispeech to use. Defaults to "dev-clean".
        alignment_single_file_savepath (str, optional): path to save the single file. Defaults to f"{PROJECT_ROOT}/data".
    """
    if not os.path.exists(alignment_single_file_savepath):
        os.makedirs(alignment_single_file_savepath, exist_ok=True)

    librispeech_split_names = os.listdir(ALIGNMENTPATH)

    for name in librispeech_split_names:
        if name in librispeech_split:
            librispeech_split_name = name
            break
        else:
            librispeech_split_name = None

    if librispeech_split_name is None:
        raise ValueError(f"{librispeech_split} not found in {librispeech_split_names}")

    dataset_path = os.path.join(ALIGNMENTPATH, librispeech_split_name)
    transcription_files = glob.glob(f"{dataset_path}/**/*.TextGrid", recursive=True)
    # Sort transcription_files
    transcription_files = sorted(
        transcription_files,
        key=lambda x: os.path.splitext(os.path.basename(x))[0],
    )

    extracted_alignments = []
    for file_path in tqdm(transcription_files, desc="Reading textgrid files:"):
        tg = textgrids.TextGrid(file_path)
        fileID = file_path.split("/")[-1].split(".")[0]
        speakerid, chapter, utt = fileID.split("-")
        tg_dict = {}
        for tier in tg:
            tg_dict[tier] = np.array(
                [
                    (fileID, speakerid, chapter, utt, x.xmin, x.xmax, x.text)
                    for x in tg[tier]
                ]
            )
        extracted_alignments.append(tg_dict)

    tiernames = list(tg_dict.keys())  # type: ignore
    # For each tier we create a dataframe to contain all the alignment
    for tier in tiernames:
        tier_alignments = [x[tier] for x in extracted_alignments]
        tier_alignments = np.concatenate(tier_alignments, axis=0)
        df = pd.DataFrame(
            tier_alignments,
            columns=[
                "fileID",
                "speakerID",
                "chapter",
                "utterance",
                "start",
                "end",
                "text",
            ],
        )
        df["tier"] = tier

        df.to_csv(
            os.path.join(
                alignment_single_file_savepath,
                f"librispeech_{librispeech_split}_{tier}_alignment.csv",
            ),
            index=False,
            sep="\t",
        )


def efficient_syntax_parsing(transcriptions: list[dict]) -> list[np.ndarray]:
    """Efficiently extract syntax features from transcriptions.
    We aim to construct a syntactic feature extractor that functions similar to the openSMILE acoustic feature extractor. Using the textgrid information, we can extract the syntactic features of each token in the utterance and save the syntactic features in a similar way to the openSMILE acoustic feature extractor.
    We aim to have the following features for each token:
    - POS tag: The part-of-speech tag of the word
    - Dependency label: The dependency label of the word
    - Constituent label: The constituent label of the word
    - Constituency tree position: The position of the word in the constituency tree
    - Length of the sentence in numbers of words/tokens
    - Tree depth
    - Tree depth normalized: The depth of the word in the constituency tree normalized by the height of the tree
    - Word character length: The length of the word in characters #TODO move to metadata
    - Location in sentence: The location of the word in the sentence in words/tokens
    #TODO create subgrouping inside each "group" of features. e.g. word level features, sentence level features, etc.
    Args:
        transcriptions (list(dict)): List of dictionaries containing transcriptions.
    Returns:
        list(np.ndarray): List of numpy arrays containing syntax features for each transcription.
    """

    list_of_all_sents = [" ".join(x["words"]) for x in transcriptions]

    # Load syntax parsing models
    nlp = _get_spacy_benepar_nlp()
    # tagger_labels = nlp.get_pipe("tagger").labels  # type: ignore
    # tagger_label_dict = {label: i for i, label in enumerate(tagger_labels)}
    # parser_labels = nlp.get_pipe("parser").labels  # type: ignore
    # parser_label_dict = {label: i for i, label in enumerate(parser_labels)}
    all_syntax_feats = []
    for doc in tqdm(
        nlp.pipe(
            list_of_all_sents,
            disable=["tok2vec", "senter", "attribute_ruler", "ner", "lemmatizer"],
        ),
        desc="Extracting syntax features",
        total=len(list_of_all_sents),
    ):
        sent = list(doc.sents)[0]
        nltk_tree = nltk.Tree.fromstring(sent._.parse_string)

        # For every word in the sentence, extract paper-aligned syntax features.
        syntax_feats = []
        for i, word in enumerate(sent):
            # Use the text to get the constituency label from the nltk tree
            node_location_in_tree = nltk_tree.leaf_treeposition(i)

            # Word location (depth) in tree
            node_depth_in_tree = len(node_location_in_tree)
            total_tree_depth = nltk_tree.height() - 1
            total_word_count = len(sent)

            word_location_in_sentence = i + 1

            word_features = np.array(
                [
                    word.pos,
                    word.dep,
                    node_depth_in_tree,
                    word_location_in_sentence,
                    total_tree_depth,
                    total_word_count,
                ]
            )
            syntax_feats.append(word_features)

        all_syntax_feats.append(np.array(syntax_feats))

        # Pad all_syntax_feats to the same dimension on the second axis
        max_feat_len = max([x.shape[1] for x in all_syntax_feats])
        for i, feats in enumerate(all_syntax_feats):
            if feats.shape[1] < max_feat_len:
                padding = np.zeros((feats.shape[0], max_feat_len - feats.shape[1]))
                all_syntax_feats[i] = np.hstack([feats, padding])
    return all_syntax_feats


def load_librispeech_MAUS_alignment(
    librispeech_split: str = "dev-clean", transcription_savefile: str | None = None
) -> list[dict]:
    """
    phone_alignment = "_MAU_alignment"
    word_alignment = "_ORT-MAU_alignment"
    """
    ort_alignment = pd.read_csv(
        f"{PROJECT_ROOT}/data/librispeech_{librispeech_split}_ORT-MAU_alignment.csv",
        sep="\t",
    )
    phone_alignment = pd.read_csv(
        f"{PROJECT_ROOT}/data/librispeech_{librispeech_split}_MAU_alignment.csv",
        sep="\t",
    )

    # Use groupby to get list of lists of phones/words for each fileID
    words_df = ort_alignment.dropna().groupby("fileID")["text"].agg(list).reset_index()
    phones_df = (
        phone_alignment.dropna().groupby("fileID")["text"].agg(list).reset_index()
    )
    # Rename the columns
    words_df.columns = ["fileID", "words"]
    phones_df.columns = ["fileID", "phones"]
    # Join the two dataframes on fileID
    transcriptions_df = words_df.merge(phones_df, on="fileID", how="inner")

    # Turn the dataframe into a list of dictionaries
    transcriptions = transcriptions_df.to_dict(orient="records")

    # Turn ort_alignment and phone_alignment into a dictionary of dataframes for each fileID
    ort_alignment = {
        fileID: df.drop(columns=["fileID"]).reset_index(drop=True)
        for fileID, df in ort_alignment.groupby("fileID")
    }
    phone_alignment = {
        fileID: df.drop(columns=["fileID"]).reset_index(drop=True)
        for fileID, df in phone_alignment.groupby("fileID")
    }

    for example in tqdm(transcriptions, desc="Adding non-acoustic features:"):
        speakerid, chapter, utt = example["fileID"].split("-")
        example["non_acoustic"] = np.array([int(speakerid), int(chapter)])
        example["ort_alignment"] = ort_alignment[example["fileID"]]
        example["phone_alignment"] = phone_alignment[example["fileID"]]

    logger.info("Non-acoustic features added to transcriptions.")

    all_syntax_feats = efficient_syntax_parsing(transcriptions)
    # Merge all_syntax_feats into transcriptions
    for i, example in enumerate(transcriptions):
        example["syntax_feats"] = all_syntax_feats[i]
    logger.info("Syntax features extracted.")
    logger.info("Syntax features added to transcriptions.")

    logger.info("Saving transcriptions...")

    if transcription_savefile is None:
        return transcriptions
    else:
        with open(transcription_savefile, "wb") as f:
            pickle.dump(transcriptions, f)
        logger.info(f"Transcriptions saved to {transcription_savefile}")
        return transcriptions


def load_librispeech(split: str = "dev-clean") -> Dataset:
    """Loading Librispeech dataset into Huggingface Dataset format

    Args:
        split (str, optional): split of librispeech to use. Defaults to "dev-clean".

    Returns:
        datasets.Dataset: Huggingface Dataset object containing columns of [fileID, sent, audio]
    """
    dataset_path = f"{DATASETPATH}/{split}"
    transcription_files = glob.glob(f"{dataset_path}/**/*.trans.txt", recursive=True)
    fileIDs, sentences = [], []
    for file_path in transcription_files:
        with open(file_path, "r") as f:
            rows = f.read().splitlines()
        for row in rows:
            fileID, sentence = row.split(" ", 1)
            fileIDs.append(fileID)
            sentences.append(sentence)

    transcript_data = tuple(zip(fileIDs, sentences))

    df = pd.DataFrame(transcript_data, columns=["fileID", "sent"])

    def get_wav_file(fileID):
        spkid, chapter, utt = fileID.split("-")

        return os.path.join(dataset_path, spkid, chapter, f"{fileID}.flac")

    df["speakerid"] = df["fileID"].apply(lambda x: x.split("-")[0])
    df["chapter"] = df["fileID"].apply(lambda x: x.split("-")[1])
    df["utterance"] = df["fileID"].apply(lambda x: x.split("-")[2])

    df["audio"] = df["fileID"].map(get_wav_file)

    dataset = Dataset.from_pandas(df)
    # Sort by fileID
    dataset = dataset.sort("fileID")

    return dataset


def extract_syntax_features(dataset: Dataset, **kwargs) -> dict[str, np.ndarray]:
    """Extract syntax features using spacy

    Args:
        dataset (Dataset): dataset loaded with load_librispeech()

    Returns:
        dict[str, np.ndarray]: syntax features extracted from the dataset with fileIDs as keys
    """

    logger.info("Extracting syntax features from text")
    nlp = _get_spacy_benepar_nlp()
    all_syntax_feats = {}

    for example in tqdm(dataset, desc="Syntax Feature Extraction"):
        fileID = example["fileID"]  # type: ignore
        doc = nlp(example["sent"])  # type: ignore
        sent = list(doc.sents)[0]
        nltk_tree = nltk.Tree.fromstring(sent._.parse_string)

        syntax_feats = []
        offset_mapping = []
        for i, word in enumerate(sent):
            node_location_in_tree = nltk_tree.leaf_treeposition(i)

            # Word location (depth) in tree
            node_depth_in_tree = len(node_location_in_tree)
            total_tree_depth = nltk_tree.height() - 1
            total_word_count = len(sent)

            word_location_in_sentence = i + 1

            word_features = np.array(
                [
                    word.pos,
                    word.dep,
                    node_depth_in_tree,
                    word_location_in_sentence,
                    total_tree_depth,
                    total_word_count,
                ]
            )
            syntax_feats.append(word_features)

            # Create offset mapping
            word_start = word.idx
            word_end = word.idx + len(word.text)
            offset_mapping.append((word_start, word_end))
        feature_names = [
            "pos",
            "dep",
            "node_depth_in_tree",
            "word_location_in_sentence",
            "total_tree_depth",
            "total_word_count",
        ]
        if len(feature_names) != len(syntax_feats[0]):
            raise ValueError(
                "Feature names length does not match syntax features length "
                f"for fileID={fileID}: names={len(feature_names)}, "
                f"features={len(syntax_feats[0])}"
            )

        all_syntax_feats[fileID] = {
            "features": np.array(syntax_feats),
            "offset_mapping": offset_mapping,
            "words": [word.text for word in sent],
            "names": feature_names,
        }

    return all_syntax_feats


def extract_opensmile_features(
    dataset: Dataset, feature_set: str = "eGeMAPSv02", **kwargs
) -> pd.DataFrame:
    """Extracting opensmile features from audio file

    Args:
        dataset (datasets.Dataset): dataset containing audio
        feature_set (str, optional): opensmile feature set to use. Defaults to "eGeMAPSv02".

    Returns:
        pd.DataFrame: DataFrame containing opensmile features
    """

    if kwargs["feature_level"] == "lld":
        feature_level = opensmile.FeatureLevel.LowLevelDescriptors
    elif kwargs["feature_level"] == "functionals":
        feature_level = opensmile.FeatureLevel.Functionals
    else:
        feature_level = opensmile.FeatureLevel.Functionals

    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
        verbose=True,
        num_workers=8,
        sampling_rate=16000,
        resample=True,
    )

    files = dataset["audio"]
    opensmile_features = smile.process_files(files)
    if kwargs["feature_level"] == "lld":
        return opensmile_features
    else:
        return opensmile_features.reset_index()


def extract_speaker_embedding(dataset: Dataset, **kwargs) -> dict[str, np.ndarray]:
    from datasets import Audio
    from pyannote.audio import Model

    spk_embd_model = Model.from_pretrained("pyannote/embedding")
    spk_embd_model.to(device)
    logger.info(f"Extracting speaker embedding from audio file. Device:{device}")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    speaker_embedding_features = {}

    for example in tqdm(dataset, desc="Speaker Embedding Extraction"):
        fileID = example["fileID"]  # type: ignore
        audio_array = example["audio"]["array"]  # type: ignore
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        with torch.no_grad():
            spk_embs = spk_embd_model(
                audio_tensor.to(device=device, dtype=torch.float32)
            )
        speaker_embedding_features[fileID] = spk_embs.cpu().squeeze().numpy()

    return speaker_embedding_features


def extract_phonetic_posteriorgram(dataset: Dataset, **kwargs) -> dict[str, np.ndarray]:
    import ppgs
    from datasets import Audio

    logger.info("Extracting ppgs features from audio file")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=ppgs.SAMPLE_RATE))
    logger.info("Audio column recasted to correct sampling rate.")

    ppgs_features = {}
    for example in tqdm(dataset, desc="PPG Extraction"):
        fileID = example["fileID"]  # type: ignore
        audio_tensor = torch.from_numpy(example["audio"]["array"]).unsqueeze(0)
        if device == torch.device("cpu"):
            ppgs_feat = ppgs.from_audio(audio_tensor, sample_rate=ppgs.SAMPLE_RATE)
            ppgs_features[fileID] = ppgs_feat.double().squeeze().numpy()
        else:
            ppgs_feat = ppgs.from_audio(
                audio_tensor, sample_rate=ppgs.SAMPLE_RATE, gpu=0
            )
            ppgs_features[fileID] = ppgs_feat.cpu().squeeze().numpy()
    return ppgs_features


def extract_fasttext_embeddings(dataset: Dataset, **kwargs) -> dict[str, np.ndarray]:
    ft = _get_fasttext_model(target_dim=100)

    logger.info("Extracting fasttext word embeddings from text")
    fasttext_embeddings = {}

    for example in tqdm(dataset, desc="FastText Embedding Extraction"):
        fileID = example["fileID"]  # type: ignore
        tokens = [x for x in example["tokens"] if x != "<pad>"]  # type: ignore
        embeddings = [ft.get_word_vector(word) for word in tokens]
        fasttext_embeddings[fileID] = np.array(embeddings)

    return fasttext_embeddings


def extract_special_features(dataset: Dataset, **kwargs) -> dict[str, np.ndarray]:
    """Experimental combined extractor (legacy).

    This path is kept for backward compatibility and is not used by the
    default preprocessing workflow.
    """

    import ppgs
    import umap.umap_ as umap
    from datasets import Audio
    from pyannote.audio import Model

    logger.warning(
        "extract_special_features is a legacy experimental path and may be slow."
    )

    spk_embd_model = Model.from_pretrained("pyannote/embedding")
    spk_embd_model.to(device)
    # Cast the audio column to the right sampling rate
    logger.info(f"Recasting audio sampling rate: {ppgs.SAMPLE_RATE}")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=ppgs.SAMPLE_RATE))
    logger.info("Audio column recasted to correct sampling rate.")

    ft = _get_fasttext_model(target_dim=100)

    logger.info("""Extracting ppgs features, speaker embedding from audio file""")
    special_features = {}

    def _map_example(example: dict) -> dict:
        audio_tensor = torch.from_numpy(example["audio"]["array"]).unsqueeze(0)
        if device == torch.device("cpu"):
            ppgs_features = ppgs.from_audio(audio_tensor, sample_rate=ppgs.SAMPLE_RATE)
            example["ppgs"] = ppgs_features.double().squeeze().numpy()
        else:
            ppgs_features = ppgs.from_audio(
                audio_tensor, sample_rate=ppgs.SAMPLE_RATE, gpu=0
            )
            example["ppgs"] = ppgs_features.cpu().squeeze().numpy()

        tokens = [x for x in example["tokens"] if x != "<pad>"]
        fasttext_embeddings = [ft.get_word_vector(word) for word in tokens]
        example["fasttext"] = fasttext_embeddings

        with torch.no_grad():
            spk_embs = spk_embd_model(
                audio_tensor.to(device=device, dtype=torch.float32)
            )
        example["spk_emb"] = spk_embs.cpu().squeeze().numpy()

        return example

    dataset = dataset.map(
        _map_example,
        remove_columns=["sent", "audio", "speakerid", "chapter", "utterance", "tokens"],
        desc="Extracting text embeddings, phonetic posteriorgram, speaker embedding",
    )

    spk_emb_array = np.array(dataset["spk_emb"])
    # Use UMAP to reduce speaker embedding to 100 dimensions
    logger.info("Reducing speaker embedding to 100 dimensions using UMAP")
    reducer = umap.UMAP(n_components=100)  # , random_state=42)
    reduced = reducer.fit_transform(spk_emb_array)

    # Put the reduced speaker embeddings into special_features
    special_features = {}
    for i in range(len(dataset)):
        fileID = dataset[i]["fileID"]
        special_features[fileID] = {
            "ppgs": dataset[i]["ppgs"],
            "fasttext": np.array(dataset[i]["fasttext"]),
            "spk_emb": reduced[i],  # type: ignore
        }

    return special_features


def extract_audio_representation(
    dataset: Dataset,
    modelname: str = "facebook/wav2vec2-base",
    device: torch.device = torch.device("cuda"),
    random_seed: int = 42,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Extracting audio representation from audio file

    Args:
        dataset (datasets.Dataset): dataset containing audio
        modelname (str, optional): model name to use. Defaults to "facebook/wav2vec2-base".
        device (str, optional): device to run the model on. Defaults to "cuda".

    Returns:
        dict[str, np.ndarray]: Dictionary containing audio representations and the corresponding frame timestamps in ms
    """
    from datasets import Audio
    from transformers import AutoFeatureExtractor, AutoModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname)
    seq_sampling = kwargs.get("seq_sampling", "random_frames").lower()
    args_n_frames = kwargs.get("n_frames", 10)

    model.to(device)  # type: ignore
    logger.info(f"Using device: {device}")
    model.eval()

    # Set the random state of np random generator
    rng = np.random.default_rng(random_seed)

    # Cast the audio column to the right sampling rate
    logger.info(f"Recasting audio sampling rate: {feature_extractor.sampling_rate}")
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
    )
    logger.info("Audio column recasted to correct sampling rate.")

    audio_representations = {}
    num_skipped_short_sequences = 0
    for example in tqdm(
        dataset, desc=f"Extracting audio representations with {modelname}"
    ):
        waveform = example["audio"]["array"]  # type: ignore
        fileID = example["fileID"]  # type: ignore
        waveform = waveform.squeeze()
        inputs = feature_extractor(
            waveform, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        )
        n_frames = args_n_frames
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # output shape need to be (batch_size, seq_len, hidden_size)
        # Save all hidden states and preserve seq_len dimension with shape (batch_size, layer, seq_len, hidden_size)
        hidden_states = outputs.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0)
        raw_hidden_state_shape = hidden_states.shape
        hidden_states = hidden_states.to("cpu")
        if seq_sampling == "mean":
            # Take the mean over the seq_len dimension
            hidden_states = hidden_states.mean(dim=2, keepdim=True)
            frame_indices = None
        elif seq_sampling == "none":
            frame_indices = np.arange(hidden_states.shape[2])
        elif seq_sampling == "random_frames":
            # We randomly select n_frames from the hidden states
            if n_frames > hidden_states.shape[2]:
                num_skipped_short_sequences += 1
                continue
            # Randomly select n_frames from the hidden states along the seq_len dimension
            # We do this to avoid using too much memory and disk space
            frame_indices = rng.choice(hidden_states.shape[2], n_frames, replace=False)
            frame_indices = np.sort(frame_indices)
        else:
            raise ValueError(f"Unknown seq_sampling method: {seq_sampling}")

        # Select the frames from the hidden states
        selected_hidden_states = hidden_states[:, :, frame_indices, :]

        # We also want to convert the indices to the original time stamps in ms
        # So we can use the timestamps to lookup the corresponding text tokens
        # We will bypass the issue of wav2vec2 "frame rates" by using the total number of frames and the original audio length
        audio_length_ms = (
            len(waveform) / feature_extractor.sampling_rate * 1000
        )  # in ms
        frame_indices_in_ms = (
            (frame_indices / raw_hidden_state_shape[2]) * audio_length_ms
            if frame_indices is not None
            else None
        )
        # Round the frame_indices_in_ms to the nearest 20ms
        frame_indices_in_ms = (
            np.round(frame_indices_in_ms / 20) * 20
            if frame_indices_in_ms is not None
            else None
        )

        # Store the hidden states in a dictionary with the fileID as key
        audio_representations[fileID] = {
            "hidden_states": selected_hidden_states.cpu().squeeze().numpy(),
            "frame_token_indices": {
                "frame_indices": frame_indices,
                "frame_indices_in_ms": frame_indices_in_ms,
            },
        }

    if num_skipped_short_sequences > 0:
        logger.info(
            "Skipped %s examples with fewer frames than n_frames=%s",
            num_skipped_short_sequences,
            kwargs.get("n_frames", 10),
        )

    return audio_representations


def extract_dnn_word_embedding(
    dataset: Dataset,
    modelname: str = "bert-base-uncased",
    device: torch.device = torch.device("cuda"),
    **kwargs,
) -> dict[str, np.ndarray]:
    """Extract word embedding using DNN models such as BERT

    Args:
        dataset (Dataset): The dataset to extract embeddings from.
        modelname (str, optional): The name of the model to use. Defaults to "bert-base-uncased".
        device (torch.device, optional): The device to run the model on. Defaults to torch.device("cuda").

    Returns:
        dict[str, np.ndarray]: A dictionary mapping file IDs to their corresponding word embeddings.
    """
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    model.to(device)
    logger.info(f"Using device: {device}")
    model.eval()

    word_embeddings = {}
    for example in tqdm(
        dataset, desc=f"Extracting DNN word embedding with {modelname}"
    ):
        text = example["sent"]  # type: ignore

        encoded_input = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        word_ids = encoded_input.word_ids()
        offset_mapping = encoded_input.pop("offset_mapping").cpu().squeeze().numpy()

        inputs = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get the non-contextualized token embeddings from the 0-th layer

        token_embeddings = (
            outputs.hidden_states[0].cpu().squeeze(0)
        )  # (seq_len, hidden_size)

        # Aggregate Sub-words into Word Embeddings
        word_level_embeddings = []
        word_list = []

        # Get unique word IDs, removing None (which corresponds to [CLS] and [SEP])
        unique_word_ids = set(w for w in word_ids if w is not None)

        for w_id in sorted(list(unique_word_ids)):
            indices = [i for i, x in enumerate(word_ids) if x == w_id]
            selected_embeddings = token_embeddings[indices]
            avg_embedding = selected_embeddings.mean(dim=0)
            word_level_embeddings.append(avg_embedding)

            start_char = offset_mapping[indices[0]][0]
            end_char = offset_mapping[indices[-1]][1]
            current_word = text[start_char:end_char]
            word_list.append(current_word)

        # Convert list of tensors to a single tensor or numpy array
        word_level_embeddings = torch.stack(word_level_embeddings).cpu().numpy()

        fileID = example["fileID"]  # type: ignore
        tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])

        word_embeddings[fileID] = {
            "word_embeddings": word_level_embeddings,
            "words": word_list,
            "tokens": tokens,
            "offset_mapping": offset_mapping,
            "word_ids": word_ids,
        }

    return word_embeddings


def extract_text_representation(
    dataset: Dataset,
    modelname: str = "answerdotai/ModernBERT-base",
    device: torch.device = torch.device("cuda"),
    random_seed: int = 42,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Extracting text representation from text

    Args:
        dataset (datasets.Dataset): dataset containing text
        modelname (str, optional): model name to use. Defaults to "answerdotai/ModernBERT-base".
        device (str, optional): device to run the model on. Defaults to "cuda".
    Returns:
        dict[str, np.ndarray]: Dictionary containing text representations and the corresponding token indices
    """
    from transformers import AutoModel, AutoTokenizer

    if "modernbert" in modelname.lower():
        model = AutoModel.from_pretrained(modelname, reference_compile=False)
    else:
        model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    seq_sampling = kwargs.get("seq_sampling", "random_frames").lower()

    model.to(device)
    logger.info(f"Using device: {device}")
    model.eval()

    rng = np.random.default_rng(random_seed)

    text_representations = {}

    for example in tqdm(
        dataset, desc=f"Extracting text representations with {modelname}"
    ):
        n_frames = kwargs.get("n_frames", 5)

        encoded_input = tokenizer(
            example["sent"],  # type: ignore
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        offset_mapping = encoded_input.pop("offset_mapping").cpu().squeeze().numpy()
        fileID = example["fileID"]  # type: ignore
        word_ids = encoded_input.word_ids()
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            outputs = model(**encoded_input, output_hidden_states=True)
            # output shape need to be (batch_size, seq_len, hidden_size)
            # Save all hidden states and preserve seq_len dimension with shape (batch_size, layer, seq_len, hidden_size)
            hidden_states = outputs.hidden_states
        hidden_states = torch.stack(hidden_states, dim=1)

        if seq_sampling == "mean":
            # Take the mean over the seq_len dimension
            hidden_states = hidden_states.mean(dim=2)
            frame_indices = None
        elif seq_sampling == "random_frames":
            if n_frames > hidden_states.shape[2]:
                # Skip examples where there are too limited amount of tokens
                continue
            frame_indices = rng.choice(hidden_states.shape[2], n_frames, replace=False)
            frame_indices = np.sort(frame_indices)

        elif seq_sampling == "none":
            frame_indices = np.arange(hidden_states.shape[2])
            pass

        else:
            raise ValueError(f"Unknown sequence sampling method: {seq_sampling}")
        if frame_indices is not None:
            selected_hidden_states = hidden_states[:, :, frame_indices, :]
        else:
            selected_hidden_states = hidden_states

        # Store the hidden states in a dictionary with the fileID as key
        text_representations[fileID] = {
            "hidden_states": selected_hidden_states.cpu().squeeze().numpy(),
            "frame_token_indices": {
                "frame_indices": frame_indices,
                "offset_mapping": offset_mapping,
                "word_ids": word_ids,
            },
        }
    return text_representations


def transcription_to_string_embeddings(
    transcriptions, embedding_size=100, level="words", **kwargs
) -> list:
    """Extracting embedding based on the string level

    Args:
        level (str, optional): Select the level in transcription dictionary to turn into embedding. Defaults to "words".

    Returns:
        list: List of numpy arrays of embeddings
    """

    strings = [x[level] for x in transcriptions]
    flattened_all_strings = [item for sublist in strings for item in sublist]
    unique_strings = list(set(flattened_all_strings))
    # Sort unique strings by alphabetical order
    unique_strings.sort()
    unique_strings = ["<pad>"] + unique_strings
    token_to_idx = {token: i for i, token in enumerate(unique_strings)}
    embedding = torch.nn.Embedding(len(unique_strings), embedding_size)
    embedding_bag = torch.nn.EmbeddingBag.from_pretrained(embedding.weight, mode="mean")
    # Turn flattened_all_strings into embeddings
    utterance_embeddings = []
    for utterance in tqdm(strings, desc=f"Extracting {level} embeddings"):
        # First indexize the strings
        indices = [token_to_idx[y] for y in utterance]
        indices = torch.tensor(indices).unsqueeze(0)
        # Finally turn the indices into embeddings
        utterance_embeddings.append(embedding_bag(indices).detach().numpy().squeeze())

    # # First pad the strings to the same length with <pad> token
    # max_len = max([len(x) for x in strings])
    # strings = [x + ["<pad>"] * (max_len - len(x)) for x in strings]
    # # Then indexize the strings
    # indices = [[unique_strings.index(y) for y in x] for x in strings]
    # # Finally turn the indices into embeddings
    # utterance_embeddings = embedding_bag(torch.tensor(indices)).detach().numpy()

    # Save the embedding weights
    if "emb_savepath" in kwargs:
        with open(kwargs["emb_savepath"], "wb") as f:
            pickle.dump(utterance_embeddings, f)
    if "emb_weights_savepath" in kwargs:
        with open(kwargs["emb_weights_savepath"], "wb") as f:
            pickle.dump(embedding_bag.weight.detach().numpy(), f)
    if "emb_dict_savepath" in kwargs:
        with open(kwargs["emb_dict_savepath"], "wb") as f:
            pickle.dump(unique_strings, f)

    return utterance_embeddings


def get_letter_unigram_embeddings(dataset: Dataset, **kwargs) -> dict[str, np.ndarray]:
    """Extracting letter unigram embeddings from the dataset

    Args:
        dataset (datasets.Dataset): dataset containing text
    Returns:
        dict[str, np.ndarray]: Dictionary containing letter unigram embeddings
    """
    all_letters = set()
    for example in dataset:
        sent = example["sent"]  # type: ignore
        for letter in sent:
            all_letters.add(letter.lower())
    all_letters = sorted(list(all_letters))
    all_letters = ["<pad>"] + all_letters
    logger.info(f"Total unique letters: {len(all_letters)}")

    # For each word in each utterance in the dataset, we create a letter unigram embedding
    letter_unigram_embeddings = {}
    for example in tqdm(dataset, desc="Extracting letter unigram embeddings"):
        fileID = example["fileID"]  # type: ignore
        tokens = [x for x in example["tokens"] if x != "<pad>"]  # type: ignore
        utt_letter_unigram_embeddings = []
        for token in tokens:
            # Make an empty zero array so that we can count the frequency of the letter at respective indices
            token_letter_unigram = np.zeros((len(all_letters),), dtype=np.int32)
            if token == "<pad>":
                token_letter_unigram[0] = 1
            else:
                for letter in token:
                    letter_index = all_letters.index(letter.lower())
                    token_letter_unigram[letter_index] += 1
            utt_letter_unigram_embeddings.append(token_letter_unigram)
        letter_unigram_embeddings[fileID] = np.array(utt_letter_unigram_embeddings)

    return letter_unigram_embeddings


def process_dataset(
    librispeech_split: str = "dev-clean",
    savepath: str = SAVEPATH,
    overwrite: bool = False,
):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset = load_librispeech(librispeech_split)
    transcription_savefile = (
        f"{savepath}/librispeech-{librispeech_split}_transcriptions.pickle"
    )
    if os.path.exists(transcription_savefile) and not overwrite:
        logger.info(
            "Transcriptions already exist and not overwriting, loading from file..."
        )
        with open(transcription_savefile, "rb") as f:
            transcriptions = pickle.load(f)
    else:
        logger.info("Transcriptions do not exist or overwrite selected, extracting...")
        transcriptions = load_librispeech_MAUS_alignment(
            librispeech_split, transcription_savefile
        )

    # Remove fileID without alignment
    dataset_ID = dataset["fileID"]
    transcription_ID = [x["fileID"] for x in transcriptions]
    difference = list(set(dataset_ID) - set(transcription_ID))
    dataset = dataset.filter(lambda x: x["fileID"] not in difference)

    tokens_for_each_utt = [
        example["ort_alignment"]["text"].fillna("<pad>").tolist()
        for example in tqdm(transcriptions)
    ]
    dataset_file_ids = dataset["fileID"]
    if transcription_ID != dataset_file_ids:
        raise ValueError(
            "FileIDs do not match after alignment filtering. "
            f"transcriptions={len(transcription_ID)}, dataset={len(dataset_file_ids)}"
        )
    # Add tokens for each utt to the dataset
    dataset = dataset.add_column("tokens", tokens_for_each_utt)  # type: ignore

    return dataset, transcriptions


def extract_base_features(
    dataset: Dataset,
    librispeech_split: str = "dev-clean",
    savepath: str = SAVEPATH,
    overwrite: bool = False,
):
    """Extracting base features from the dataset and save them to disk.

    Args:
        dataset (datasets.Dataset): dataset containing audio and tokens
        librispeech_split (str, optional): split of librispeech to use. Defaults to "dev-clean".
        savepath (str, optional): path to save the features. Defaults to SAVEPATH.
        overwrite (bool, optional): whether to overwrite existing features. Defaults to False.
    """

    base_probe_inputs = {
        # "special_features": {
        #     "function": extract_special_features,
        #     "save_dir": f"{savepath}/librispeech-{librispeech_split}_special_features.pickle",
        # },
        "syntax_features": {
            "function": extract_syntax_features,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_syntax_features.pickle",
        },
        "speaker_embedding": {
            "function": extract_speaker_embedding,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_speaker_embedding.pickle",
        },
        "phonetic_posteriorgram": {
            "function": extract_phonetic_posteriorgram,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_phonetic_posteriorgram.pickle",
        },
        "fasttext_word_embeddings": {
            "function": extract_fasttext_embeddings,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_fasttext_word_embeddings.pickle",
        },
        "opensmile_features": {
            "function": extract_opensmile_features,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_opensmile_features.pickle",
            "feature_level": "functionals",
            "feature_set": "eGeMAPSv02",
        },
        "opensmile_features_lld": {
            "function": extract_opensmile_features,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_opensmile_features_lld.pickle",
            "feature_level": "lld",
            "feature_set": "eGeMAPSv02",
        },
        "letter_unigram_embeddings": {
            "function": get_letter_unigram_embeddings,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_letter_unigram_embeddings.pickle",
        },
        "dnn_word_embeddings": {
            "function": extract_dnn_word_embedding,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_dnn_word_embeddings.pickle",
            "modelname": "bert-base-uncased",
        },
    }

    for probe_data_type in tqdm(
        base_probe_inputs, desc="Running BASIC feature extraction:"
    ):
        feature = base_probe_inputs[probe_data_type]
        if not os.path.exists(feature["save_dir"]) or overwrite:
            logger.info(f"Extracting {probe_data_type} features...")
            extracted_feature = feature["function"](dataset, **feature)
            if isinstance(extracted_feature, pd.DataFrame):
                # extracted_feature.to_csv(feature["save_dir"], index=False)
                extracted_feature.to_pickle(feature["save_dir"])
            else:
                # torch.save(extracted_feature, feature["save_dir"])
                with open(feature["save_dir"], "wb") as f:
                    pickle.dump(extracted_feature, f)
        else:
            logger.info(f"{feature['save_dir']} already exists, skipping...")

    logger.info("Basic feature extraction completed!")


def extract_transformer_features(
    dataset: Dataset,
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    savepath: str = SAVEPATH,
    overwrite: bool = False,
    seq_sampling: str = "random_frames",
    n_frames: int = 5,
    random_seed: int = 42,
):
    """Extracting Transformer based features from the dataset and save them to disk.

    Args:
        dataset (datasets.Dataset): dataset containing audio and tokens
        librispeech_split (str, optional): split of librispeech to use. Defaults to "dev-clean".
        modelname (str, optional): The name of the model to use. Defaults to "facebook/wav2vec2-base".
        savepath (str, optional): path to save the features. Defaults to SAVEPATH.
        overwrite (bool, optional): Whether to overwrite existing features. Defaults to False.
    """

    # Extract Transformer model based features
    # First determine if the modality is audio or text
    if modelname in [
        "answerdotai/ModernBERT-base",
        "google-bert/bert-base-uncased",
        "FacebookAI/roberta-base",
    ]:
        extraction_function = extract_text_representation
    elif "wav" in modelname.lower() or "hubert" in modelname.lower():
        extraction_function = extract_audio_representation
    else:
        raise ValueError(f"Unknown modelname: {modelname}")

    transformer_feature_savepath = f"{savepath}/librispeech-{librispeech_split}_{modelname.split('/')[-1]}_representation_{seq_sampling}.pickle"
    if random_seed != 42:
        # Append random seed to differentiate save file
        transformer_feature_savepath = transformer_feature_savepath.replace(
            "_representation_", f"_representation-seed-{random_seed}_"
        )
    if not os.path.exists(transformer_feature_savepath) or overwrite:
        logger.info(
            f"Saving generated transformer features to {transformer_feature_savepath}"
        )
        transformer_features = extraction_function(
            dataset,
            modelname=modelname,
            device=device,  # type: ignore
            seq_aggregation="none",
            seq_sampling=seq_sampling,
            n_frames=n_frames,
            random_seed=random_seed,
        )
        with open(transformer_feature_savepath, "wb") as f:
            pickle.dump(transformer_features, f)
    else:
        logger.info(f"{transformer_feature_savepath} already exists, skipping...")

    logger.info("Transformer feature extraction completed!")


def extract_features(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    overwrite: bool = False,
    seq_sampling: str = "random_frames",
    n_frames: int = 5,
    random_seed: int = 42,
    do_base: bool = False,
    do_transformer: bool = True,
    overwrite_base: bool = False,
    overwrite_textgrid: bool = False,
    overwrite_transcriptions: bool = False,
):
    """Extracting features from the dataset and save them to disk.

    Args:
        librispeech_split (str, optional): split of librispeech to use. Defaults to "dev-clean".
        modelname (str, optional): The name of the model to use. Defaults to "facebook/wav2vec2-base".
        overwrite (bool, optional): Whether to overwrite existing features. Defaults to False.
        seq_sampling (str, optional): Sequence sampling method to use. Defaults to "random_frames".
        do_base (bool, optional): Only do base feature extraction. Defaults to False.
        overwrite_base (bool, optional): Overwrite base features even if they exist. Defaults to False.
    """

    if overwrite_textgrid:
        logger.info("Rewriting or saving textgrids into single file")
        logger.info("%s", "-" * 30)
        save_librispeech_tg_to_single_file(librispeech_split=librispeech_split)

    dataset, transcriptions = process_dataset(
        librispeech_split=librispeech_split,
        overwrite=overwrite_transcriptions,
    )

    if do_base:
        extract_base_features(
            dataset,
            librispeech_split=librispeech_split,
            overwrite=overwrite_base,
        )

    if do_transformer:
        extract_transformer_features(
            dataset,
            librispeech_split=librispeech_split,
            modelname=modelname,
            overwrite=overwrite,
            seq_sampling=seq_sampling,
            n_frames=n_frames,
            random_seed=random_seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="dev-clean",
        help="Librispeech split to use",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="facebook/wav2vec2-base",
        help="The name of the model to use. Choose from 'facebook/wav2vec2-base', 'facebook/wav2vec2-large-960h', 'answerdotai/ModernBERT-base'",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features",
    )
    parser.add_argument(
        "--seq_sampling",
        type=str,
        default="random_frames",
        help="Sequence sampling method to use. Choose from 'mean', 'random_frames', 'none'",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=5,
        help="Number of frames to sample if seq_sampling is 'random_frames'",
    )
    parser.add_argument(
        "--do_base",
        action="store_true",
        help="Do base feature extraction",
    )
    parser.add_argument(
        "--do_transformer",
        action="store_true",
        help="Do transformer feature extraction",
    )
    parser.add_argument(
        "--overwrite_base",
        action="store_true",
        help="Overwrite base features even if they exist",
    )
    parser.add_argument(
        "--overwrite_textgrid",
        action="store_true",
        help="Overwrite textgrid ensemble file if it exists",
    )
    parser.add_argument(
        "--overwrite_transcriptions",
        action="store_true",
        help="Overwrite transcriptions file if it exists",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for random frame sampling",
    )
    args = parser.parse_args()
    librispeech_split = args.librispeech_split
    modelname = args.modelname
    overwrite = args.overwrite
    seq_sampling = args.seq_sampling
    do_base = args.do_base
    do_transformer = args.do_transformer
    overwrite_base = args.overwrite_base
    overwrite_textgrid = args.overwrite_textgrid
    overwrite_transcriptions = args.overwrite_transcriptions
    n_frames = args.n_frames
    random_seed = args.random_seed

    logger.info("Preprocessing settings are as follows:")
    logger.info(f"  Librispeech Split: {librispeech_split}")
    logger.info(f"  Modelname: {modelname}")
    logger.info(f"  Overwrite transformers: {overwrite}")
    logger.info(f"  Sequence Sampling: {seq_sampling}")
    logger.info(f"  Do Base: {do_base}")
    logger.info(f"  Do Transformer: {do_transformer}")
    logger.info(f"  Overwrite Base: {overwrite_base}")
    logger.info(f"  Overwrite TextGrid: {overwrite_textgrid}")
    logger.info(f"  Overwrite Transcriptions: {overwrite_transcriptions}")
    logger.info(f"  Random Seed: {random_seed}")
    extract_features(
        librispeech_split=librispeech_split,
        modelname=modelname,
        overwrite=overwrite,
        seq_sampling=seq_sampling,
        n_frames=n_frames,
        random_seed=random_seed,
        do_base=do_base,
        do_transformer=do_transformer,
        overwrite_base=overwrite_base,
        overwrite_textgrid=overwrite_textgrid,
        overwrite_transcriptions=overwrite_transcriptions,
    )
