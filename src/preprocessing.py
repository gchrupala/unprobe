import argparse
import glob
import os
import pickle

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
import yaml
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

# Get the hostname of the machine running the code
hostname = os.uname().nodename
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


if "snellius" in hostname:
    # If running on Snellius, use the Snellius dataset root
    DATASETPATH = os.path.realpath("/projects/prjs1586/corpora/LibriSpeech")
    ALIGNMENTPATH = DATASETPATH.replace("LibriSpeech", "librispeech_textgrids")
    SAVEPATH = "/projects/prjs1586/experimental_data"

else:
    # If running on local machine, use the local dataset root
    DATASETPATH = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENTPATH = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENTPATH = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")

# Load syntax parsing models
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
tagger_labels = nlp.get_pipe("tagger").labels  # type: ignore
tagget_label_dict = {label: i for i, label in enumerate(tagger_labels)}
parser_labels = nlp.get_pipe("parser").labels  # type: ignore
parser_label_dict = {label: i for i, label in enumerate(parser_labels)}
# Similarly also get all the benepar labels
with open(f"{PROJECT_ROOT}/src/penn_treebank_labels.yml", "r") as f:
    benepar_labels = yaml.safe_load(f)
    benepar_labels["<unk>"] = "UNK"  # Add an unknown label
    benepar_labels["<pad>"] = "PAD"  # Add a padding label
    benepar_labels_dict = {label: i for i, label in enumerate(benepar_labels.keys())}

# Load fasttext model for word embeddings
fasttext.util.download_model("en", if_exists="ignore")  # English
ft = fasttext.load_model("cc.en.300.bin")
fasttext.util.reduce_model(ft, 100)  # Reduce to 100 dimensions


def save_librispeech_tg_to_single_file(
    librispeech_split="dev-clean", alignment_single_file_savepath=f"{PROJECT_ROOT}/data"
):
    """
    Extracting the textgrid files from the Librispeech dataset and save them to a single file for each tier.
    Args:
        librispeech_split (str, optional): split of librispeech to use. Defaults to "dev-clean".
        alignment_single_file_savepath (str, optional): path to save the single file. Defaults to f"{PROJECT_ROOT}/data".
    """
    if not os.path.exists(alignment_single_file_savepath):
        os.makedirs(alignment_single_file_savepath, exist_ok=True)

    # TODO unify the split names with the ones in the dataset
    if librispeech_split == "dev-clean":
        librispeech_split = "dev"
    elif librispeech_split == "train-clean-100":
        librispeech_split = "train"
    dataset_path = os.path.join(ALIGNMENTPATH, librispeech_split)
    transcription_files = glob.glob(f"{dataset_path}/**/*.TextGrid", recursive=True)
    # Sort transcription_files
    transcription_files = sorted(
        transcription_files,
        key=lambda x: os.path.splitext(os.path.basename(x))[0],
    )

    extracted_alignments = []
    for file_path in tqdm(transcription_files, desc="Reading textgrid files:"):
        tg = textgrids.TextGrid(file_path)
        fileid = file_path.split("/")[-1].split(".")[0]
        speakerid, chapter, utt = fileid.split("-")
        tg_dict = {}
        for tier in tg:
            tg_dict[tier] = np.array(
                [
                    (fileid, speakerid, chapter, utt, x.xmin, x.xmax, x.text)
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

        # TODO: change librispeech_split to the correct split name
        if librispeech_split == "dev":
            librispeech_split = "dev-clean"
        elif librispeech_split == "train":
            librispeech_split = "train-clean-100"

        df.to_csv(
            os.path.join(
                alignment_single_file_savepath,
                f"librispeech_{librispeech_split}_{tier}_alignment.csv",
            ),
            index=False,
            sep="\t",
        )


def process_fileid(fileid, ort_alignment, phone_alignment):
    speakerid, chapter, utt = fileid.split("-")
    utt_phones = []
    utt_words = []

    for i, word_row in ort_alignment[ort_alignment.fileID == fileid].iterrows():
        # Skip empty words, silences, and
        word = word_row["text"]
        if (
            word == "sil"
            or word == ""
            or word == "sp"
            or word == "<unk>"
            or str(word).lower() == "nan"
        ):
            continue
        utt_words.append(word)
        for phone in phone_alignment[
            (phone_alignment.fileID == fileid)
            & (word_row.start <= phone_alignment.start)
            & (phone_alignment.start <= word_row.end)
        ]["text"]:
            if phone == "sil" or phone == "sp" or phone == "" or phone == "<p:>":
                continue
            utt_phones.append(phone)
    # syntax_feats = syntax_parsing(utt_words)
    return {
        "fileid": fileid,
        "phones": utt_phones,
        "words": utt_words,
        "non_acoustic": [int(speakerid), int(chapter)],
        "phone_alignment": phone_alignment[
            (phone_alignment.fileID == fileid)
        ].reset_index(drop=True),
        "ort_alignment": ort_alignment[(ort_alignment.fileID == fileid)].reset_index(
            drop=True
        ),
        # "textgrid": tg,
        # "syntax_feats": syntax_feats,
    }


def load_librispeech_tg(librispeech_split="dev-clean", transcription_savefile=None):
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

    unique_fileids = ort_alignment["fileID"].unique().tolist()

    # Use multiprocessing to speed up the processing of fileids
    from functools import partial
    from multiprocessing import Pool

    # Create a partial function with fixed arguments
    process_func = partial(
        process_fileid, ort_alignment=ort_alignment, phone_alignment=phone_alignment
    )

    with Pool() as pool:
        transcriptions = pool.map(
            process_func, tqdm(unique_fileids, desc="Processing fileids")
        )

    for file in tqdm(transcriptions):
        file["syntax_feats"] = syntax_parsing(file["words"])

    if transcription_savefile is None:
        return transcriptions
    else:
        with open(transcription_savefile, "wb") as f:
            pickle.dump(transcriptions, f)
        return transcriptions


def load_librispeech(split="dev-clean"):
    """Loading Librispeech dataset into Huggingface Dataset format

    Args:
        split (str, optional): split of librispeech to use. Defaults to "dev-clean".

    Returns:
        datasets.Dataset: Huggingface Dataset object containing columns of [fileID, sent, audio]
    """
    dataset_path = f"{DATASETPATH}/{split}"
    transcription_files = glob.glob(f"{dataset_path}/**/*.trans.txt", recursive=True)
    fileids, sentences = [], []
    for file_path in transcription_files:
        with open(file_path, "r") as f:
            rows = f.read().splitlines()
        for row in rows:
            fileid, sentence = row.split(" ", 1)
            fileids.append(fileid)
            sentences.append(sentence)

    transcript_data = tuple(zip(fileids, sentences))

    df = pd.DataFrame(transcript_data, columns=["fileID", "sent"])

    def get_wav_file(fileid):
        spkid, chapter, utt = fileid.split("-")

        return os.path.join(dataset_path, spkid, chapter, f"{fileid}.flac")

    df["speakerid"] = df["fileID"].apply(lambda x: x.split("-")[0])
    df["chapter"] = df["fileID"].apply(lambda x: x.split("-")[1])
    df["utterance"] = df["fileID"].apply(lambda x: x.split("-")[2])

    df["audio"] = df["fileID"].map(get_wav_file)

    dataset = Dataset.from_pandas(df)
    # Sort by fileID
    dataset = dataset.sort("fileID")

    return dataset


def extract_opensmile_features(
    dataset, feature_set="eGeMAPSv02", **kwargs
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


def extract_spacy_features(dataset, spacy_modelname="en_core_web_sm", **kwargs):
    """Extracting spacy features from text"""
    nlp = spacy.load(spacy_modelname)
    text = dataset["sent"]
    spacy_features = []
    for sentence in tqdm(text):
        doc = nlp(sentence)
        spacy_features.append(doc.vector)
    return np.vstack(spacy_features)


def extract_ppgs_features(dataset, **kwargs):
    import ppgs
    from datasets import Audio

    # Cast the audio column to the right sampling rate
    print(f"Recasting audio sampling rate: {ppgs.SAMPLE_RATE}")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=ppgs.SAMPLE_RATE))
    print("Audio column recasted to correct sampling rate.")

    """Extracting ppgs features from audio file"""
    all_ppgs_features = []
    for example in tqdm(dataset):
        audio_tensor = torch.from_numpy(example["audio"]["array"]).unsqueeze(0)
        ppgs_features = ppgs.from_audio(
            audio_tensor, sample_rate=ppgs.SAMPLE_RATE, gpu=0
        )

        all_ppgs_features.append(ppgs_features.cpu().squeeze().numpy())


def extract_spk_embs(dataset, **kwargs):
    # Load speaker embedding model from pyannote
    from datasets import Audio
    from pyannote.audio import Model

    # Cast the audio column to the right sampling rate
    print("Recasting audio sampling rate: 16kHz")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("Audio column recasted to correct sampling rate.")

    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device("cpu")
    )
    spk_embd_model = Model.from_pretrained("pyannote/embedding")
    spk_embd_model.to(device)

    all_spk_embs = []
    for example in tqdm(dataset):
        audio_tensor = torch.from_numpy(example["audio"]["array"]).unsqueeze(0)
        spk_embs = spk_embd_model(audio_tensor.to(device=device, dtype=torch.float32))

        all_spk_embs.append(spk_embs.cpu().squeeze().numpy())

    return all_spk_embs


def extract_audio_representation(
    dataset, model, feature_extractor, device="cuda", **kwargs
) -> np.ndarray:
    """Extracting audio representation from audio file

    Args:
        dataset (datasets.Dataset): dataset containing audio
        model (transformers.Wav2Vec2Model): model to extract audio representation
        feature_extractor (transformers.Wav2Vec2FeatureExtractor): feature extractor to process audio file
        device (str, optional): device to run the model on. Defaults to "cuda".

    Returns:
        pd.DataFrame: DataFrame containing audio representation
    """
    model.to(device)
    model.eval()

    from datasets import Audio

    # Cast the audio column to the right sampling rate
    print(f"Recasting audio sampling rate: {feature_extractor.sampling_rate}")
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
    )
    print("Audio column recasted to correct sampling rate.")

    audio_representations = []
    for audio_file in tqdm(dataset["audio"]):
        # waveform, sample_rate = torchaudio.load(audio_file)
        waveform = audio_file["array"]
        # sample_rate = audio_file["sampling_rate"]
        waveform = waveform.squeeze()
        # # resample waveform
        # if sample_rate != feature_extractor.sampling_rate:
        #     waveform = torchaudio.transforms.Resample(
        #         orig_freq=sample_rate, new_freq=feature_extractor.sampling_rate
        #     )(waveform)
        inputs = feature_extractor(
            waveform, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # # output shape need to be (batch_size, seq_len, hidden_size)
            # # Mean pool the last hidden states on the seq_len dimension
            # # Then convert to numpy
            # audio_representations.append(
            #     outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()
            # )
            # Save all hidden states and preserve seq_len dimension with shape (batch_size, layer, seq_len, hidden_size)
            hidden_states = outputs.hidden_states
            hidden_states = torch.stack(hidden_states, dim=0)
            if kwargs.get("time_aggregation", "mean").lower() == "mean":
                # Take the mean over the seq_len dimension
                hidden_states = hidden_states.mean(dim=2)
            elif kwargs.get("time_aggregation", "mean").lower() == "none":
                pass
            else:
                raise ValueError(
                    f"Unknown time_aggregation method: {kwargs.get('time_aggregation', 'mean')}"
                )

            # Append everything to the list
            audio_representations.append(hidden_states.cpu().squeeze().numpy())
    try:
        # Stack the audio representations into a numpy array
        audio_representations = np.stack(audio_representations)
    except ValueError:
        # If the audio representations have different shapes, return a list of numpy arrays
        print(
            "Audio representations have different shapes, returning a list of numpy arrays"
        )
    return audio_representations
    # Return a list of numpy arrays of audio representations
    # return audio_representations


def extract_text_representation(
    dataset, model, tokenizer, device="cuda", **kwargs
) -> list:
    """Extracting text representation from text

    Args:
        dataset (datasets.Dataset): dataset containing text
        model (TextModel): model to extract text representation
        tokenizer (Tokenizer): tokenizer to process text
        device (str, optional): device to run the model on. Defaults to "cuda".
    """
    model.to(device)
    model.eval()

    text_representations = []

    for sentence in tqdm(dataset["sent"]):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # # output shape need to be (batch_size, seq_len, hidden_size)
            # # Take the [CLS] token representation
            # # Then convert to numpy
            # text_representations.append(
            #     outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
            # )
            # Save all hidden states and preserve seq_len dimension with shape (batch_size, layer, seq_len, hidden_size)
            hidden_states = outputs.hidden_states
            hidden_states = torch.stack(hidden_states, dim=1)
            # Append everything to the list
            text_representations.append(hidden_states.cpu().squeeze().numpy())
    # return np.vstack(text_representations)
    # Return a list of numpy arrays of text representations
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
    embedding = torch.nn.Embedding(len(unique_strings), embedding_size)
    embedding_bag = torch.nn.EmbeddingBag.from_pretrained(embedding.weight, mode="mean")
    # Turn flattened_all_strings into embeddings
    utterance_embeddings = []
    for utterance in tqdm(strings, desc=f"Extracting {level} embeddings"):
        # First indexize the strings
        indices = [unique_strings.index(y) for y in utterance]
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


def syntax_parsing(utt_words):
    """We aim to construct a syntactic feature extractor that functions similar to the openSMILE acoustic feature extractor. Using the textgrid information, we can extract the syntactic features of each token in the utterance and save the syntactic features in a similar way to the openSMILE acoustic feature extractor.

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
        utt_words (list(str)): List of strings of words in the utterance.
    """
    # nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    utt = " ".join(utt_words)

    doc = nlp(utt)
    sent = list(doc.sents)[0]

    # Convert benepar parse tree to NLTK format
    nltk_tree = nltk.Tree.fromstring(sent._.parse_string)
    # Print the parse tree
    # print(nltk_tree.pretty_print())

    # for every word in the sentence, print the word, dependency label, constituent label, depth in constituency tree, word_character_length, location in sentence,
    syntax_feats = []
    for i, word in enumerate(sent):
        # Skip contractions like 's, 're, 've, 'll, 'd, 'm
        # Hard coding for now, may need to change
        # #TODO try to map to phone alignments, that might be more accurate
        # OR re-force align with these subword tokens
        if word.text in ["'s", "'re", "'ve", "'ll", "'d", "'m", "n't"]:
            continue
        # Use the text to get the constituency label from the nltk tree
        tree_node = nltk_tree.leaf_treeposition(i)

        constituent_label = nltk_tree[tree_node[:-1]]._label
        constituent_label = benepar_labels_dict.get(constituent_label, 67)

        tree_depth = len(tree_node)
        tree_depth_norm = tree_depth / (nltk_tree.height() - 1)

        word_length = len(word.text)
        word_location_in_sentence = i + 1
        word_location_in_sentence_norm = word_location_in_sentence / len(sent)
        # Print the features
        # print(f"{word.text} - {word.pos} - {word.dep} - {constituent_label} - {depth} - {word_length} - {word_location_in_sentence_normalized:.2f}")

        # Construct word features with vectorized features
        word_features = np.array(
            [
                word.pos,
                word.dep,
                constituent_label,
                tree_depth,
                tree_depth_norm,
                word_length,
                word_location_in_sentence,
                word_location_in_sentence_norm,
            ]
        )
        syntax_feats.append(word_features)

    # Convert the list of features to a numpy array
    syntax_feats = np.array(syntax_feats)
    return syntax_feats


def extract_all_features(librispeech_split="dev-clean"):
    """_summary_

    Args:
        librispeech_split (str, optional): _description_. Defaults to "dev-clean".
    """

    savepath = SAVEPATH
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset = load_librispeech(librispeech_split)
    transcription_savefile = (
        f"{savepath}/librispeech-{librispeech_split}_transcriptions.pickle"
    )
    if os.path.exists(transcription_savefile) and not args.overwrite:
        print("Transcriptions already exist, loading from file...")
        with open(transcription_savefile, "rb") as f:
            transcriptions = pickle.load(f)
    else:
        print("Transcriptions do not exist, extracting...")
        transcriptions = load_librispeech_tg(librispeech_split, transcription_savefile)

    # Extracting string embeddings
    word_embedding_path = (
        f"{savepath}/librispeech-{librispeech_split}_words_embeddings.pickle"
    )
    phone_embedding_path = (
        f"{savepath}/librispeech-{librispeech_split}_phones_embeddings.pickle"
    )
    word_embedding_weights_path = (
        f"{savepath}/librispeech-{librispeech_split}_words_embedding_weights.pickle"
    )
    phone_embedding_weights_path = (
        f"{savepath}/librispeech-{librispeech_split}_phones_embedding_weights.pickle"
    )
    word_embedding_dict_path = (
        f"{savepath}/librispeech-{librispeech_split}_words_embedding_dict.pickle"
    )
    phone_embedding_dict_path = (
        f"{savepath}/librispeech-{librispeech_split}_phones_embedding_dict.pickle"
    )

    embedding_paths = [
        word_embedding_path,
        phone_embedding_path,
        word_embedding_weights_path,
        phone_embedding_weights_path,
        word_embedding_dict_path,
        phone_embedding_dict_path,
    ]

    # Check if the embeddings already exist
    if all(os.path.exists(path) for path in embedding_paths) and not args.overwrite:
        print("String embeddings already exist, skipping...")
    else:
        print("String embeddings do not exist, extracting...")
        # Extracting string embeddings
        word_string_embeddings = transcription_to_string_embeddings(
            transcriptions=transcriptions,
            embedding_size=100,
            level="words",
            emb_savepath=word_embedding_path,
            emb_weights_savepath=word_embedding_weights_path,
            emb_dict_savepath=word_embedding_dict_path,
        )
        phone_string_embeddings = transcription_to_string_embeddings(
            transcriptions=transcriptions,
            embedding_size=100,
            level="phones",
            emb_savepath=phone_embedding_path,
            emb_weights_savepath=phone_embedding_weights_path,
            emb_dict_savepath=phone_embedding_dict_path,
        )

    # Remove fileid without alignment
    dataset_ID = dataset["fileID"]
    transcription_ID = [x["fileid"] for x in transcriptions]
    difference = list(set(dataset_ID) - set(transcription_ID))
    dataset = dataset.filter(lambda x: x["fileID"] not in difference)

    # We then go on to extract the features
    probe_data_types = {
        "opensmile_features": {
            "function": extract_opensmile_features,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_opensmile_features.pickle",
            "overwrite": args.overwrite,
            "feature_level": "functionals",
            "feature_set": "eGeMAPSv02",
        },
        "opensmile_features_lld": {
            "function": extract_opensmile_features,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_opensmile_features_lld.pickle",
            "overwrite": args.overwrite,
            "feature_level": "lld",
            "feature_set": "eGeMAPSv02",
        },
        "audio_representation": {
            "function": extract_audio_representation,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_audio_representation_full.pickle",
            "overwrite": args.overwrite,
            "device": "cuda",
            "model": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base"),
            "feature_extractor": Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            ),
            "time_aggregation": "none",
        },
        "audio_representation_mean": {
            "function": extract_audio_representation,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_audio_representation.pickle",
            "overwrite": args.overwrite,
            "device": "cuda",
            "model": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base"),
            "feature_extractor": Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            ),
            "time_aggregation": "mean",
        },
        # "text_representation": {
        #     "function": extract_text_representation,
        #     "save_dir": f"{savepath}/librispeech-{librispeech_split}_text_representation.pickle",
        #     "overwrite": False,
        #     "device": "cuda",
        #     "model": AutoModel.from_pretrained(
        #         "answerdotai/ModernBERT-base", reference_compile=False
        #     ),
        #     "tokenizer": AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"),
        # },
    }

    for probe_data_type in tqdm(probe_data_types):
        tqdm.write(f"Extracting {probe_data_type} features...")
        feature = probe_data_types[probe_data_type]
        if not os.path.exists(feature["save_dir"]) or feature["overwrite"]:
            extracted_feature = feature["function"](dataset, **feature)
            if isinstance(extracted_feature, pd.DataFrame):
                # extracted_feature.to_csv(feature["save_dir"], index=False)
                extracted_feature.to_pickle(feature["save_dir"])
            else:
                # torch.save(extracted_feature, feature["save_dir"])
                with open(feature["save_dir"], "wb") as f:
                    pickle.dump(extracted_feature, f)
        else:
            print(f"{feature['save_dir']} already exists, skipping...")
    print("All features extracted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="dev-clean",
        help="Librispeech split to use",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features",
    )
    args = parser.parse_args()
    extract_all_features(args.librispeech_split)
