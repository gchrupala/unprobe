import argparse
import glob
import os
import pickle

import numpy as np
import opensmile
import pandas as pd
import spacy
import textgrids
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def load_librispeech_tg(split="dev-clean", transcription_savefile=None):
    """Loading Librispeech dataset into Huggingface Dataset format

    Args:
        split (str, optional): split of librispeech to use. Defaults to "dev-clean".

    Returns:
        datasets.Dataset: Huggingface Dataset object containing columns of [fileID, sent, audio]
    """
    dataset_path = os.path.expanduser(f"~/corpora/librispeech_alignment/{split}")
    transcription_files = glob.glob(f"{dataset_path}/**/*.TextGrid", recursive=True)

    transcriptions = []

    for file_path in tqdm(transcription_files):
        tg = textgrids.TextGrid(file_path)

        fileid = file_path.split("/")[-1].split(".")[0]
        speakerid, chapter, utt = fileid.split("-")
        utt_phones = []
        utt_words = []
        for phone in tg["phones"]:
            if phone.text == "sil" or phone.text == "sp" or phone.text == "":
                continue
            utt_phones.append(phone.text)
        for word in tg["words"]:
            if word.text == "sil" or word.text == "" or word.text == "sp":
                continue
            utt_words.append(word.text)
        transcriptions.append(
            {
                "fileid": fileid,
                "phones": utt_phones,
                "words": utt_words,
                "non_acoustic": [int(speakerid), int(chapter)],
            }
        )

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
    dataset_path = f"{DATASET_ROOT}/{split}"
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

    return dataset


def extract_embeddings(dataset, embedding_size=100, **kwargs):
    """Extracting embedding based on the string level

    Args:
        level (str, optional): _description_. Defaults to "words".

    Returns:
        _type_: _description_
    """
    # Take into account transcriptions from all splits
    transcription_pickles = glob.glob(
        f"{PROJECT_ROOT}/data/librispeech*transcriptions.pickle", recursive=True
    )
    transcriptions = []
    for transcription_pickle in transcription_pickles:
        with open(transcription_pickles, "rb") as f:
            transcriptions.extend(pickle.load(f))

    # Check if embedding weights already exist
    level = kwargs["level"]

    embedding_weights_path = f"{PROJECT_ROOT}/data/{level}_embedding_weights.pickle"
    if os.path.exists(embedding_weights_path):
        with open(embedding_weights_path, "rb") as f:
            embedding_weights = pickle.load(f)
        embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(embedding_weights), freeze=True
        )

    all_strings = [x[level] for x in transcriptions]
    flattened_all_strings = [item for sublist in all_strings for item in sublist]
    unique_strings = list(set(flattened_all_strings))
    # Sort unique strings by alphabetical order
    unique_strings.sort()
    unique_strings = ["<pad>"] + unique_strings
    embedding = torch.nn.Embedding(len(unique_strings), embedding_size)
    emb_model = torch.nn.EmbeddingBag.from_pretrained(embedding.weight, mode="mean")

    # Indexize the strings
    indices = [[unique_strings.index(y) for y in x] for x in all_strings]

    # pad indices to same length with <pad> token
    max_len = max([len(x) for x in indices])
    indices = [x + [0] * (max_len - len(x)) for x in indices]

    utterance_embeddings = emb_model(torch.tensor(indices)).detach().numpy()

    # # Save the embeddings
    # with open(f"{PROJECT_ROOT}/data/{level}_embeddings.pickle", "wb") as f:
    #     pickle.dump(utterance_embeddings, f)
    # Save the embedding weights
    with open(f"{PROJECT_ROOT}/data/{level}_embedding_weights.pickle", "wb") as f:
        pickle.dump(embedding.weight.detach().numpy(), f)

    return utterance_embeddings


def extract_opensmile_features(dataset, feature_set="eGeMAPSv02", **kwargs):
    """Extracting opensmile features from audio file

    Args:
        dataset (datasets.Dataset): dataset containing audio
        feature_set (str, optional): opensmile feature set to use. Defaults to "eGeMAPSv02".

    Returns:
        pd.DataFrame: DataFrame containing opensmile features
    """

    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True,
        num_workers=8,
        sampling_rate=16000,
        resample=True,
    )

    files = dataset["audio"]
    opensmile_features = smile.process_files(files)

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


def extract_audio_representation(
    dataset, model, feature_extractor, device="cuda", **kwargs
):
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

    audio_representations = []
    from datasets import Audio

    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
    )

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
            outputs = model(**inputs)
            # output shape need to be (batch_size, seq_len, hidden_size)
            # Mean pool the last hidden states on the seq_len dimension
            # Then convert to numpy
            audio_representations.append(
                outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()
            )

    return np.vstack(audio_representations)


def extract_text_representation(dataset, model, tokenizer, device="cuda", **kwargs):
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
            outputs = model(**inputs)
            # output shape need to be (batch_size, seq_len, hidden_size)
            # Take the [CLS] token representation
            # Then convert to numpy
            text_representations.append(
                outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
            )

    return np.vstack(text_representations)


def extract_all_features(librispeech_split="dev-clean"):
    """_summary_

    Args:
        librispeech_split (str, optional): _description_. Defaults to "dev-clean".
    """

    savepath = f"{PROJECT_ROOT}/data"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset = load_librispeech(librispeech_split)
    transcription_savefile = (
        f"{PROJECT_ROOT}/data/librispeech-{librispeech_split}_transcriptions.pickle"
    )
    if not os.path.exists(transcription_savefile):
        transcriptions = load_librispeech_tg(librispeech_split, transcription_savefile)

    probe_data_types = {
        "opensmile_features": {
            "function": extract_opensmile_features,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_opensmile_features.pickle",
            "overwrite": False,
            "feature_set": "eGeMAPSv02",
        },
        "word_embeddings": {
            "function": extract_embeddings,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_words_embeddings.pickle",
            "overwrite": True,
            "level": "words",
        },
        "phone_embeddings": {
            "function": extract_embeddings,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_phones_embeddings.pickle",
            "overwrite": True,
            "level": "phones",
        },
        "audio_representation": {
            "function": extract_audio_representation,
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_audio_representation.pickle",
            "overwrite": False,
            "device": "cuda",
            "model": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base"),
            "feature_extractor": Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            ),
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
    args = parser.parse_args()
    extract_all_features(args.librispeech_split)
