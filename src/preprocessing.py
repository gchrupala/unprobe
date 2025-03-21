import glob
import os
import pickle

import numpy as np
import opensmile
import pandas as pd
import spacy
import textgrids
import torch
import torchaudio
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


def load_librispeech_tg(split="dev-clean"):
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

    with open(
        f"{PROJECT_ROOT}/data/librispeech_{split}_transcriptions.pickle", "wb"
    ) as f:
        pickle.dump(transcriptions, f)


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

    for audio_file in tqdm(dataset["audio"]):
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = waveform.squeeze()
        # resample waveform
        if sample_rate != feature_extractor.sampling_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=feature_extractor.sampling_rate
            )(waveform)
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


def extract_all_features():
    load_librispeech_tg("dev-clean")
    dataset = load_librispeech("dev-clean")
    savepath = f"{PROJECT_ROOT}/data"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    probe_data_types = {
        "audio_representation": {
            "function": extract_audio_representation,
            "save_dir": f"{savepath}/audio_representation.pickle",
            "overwrite": False,
            "device": "cuda",
            "model": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base"),
            "feature_extractor": Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            ),
        },
        "text_representation": {
            "function": extract_text_representation,
            "save_dir": f"{savepath}/text_representation.pickle",
            "overwrite": False,
            "device": "cuda",
            "model": AutoModel.from_pretrained(
                "answerdotai/ModernBERT-base", reference_compile=False
            ),
            "tokenizer": AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"),
        },
        "opensmile_features": {
            "function": extract_opensmile_features,
            "save_dir": f"{savepath}/opensmile_features.pickle",
            "overwrite": False,
            "feature_set": "eGeMAPSv02",
        },
    }

    for probe_data_type in tqdm(probe_data_types):
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
    extract_all_features()
