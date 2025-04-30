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


def load_librispeech_tg(librispeech_split="dev-clean", transcription_savefile=None):
    """Loading Librispeech dataset into Huggingface Dataset format

    Args:
        librispeech_split (str, optional): split of librispeech to use. Defaults to "dev-clean".

    Returns:
        datasets.Dataset: Huggingface Dataset object containing columns of [fileID, sent, audio]
    """
    dataset_path = os.path.expanduser(
        f"~/corpora/librispeech_alignment/{librispeech_split}"
    )
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
            # Take the mean over the seq_len dimension
            hidden_states = hidden_states.mean(dim=2)
            # Append everything to the list
            audio_representations.append(hidden_states.cpu().squeeze().numpy())

    return np.stack(audio_representations)
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

    return utterance_embeddings


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
    if os.path.exists(transcription_savefile):
        with open(transcription_savefile, "rb") as f:
            transcriptions = pickle.load(f)
    else:
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

    # Check if the embeddings already exist
    if (
        os.path.exists(word_embedding_path)
        and os.path.exists(phone_embedding_path)
        and not args.overwrite
    ):
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
        )
        phone_string_embeddings = transcription_to_string_embeddings(
            transcriptions=transcriptions,
            embedding_size=100,
            level="phones",
            emb_savepath=phone_embedding_path,
            emb_weights_savepath=phone_embedding_weights_path,
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
            "save_dir": f"{savepath}/librispeech-{librispeech_split}_audio_representation.pickle",
            "overwrite": args.overwrite,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features",
    )
    args = parser.parse_args()
    extract_all_features(args.librispeech_split)
