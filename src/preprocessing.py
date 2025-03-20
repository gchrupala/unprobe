import glob
import os

import numpy as np
import opensmile
import pandas as pd
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

    df["audio"] = df["fileID"].map(get_wav_file)

    dataset = Dataset.from_pandas(df)

    return dataset


def extract_opensmile_features(dataset, feature_set="ComParE_2016"):
    """Extracting opensmile features from audio file

    Args:
        dataset (datasets.Dataset): dataset containing audio
        feature_set (str, optional): opensmile feature set to use. Defaults to "ComParE_2016".

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


def extract_audio_representation(dataset, model, feature_extractor, device="cuda"):
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


def extract_text_representation(dataset, model, tokenizer, device="cuda"):
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
    dataset = load_librispeech("dev-clean")
    
    savepath = f"{PROJECT_ROOT}/data"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    opensmile_features = extract_opensmile_features(dataset)
    


    opensmile_features.to_csv(f"{savepath}/opensmile_features.csv", index=False)

    speech_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    speech_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    audio_representation = extract_audio_representation(
        dataset, speech_model, speech_feature_extractor
    )
    torch.save(audio_representation, f"{savepath}/audio_representation.pt")

    text_model = AutoModel.from_pretrained(
        "answerdotai/ModernBERT-base", reference_compile=False
    )
    text_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    text_representation = extract_text_representation(
        dataset, text_model, text_tokenizer
    )
    torch.save(text_representation, f"{savepath}/text_representation.pt")

if __name__ == "__main__":
    extract_all_features()