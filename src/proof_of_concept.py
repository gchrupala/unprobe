import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset, load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, pairwise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.auto import tqdm

DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")


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


class FeatureEncoder:
    def __init__(
        self,
        text_modelname="answerdotai/ModernBERT-base",
        speech_modelname="facebook/wav2vec2-base",
        spacy_modelname="en_core_web_sm",
    ):
        """Load models for text and speech embedding and featurizer

        Args:
            text_modelname (str, optional): _description_. Defaults to "answerdotai/ModernBERT-base".
            speech_modelname (str, optional): _description_. Defaults to "facebook/wav2vec2-base".
            spacy_modelname (str, optional): _description_. Defaults to "en_core_web_sm".
        """
        import spacy
        from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

        self.text_modelname = text_modelname
        self.speech_modelname = speech_modelname
        self.spacy_modelname = spacy_modelname
        self.nlp = spacy.load(spacy_modelname)
        self.tokenizer = AutoTokenizer.from_pretrained(text_modelname)
        self.text_model = AutoModel.from_pretrained(
            text_modelname, reference_compile=False
        )
        self.speech_model = AutoModel.from_pretrained(speech_modelname)
        self.speech_model_fe = AutoFeatureExtractor.from_pretrained(speech_modelname)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model.to(self.device)
        self.speech_model.to(self.device)
        self.speech_model.eval()
        self.text_model.eval()

    def embed_bert(self, example):
        """Get text sentence embedding"""
        text = example["sent"]
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # output shape is (batch_size, seq_len, hidden_size)
        # return [CLS] embedding and convert to numpy
        return outputs.last_hidden_state[0, 0].cpu().numpy()

    def embed_wav2vec2(self, example):
        """Get audio sentence embedding"""
        waveform, sampling_rate = torchaudio.load(example["audio"])
        # Resample to 16kHz
        waveform = torchaudio.functional.resample(
            waveform, sampling_rate, 16_000
        ).squeeze()
        inputs = self.speech_model_fe(
            waveform, sampling_rate=16_000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.speech_model(**inputs)
            # output shape is (batch_size, seq_len, hidden_size)
        # Mean pool the last hidden states on the seq_len dimension
        # Then convert to numpy
        return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

    def get_pos_and_dep_features(self, example):
        """Extract POS and dependency features using spaCy"""
        doc = self.nlp(example["sent"])
        features = []
        for token in doc:
            # One-hot encode POS and dependency features
            # features.extend([token.pos_, token.dep_])
            features.extend([token.pos_])
        return features


def onehot_encode(X):
    # Flatten X and turn into set
    set_of_unique_features = set([item for sublist in X for item in sublist])
    # Create one-hot encoding of linguistic features
    feature_to_idx = {feature: i for i, feature in enumerate(set_of_unique_features)}

    onehot_X = []
    for example in X:
        one_hot = np.zeros(len(set_of_unique_features))
        for feature in example:
            one_hot[feature_to_idx[feature]] = 1
        onehot_X.append(one_hot)
    return onehot_X, feature_to_idx


def process_data():
    processed_data = {
        "speech_model": {
            "filename": "speech_embeds.pt",
        },
        "text_model": {
            "filename": "text_embeds.pt",
        },
        "ling_features": {
            "filename": "ling_features.pt",
        },
    }

    for key in processed_data:
        if os.path.exists(processed_data[key]["filename"]):
            print(f"Loading {key} embeddings")
            try:
                processed_data[key]["data"] = torch.load(
                    processed_data[key]["filename"], weights_only=False
                )
            except Exception as e:
                print(f"Error loading {key} embeddings: {e}")
                processed_data[key]["data"] = None
        else:
            processed_data[key]["data"] = None

    encode = FeatureEncoder(
        text_modelname="answerdotai/ModernBERT-base",
        speech_modelname="facebook/wav2vec2-base",
        spacy_modelname="en_core_web_sm",
    )

    # Load dataset
    dataset = load_librispeech("dev-clean")
    # Prepare dataset
    ling_features = []  # Linguistic features
    text_embeds = []
    speech_embeds = []

    # Encode examples in the dataset
    for example in tqdm(dataset):
        # Only process if processed data is not already present
        if processed_data["ling_features"]["data"] is None:
            ling_features.append(encode.get_pos_and_dep_features(example))
        if processed_data["text_model"]["data"] is None:
            text_embeds.append(encode.embed_bert(example))
        if processed_data["speech_model"]["data"] is None:
            speech_embeds.append(encode.embed_wav2vec2(example))

    if processed_data["ling_features"]["data"] is None:
        ling_features, feature_to_idx = onehot_encode(ling_features)

    # Update processed_data with encoded data if not already present
    processed_data["ling_features"]["data"] = (
        (np.stack(ling_features), feature_to_idx)
        if processed_data["ling_features"]["data"] is None
        else processed_data["ling_features"]["data"]
    )
    processed_data["text_model"]["data"] = (
        np.stack(text_embeds)
        if processed_data["text_model"]["data"] is None
        else processed_data["text_model"]["data"]
    )
    processed_data["speech_model"]["data"] = (
        np.stack(speech_embeds)
        if processed_data["speech_model"]["data"] is None
        else processed_data["speech_model"]["data"]
    )

    # Save processed_data to disk
    for key in processed_data:
        if "feature_to_idx" in list(processed_data[key].keys()):
            torch.save(
                (processed_data[key]["data"], processed_data[key]["feature_to_idx"]),
                processed_data[key]["filename"],
            )
        else:
            torch.save(processed_data[key]["data"], processed_data[key]["filename"])

    return processed_data


def train_probe():
    processed_data = process_data()
    (X, feature_to_idx) = processed_data["ling_features"]["data"]
    text_embeds = processed_data["text_model"]["data"]
    speech_embeds = processed_data["speech_model"]["data"]

    # Setup probe
    X = np.array(X)
    text_embeds = np.array(text_embeds)
    speech_embeds = np.array(speech_embeds)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for modality in ["text", "speech"]:
        if modality == "text":
            y = text_embeds
        elif modality == "speech":
            y = speech_embeds
        else:
            raise ValueError(f"Invalid modality: {modality}")

        # Minmax scale embeddings
        y = scaler.fit_transform(y)

        print(f"Modality: {modality}")
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(X, y, test_size=0.2)
        print("Training probe...")
        # Train probe (linear regression)
        probe = LinearRegression()
        probe.fit(X_train, y_train)

        # Evaluate
        preds = probe.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        cos_sim = np.mean(pairwise.cosine_similarity(y_test, preds))
        print(f"Modality: {modality}")
        print(f"MSE: {mse:.4f}")
        print(f"Average Cosine Similarity: {cos_sim:.4f}")

        # Use representational similarity analysis (RSA) to compare the similarity of the embeddings
        rsa = pairwise.cosine_similarity(y_test)
        rsa_preds = pairwise.cosine_similarity(preds)
        rsa_mse = mean_squared_error(rsa, rsa_preds)
        print(f"RSA MSE: {rsa_mse:.4f}")

        # Visualize the probe weights and relabell the y axis with the feature names
        plt.figure(figsize=(20, 10))
        plt.imshow(probe.coef_.T, aspect="auto")
        plt.colorbar()
        plt.yticks(range(len(feature_to_idx)), list(feature_to_idx.keys()), rotation=45)
        plt.title("Probe Weights")
        plt.show()

        plot(y_test, preds)


def plot(y_test, preds):
    # Visualization of embedding space

    # Scale embeddings
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_test = scaler.fit_transform(y_test)
    preds = scaler.transform(preds)

    combined_embeddings = np.concatenate([y_test, preds], axis=0)

    # # Perform t-SNE to reduce to 2D
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # reduced_embeddings = tsne.fit_transform(combined_embeddings)

    # # Perform PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # Split reduced embeddings back into the two sets
    reduced_emb1, reduced_emb2 = np.vsplit(reduced_embeddings, 2)

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_emb1[:, 0],
        reduced_emb1[:, 1],
        c="blue",
        label="Real Embedding",
        alpha=0.6,
    )
    plt.scatter(
        reduced_emb2[:, 0],
        reduced_emb2[:, 1],
        c="red",
        label="Probe prediction",
        alpha=0.6,
    )

    # Optional: Add lines showing the shift between paired points
    for i in range(reduced_emb2.shape[0]):
        plt.plot(
            [reduced_emb1[i, 0], reduced_emb2[i, 0]],
            [reduced_emb1[i, 1], reduced_emb2[i, 1]],
            "k-",
            alpha=0.2,
        )

    plt.legend()
    plt.title("Visualization of Two Embedding Spaces")
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    plt.grid(True)
    plt.show()
