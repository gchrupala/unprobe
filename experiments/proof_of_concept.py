import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import opensmile
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from comparator import RepresentationComparator

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


class FeatureEncoder:
    def __init__(
        self,
        text_modelname="answerdotai/ModernBERT-base",
        audio_modelname="facebook/wav2vec2-base",
        spacy_modelname="en_core_web_sm",
    ):
        """Load models for text and speech embedding and featurizer

        Args:
            text_modelname (str, optional): _description_. Defaults to "answerdotai/ModernBERT-base".
            audio_modelname (str, optional): _description_. Defaults to "facebook/wav2vec2-base".
            spacy_modelname (str, optional): _description_. Defaults to "en_core_web_sm".
        """
        import spacy
        from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

        self.text_modelname = text_modelname
        self.audio_modelname = audio_modelname
        self.spacy_modelname = spacy_modelname
        self.nlp = spacy.load(spacy_modelname)
        self.tokenizer = AutoTokenizer.from_pretrained(text_modelname)
        self.text_model = AutoModel.from_pretrained(
            text_modelname, reference_compile=False
        )
        self.audio_model = AutoModel.from_pretrained(audio_modelname)
        self.audio_model_fe = AutoFeatureExtractor.from_pretrained(audio_modelname)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model.to(self.device)
        self.audio_model.to(self.device)
        self.audio_model.eval()
        self.text_model.eval()
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.smile_featurenames = self.smile.feature_names

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
        inputs = self.audio_model_fe(
            waveform, sampling_rate=16_000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            # output shape is (batch_size, seq_len, hidden_size)
        # Mean pool the last hidden states on the seq_len dimension
        # Then convert to numpy
        return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

    def get_pos_and_dep_features(self, example):
        """Extract POS and dependency features using spaCy"""
        doc = self.nlp(example["sent"])
        features = []
        for token in doc:
            # features.extend([token.pos_, token.dep_])
            features.extend([token.pos_])
        return features

    def featurize_audio(self, example):
        waveform, sampling_rate = torchaudio.load(example["audio"])
        audio_features = self.smile.process_signal(waveform, sampling_rate)

        return audio_features.to_numpy().squeeze()


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
        "audio_model": {
            "filename": f"{PROJECT_ROOT}/data/audio_embeds.pt",
        },
        "text_model": {
            "filename": f"{PROJECT_ROOT}/data/text_embeds.pt",
        },
        "text_features": {
            "filename": f"{PROJECT_ROOT}/data/text_features.pt",
        },
        "audio_features": {
            "filename": f"{PROJECT_ROOT}/data/audio_features.pt",
        },
    }
    os.makedirs(f"{PROJECT_ROOT}/data", exist_ok=True)

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
        audio_modelname="facebook/wav2vec2-base",
        spacy_modelname="en_core_web_sm",
    )

    # Load dataset
    dataset = load_librispeech("dev-clean")
    # Prepare dataset
    text_features = []  # Linguistic features
    text_embeds = []
    audio_embeds = []
    audio_features = []

    # Encode examples in the dataset
    for example in tqdm(dataset):
        # Only process if processed data is not already present
        if processed_data["text_features"]["data"] is None:
            text_features.append(encode.get_pos_and_dep_features(example))
        if processed_data["text_model"]["data"] is None:
            text_embeds.append(encode.embed_bert(example))
        if processed_data["audio_model"]["data"] is None:
            audio_embeds.append(encode.embed_wav2vec2(example))
        if processed_data["audio_features"]["data"] is None:
            audio_features.append(encode.featurize_audio(example))

    if processed_data["text_features"]["data"] is None:
        text_features, feature_to_idx = onehot_encode(text_features)

    # Update processed_data with encoded data if not already present
    processed_data["text_features"]["data"] = (
        (np.stack(text_features), feature_to_idx)
        if processed_data["text_features"]["data"] is None
        else processed_data["text_features"]["data"]
    )
    processed_data["text_model"]["data"] = (
        np.stack(text_embeds)
        if processed_data["text_model"]["data"] is None
        else processed_data["text_model"]["data"]
    )
    processed_data["audio_model"]["data"] = (
        np.stack(audio_embeds)
        if processed_data["audio_model"]["data"] is None
        else processed_data["audio_model"]["data"]
    )
    processed_data["audio_features"]["data"] = (
        np.stack(audio_features)
        if processed_data["audio_features"]["data"] is None
        else processed_data["audio_features"]["data"]
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


def choose_probe(probe_name="linear"):
    if probe_name == "linear":
        return LinearRegression()
    elif probe_name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
    elif probe_name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge()
    elif probe_name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso()


def train_probe(processed_data, scale_outputs=False):
    (text_features, feature_to_idx) = processed_data["text_features"]["data"]
    text_embeds = processed_data["text_model"]["data"]
    audio_embeds = processed_data["audio_model"]["data"]
    audio_features = processed_data["audio_features"]["data"]

    # Setup probe
    text_features = np.array(text_features)
    text_embeds = np.array(text_embeds)
    audio_embeds = np.array(audio_embeds)
    audio_features = np.array(audio_features)

    x_y_lookup = {
        "input": {"text": text_features, "speech": audio_features},
        "target": {"text": text_embeds, "speech": audio_embeds},
    }

    modalities = ["text", "speech"]
    probe_names = ["linear", "mlp", "ridge"]

    all_metrics = []

    for input_modality in modalities:
        for target_modality in modalities:
            for probe_name in probe_names:
                X = x_y_lookup["input"][input_modality]
                y = x_y_lookup["target"][target_modality]
                if scale_outputs:
                    # Minmax scale embeddings between -1 and 1
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    y = scaler.fit_transform(y)

                (X_train, X_test, y_train, y_test) = train_test_split(
                    X, y, test_size=0.2
                )
                print(
                    f"Training probe for {input_modality} -> {target_modality} with {probe_name}..."
                )
                # Train probe
                # probe = LinearRegression()
                probe = choose_probe(probe_name)
                probe.fit(X_train, y_train)

                # Evaluate
                preds = probe.predict(X_test)
                # Use r-squared to measure the probe performance

                r2 = probe.score(X_test, y_test)
                mse = mean_squared_error(y_test, preds)

                metrics = {}

                # More advanced comparisons between predicted and true embeddings
                rep_comparator = RepresentationComparator(y_test, preds)
                plotname = f"{PROJECT_ROOT}/plots/{input_modality}_{target_modality}_{probe_name}.png"
                os.makedirs(os.path.dirname(plotname), exist_ok=True)
                # metrics.update(rep_comparator.compare())
                # metrics.update(rep_comparator.full_analysis(save_plot=plotname, add_lines=True))

                metrics["input_modality"] = input_modality
                metrics["target_modality"] = target_modality
                metrics["probe_name"] = probe_name
                metrics["r2"] = r2
                metrics["mse"] = mse
                all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    return df


# TODO: use feature ablation to determine which parts of the input are most important for the probe
# Perhaps ablating groups of features at the same time to reduce the number of experiments


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

    # # Perform PCA to reduce to 3D
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    # Split reduced embeddings back into the two sets
    reduced_emb1, reduced_emb2 = np.vsplit(reduced_embeddings, 2)

    import plotly.graph_objects as go

    # Create interactive 3D scatter plot using Plotly
    fig = go.Figure()

    # Plot Real Embedding points
    fig.add_trace(
        go.Scatter3d(
            x=reduced_emb1[:, 0],
            y=reduced_emb1[:, 1],
            z=reduced_emb1[:, 2],
            mode="markers",
            marker=dict(color="blue"),
            name="Real Embedding",
        )
    )

    # Plot Probe prediction points
    fig.add_trace(
        go.Scatter3d(
            x=reduced_emb2[:, 0],
            y=reduced_emb2[:, 1],
            z=reduced_emb2[:, 2],
            mode="markers",
            marker=dict(color="red"),
            name="Probe prediction",
        )
    )

    # Optional: Add lines showing the shift between paired points
    for i in range(reduced_emb2.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=[reduced_emb1[i, 0], reduced_emb2[i, 0]],
                y=[reduced_emb1[i, 1], reduced_emb2[i, 1]],
                z=[reduced_emb1[i, 2], reduced_emb2[i, 2]],
                mode="lines",
                line=dict(color="black", width=2),
                opacity=0.2,
                showlegend=False,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="Reduced Dimension 1",
            yaxis_title="Reduced Dimension 2",
            zaxis_title="Reduced Dimension 3",
        ),
        title="Visualization of Two Embedding Spaces",
    )
    fig.update_traces(marker_size=2)
    fig.show()

    # # Plot the t-SNE visualization
    # plt.figure(figsize=(8, 6))
    # plt.scatter(
    #     reduced_emb1[:, 0],
    #     reduced_emb1[:, 1],
    #     c="blue",
    #     label="Real Embedding",
    #     alpha=0.6,
    # )
    # plt.scatter(
    #     reduced_emb2[:, 0],
    #     reduced_emb2[:, 1],
    #     c="red",
    #     label="Probe prediction",
    #     alpha=0.6,
    # )

    # # Optional: Add lines showing the shift between paired points
    # for i in range(reduced_emb2.shape[0]):
    #     plt.plot(
    #         [reduced_emb1[i, 0], reduced_emb2[i, 0]],
    #         [reduced_emb1[i, 1], reduced_emb2[i, 1]],
    #         "k-",
    #         alpha=0.2,
    #     )

    # plt.legend()
    # plt.title("Visualization of Two Embedding Spaces")
    # plt.xlabel("Reduced Dimension 1")
    # plt.ylabel("Reduced Dimension 2")
    # plt.grid(True)
    # plt.show()


def plot_probe_weights(probe, y_tick_labels=None):
    # Visualize the probe weights and relabell the y axis with the feature names
    plt.figure(figsize=(20, 10))
    plt.imshow(probe.coef_.T, aspect="auto")
    plt.colorbar()
    if y_tick_labels:
        plt.yticks(range(len(y_tick_labels)), y_tick_labels, rotation=45)
    plt.title("Probe Weights")
    plt.show()


def main():
    processed_data = process_data()
    df = train_probe(processed_data)
    df.to_csv(f"{PROJECT_ROOT}/probe_results.csv")
    print(df)
    # plot(y_test, preds)
    # plot_probe_weights(probe, feature_to_idx)


if __name__ == "__main__":
    main()
