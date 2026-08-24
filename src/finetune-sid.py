import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from utils import PROJECT_ROOT


def _label_count_summary(labels: np.ndarray) -> tuple[int, int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    n_speakers = int(unique.shape[0])
    n_samples = int(labels.shape[0])
    min_count = int(counts.min()) if counts.size > 0 else 0
    return n_speakers, n_samples, min_count


def _filter_dataset_min_samples_per_label(dataset, min_count: int = 2):
    labels = np.asarray(dataset["label"])
    unique, counts = np.unique(labels, return_counts=True)
    keep_labels = set(unique[counts >= min_count].tolist())
    keep_indices = [
        idx for idx, label in enumerate(labels) if int(label) in keep_labels
    ]
    filtered = dataset.select(keep_indices)
    dropped = int(labels.shape[0] - len(keep_indices))
    return filtered, dropped


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_score(
        y_true=labels,
        y_pred=predictions,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_true=labels,
        y_pred=predictions,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        y_true=labels,
        y_pred=predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def sanitize_modelname(modelname: str) -> str:
    return modelname.replace("/", "-")


def resolve_model_list(modelnames: list[str] | None = None) -> list[str]:
    defaults = [
        "superb/wav2vec2-base-superb-sid",
        "facebook/wav2vec2-base",
        os.path.join(
            PROJECT_ROOT, "finetuned_models/wav2vec2-sid-finetuned/checkpoint-1300"
        ),
    ]
    if modelnames is None or len(modelnames) == 0:
        return defaults
    resolved = []
    for modelname in modelnames:
        if modelname == "finetuned":
            resolved.append(defaults[-1])
        else:
            resolved.append(modelname)
    return resolved


def build_speaker_dataset(librispeech_split: str):
    from datasets import Audio, ClassLabel

    from preprocessing import process_dataset

    dataset, _ = process_dataset(librispeech_split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column(
        "speakerid", ClassLabel(names=sorted(set(dataset["speakerid"])))
    )
    dataset = dataset.rename_column("speakerid", "label")
    return dataset


def build_decoding_dataset(
    librispeech_split: str, num_samples: int | None = None, random_seed: int = 42
):
    dataset = build_speaker_dataset(librispeech_split)
    dataset, dropped_initial = _filter_dataset_min_samples_per_label(
        dataset, min_count=2
    )
    if dropped_initial > 0:
        print(
            f"[Dropped {dropped_initial} samples from low-frequency speakers before sampling."
        )

    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        labels = np.asarray(dataset["label"])
        indices = np.arange(len(dataset))
        if num_samples < len(dataset):
            chosen_indices, _ = train_test_split(
                indices,
                train_size=num_samples,
                random_state=random_seed,
                stratify=labels,
            )
            dataset = dataset.select(chosen_indices.tolist())

    dataset, dropped_after_sampling = _filter_dataset_min_samples_per_label(
        dataset, min_count=2
    )
    if dropped_after_sampling > 0:
        print(
            f"[Dropped {dropped_after_sampling} additional samples after sampling to keep >=2 per speaker."
        )

    final_labels = np.asarray(dataset["label"])
    n_speakers, n_samples_final, min_count_final = _label_count_summary(final_labels)
    if n_speakers < 2 or min_count_final < 2:
        raise ValueError(
            "Not enough speaker coverage after filtering/sampling. "
            f"n_speakers={n_speakers}, n_samples={n_samples_final}, min_per_speaker={min_count_final}."
        )
    print(
        f"[Decoding run uses {n_samples_final} samples across {n_speakers} speaker labels "
        f"(min samples per speaker: {min_count_final})."
    )

    return dataset


def extract_hidden_states_cache(
    modelname: str,
    librispeech_split: str,
    dataset,
    overwrite: bool = False,
    cache_dir: str | None = None,
) -> dict:
    import torch
    from transformers import AutoProcessor, Wav2Vec2Model

    cache_dir = cache_dir or os.path.join(
        PROJECT_ROOT, "results", "sid_hiddenstate_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"{librispeech_split}_{sanitize_modelname(modelname)}_hiddenstates.pkl",
    )

    if os.path.isfile(cache_file) and not overwrite:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        processor = AutoProcessor.from_pretrained(modelname)
    except OSError:
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained(modelname)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    pooled_hidden_states = []
    labels = []
    file_ids = []

    for example in tqdm(dataset, desc=f"Extracting hidden states: {modelname}"):
        audio = example["audio"]["array"]
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        layer_states = torch.stack(outputs.hidden_states, dim=0)
        pooled = layer_states.mean(dim=2).squeeze(1).cpu().numpy()

        pooled_hidden_states.append(pooled)
        labels.append(int(example["label"]))
        file_ids.append(example["fileID"])

    payload = {
        "modelname": modelname,
        "librispeech_split": librispeech_split,
        "hidden_states": np.asarray(pooled_hidden_states),
        "labels": np.asarray(labels),
        "file_ids": file_ids,
        "label_names": dataset.features["label"].names,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)

    return payload


def layerwise_speaker_decoding(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    modelname: str,
    librispeech_split: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape (num_samples, num_layers, hidden_size)"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states,
        labels,
        test_size=0.2,
        random_state=random_seed,
        stratify=labels,
    )

    results = []

    for layer_idx in range(hidden_states.shape[1]):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifier()),
            ]
        )
        pipeline.fit(X_train[:, layer_idx, :], y_train)
        y_pred = pipeline.predict(X_test[:, layer_idx, :])

        results.append(
            {
                "modelname": modelname,
                "librispeech_split": librispeech_split,
                "layer": layer_idx,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "f1_macro": float(
                    f1_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "n_samples": int(len(labels)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_speakers": int(np.unique(labels).shape[0]),
            }
        )

    return pd.DataFrame(results)


def test_decodability_speakerid(
    librispeech_split: str = "train-clean-100",
    num_samples: int = 1000,
    modelnames: list[str] | None = None,
    overwrite_cache: bool = False,
    random_seed: int = 42,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    modelnames = resolve_model_list(modelnames)
    all_results = []

    dataset = build_decoding_dataset(
        librispeech_split, num_samples=num_samples, random_seed=random_seed
    )

    for modelname in modelnames:
        cache = extract_hidden_states_cache(
            modelname=modelname,
            librispeech_split=librispeech_split,
            dataset=dataset,
            overwrite=overwrite_cache,
            cache_dir=cache_dir,
        )
        cache_n_speakers, cache_n_samples, _ = _label_count_summary(cache["labels"])
        print(
            f"[{modelname}] Cached decoding payload: {cache_n_samples} samples, {cache_n_speakers} speaker labels."
        )
        result_df = layerwise_speaker_decoding(
            hidden_states=cache["hidden_states"],
            labels=cache["labels"],
            modelname=modelname,
            librispeech_split=librispeech_split,
            random_seed=random_seed,
        )
        all_results.append(result_df)

    combined = pd.concat(all_results, ignore_index=True)
    save_file = os.path.join(
        PROJECT_ROOT,
        "results",
        f"speakerid_hiddenstate_decoding_{librispeech_split}.csv",
    )
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    combined.to_csv(save_file, index=False)
    return combined


def finetuning():
    from transformers import (
        AutoProcessor,
        Trainer,
        TrainingArguments,
        Wav2Vec2ForSequenceClassification,
    )
    from transformers.trainer_utils import get_last_checkpoint

    output_dir = os.path.join(PROJECT_ROOT, "finetuned_models/wav2vec2-sid-finetuned")
    batch_size = 64
    seed = 42
    num_epochs = 10
    overwrite_output_dir = False
    no_cuda = False

    os.makedirs(output_dir, exist_ok=True)

    dataset = build_speaker_dataset("train-clean-100")

    train_test_split_ = dataset.train_test_split(
        test_size=0.2, stratify_by_column="label"
    )
    train_dataset = train_test_split_["train"]
    test_dataset = train_test_split_["test"]

    label2id = {label: i for i, label in enumerate(dataset.features["label"].names)}
    id2label = {i: label for label, i in label2id.items()}

    modelname = "facebook/wav2vec2-base"
    processor = AutoProcessor.from_pretrained(modelname)
    sampling_rate = processor.feature_extractor.sampling_rate

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        modelname, num_labels=len(label2id), label2id=label2id, id2label=id2label
    )
    model.freeze_feature_encoder()

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            max_length=sampling_rate * 15,
            truncation=True,
        )
        inputs["labels"] = examples["label"]
        return inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        num_proc=8,
        remove_columns=[col for col in train_dataset.column_names if col != "label"],
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        num_proc=8,
        remove_columns=[col for col in test_dataset.column_names if col != "label"],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=seed,
        lr_scheduler_type="linear",
        data_seed=42,
        overwrite_output_dir=overwrite_output_dir,
        save_total_limit=3,
        weight_decay=0.001,
        use_cpu=no_cuda,
    )

    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["hiddenstate_decode", "finetune"],
        default="hiddenstate_decode",
    )
    parser.add_argument("--librispeech_split", type=str, default="train-clean-100")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--modelnames", nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    if args.mode == "finetune":
        finetuning()
    else:
        test_decodability_speakerid(
            librispeech_split=args.librispeech_split,
            num_samples=args.num_samples,
            modelnames=args.modelnames,
            overwrite_cache=args.overwrite_cache,
            random_seed=args.random_seed,
            cache_dir=args.cache_dir,
        )
