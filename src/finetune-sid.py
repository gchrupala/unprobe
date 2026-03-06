import glob
import os
from typing import Union

import evaluate
import numpy as np
import torch
import torchaudio
from datasets import Audio, ClassLabel, Dataset, load_dataset
from torchaudio.models import Wav2Vec2Model
from tqdm.auto import tqdm
from transformers import (
    AutoModelForAudioClassification,
    AutoProcessor,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from transformers.trainer_utils import get_last_checkpoint

from preprocessing import process_dataset
from utils import PROJECT_ROOT


def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = (
        eval_pred  # eval_pred is the tuple of predictions and labels returned by the model
    )
    predictions = np.argmax(logits, axis=-1)

    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["recall"]
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]

    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def inference_(
    modelname="superb/wav2vec2-base-superb-sid",
    dataset: Union[Dataset, None] = None,
    num_samples: int = 1000,
) -> dict:
    try:
        processor = AutoProcessor.from_pretrained(modelname)
    except OSError:
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

    model = Wav2Vec2Model.from_pretrained(modelname)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    if dataset is None:
        dataset, transcriptions = process_dataset("train-clean-100")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.cast_column(
            "speakerid", ClassLabel(names=sorted(set(dataset["speakerid"])))
        )
        dataset = dataset.rename_column("speakerid", "label")

    preds, actuals = [], []
    for example in tqdm(dataset.shuffle(seed=42).select(range(num_samples))):
        audio = example["audio"]["array"]
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, return_hidden_states=True)
        hidden_states = outputs.hidden_states
        # stack the hidd

    # Calculate the accuracy
    accuracy = np.mean(np.array(preds) == np.array(actuals))

    return {"modelname": modelname, "accuracy": accuracy}


def test_decodability_speakerid(
    librispeech_split: str = "train-clean-100", num_samples: int = 1000
):
    dataset, transcriptions = process_dataset(librispeech_split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column(
        "speakerid", ClassLabel(names=sorted(set(dataset["speakerid"])))
    )
    dataset = dataset.rename_column("speakerid", "label")

    modelnames = [
        "superb/wav2vec2-base-superb-sid",
        "facebook/wav2vec2-base",
        "finetuned_models/wav2vec2-sid-finetuned",
    ]
    results = []
    for modelname in modelnames:
        if "finetuned_models" in modelname:
            modelname = os.path.join(
                PROJECT_ROOT, "finetuned_models/wav2vec2-sid-finetuned/checkpoint-1300"
            )
        result = inference_(
            modelname=modelname, dataset=dataset, num_samples=num_samples
        )
        results.append(result)

    # Write the results to a file in the `results` directory
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "speakerid_decodability_results.txt")
    with open(results_file, "w") as f:
        for result in results:
            f.write(f"{result['modelname']}: {result['accuracy']:.4f}\n")


def test_superb_model():
    dataset, transcriptions = process_dataset("train-clean-100")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column(
        "speakerid", ClassLabel(names=sorted(set(dataset["speakerid"])))
    )
    dataset = dataset.rename_column("speakerid", "label")

    # Split the dataset into train and test sets at 80:20 ratio and stratify by speaker ID
    train_test_split = dataset.train_test_split(
        test_size=0.2, stratify_by_column="label"
    )
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    modelname = "superb/wav2vec2-base-superb-sid"
    modelname = os.path.join(
        PROJECT_ROOT, "finetuned_models/wav2vec2-sid-finetuned/checkpoint-1300"
    )
    processor = AutoProcessor.from_pretrained(modelname)
    model = AutoModelForAudioClassification.from_pretrained(modelname)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    preds, actuals = [], []

    for example in tqdm(test_dataset.shuffle(seed=42)):
        audio = example["audio"]["array"]
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()
        # print(f"Predicted label: {predicted_label}, Actual label: {example['label']}")
        preds.append(predicted_label)
        actuals.append(example["label"])

    # Calculate the accuracy
    accuracy = np.mean(np.array(preds) == np.array(actuals))
    print(f"Accuracy: {accuracy:.4f}")


def finetuning():
    """Use the dataset we're loading to fine-tune the wav2vec2 model for speaker ID"""

    output_dir = os.path.join(PROJECT_ROOT, "finetuned_models/wav2vec2-sid-finetuned")
    batch_size = 64
    seed = 42
    num_epochs = 10
    overwrite_output_dir = False
    no_cuda = False

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    dataset, transcriptions = process_dataset("train-clean-100")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column(
        "speakerid", ClassLabel(names=sorted(set(dataset["speakerid"])))
    )
    dataset = dataset.rename_column("speakerid", "label")

    # Split the dataset into train and test sets at 80:20 ratio and stratify by speaker ID
    train_test_split = dataset.train_test_split(
        test_size=0.2, stratify_by_column="label"
    )
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    label2id = {label: i for i, label in enumerate(dataset.features["label"].names)}
    id2label = {i: label for label, i in label2id.items()}

    modelname = "facebook/wav2vec2-base"
    processor = AutoProcessor.from_pretrained(modelname)
    hardcoded_sampling_rate = processor.feature_extractor.sampling_rate

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        modelname, num_labels=len(label2id), label2id=label2id, id2label=id2label
    )

    # Freeze the feature extractor layers to prevent them from being updated during training
    model.freeze_feature_encoder()

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=hardcoded_sampling_rate,
            max_length=hardcoded_sampling_rate * 15,
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
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        # gradient_accumulation_steps=4,
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

    # Detecting last checkpoint.
    last_checkpoint = None
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
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    data_collator = DataCollatorWithPadding(processor, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


if __name__ == "__main__":
    test_decodability_speakerid()
