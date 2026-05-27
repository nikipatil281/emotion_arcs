from pathlib import Path
import json
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


# ----------------------------
# 1. Paths & config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = MODELS_DIR / "emotion_distilbert"

set_seed(42)


# ----------------------------
# 2. Load data & label mapping
# ----------------------------
def load_emotion_data():
    train_path = DATA_DIR / "emotions_train.parquet"
    val_path = DATA_DIR / "emotions_val.parquet"
    test_path = DATA_DIR / "emotions_test.parquet"
    labels_path = DATA_DIR / "emotion_labels.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)

    labels = label_info["labels"]
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}
    id2label = {int(k): v for k, v in label_info["id2label"].items()}

    # Map emotion strings → integer labels
    df_train["label"] = df_train["emotion"].map(label2id)
    df_val["label"] = df_val["emotion"].map(label2id)
    df_test["label"] = df_test["emotion"].map(label2id)

    # HuggingFace Datasets: keep only text + label
    train_ds = Dataset.from_pandas(df_train[["text", "label"]])
    val_ds = Dataset.from_pandas(df_val[["text", "label"]])
    test_ds = Dataset.from_pandas(df_test[["text", "label"]])

    return train_ds, val_ds, test_ds, labels, label2id, id2label


# ----------------------------
# 3. Tokenization
# ----------------------------
def tokenize_datasets(train_ds, val_ds, test_ds, tokenizer, max_length=128):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_enc = train_ds.map(tokenize_batch, batched=True)
    val_enc = val_ds.map(tokenize_batch, batched=True)
    test_enc = test_ds.map(tokenize_batch, batched=True)

    # Set format for PyTorch
    train_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_enc, val_enc, test_enc


# ----------------------------
# 4. Metrics
# ----------------------------
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# ----------------------------
# 5. Main training routine
# ----------------------------
def main():
    print("Loading data...")
    train_ds, val_ds, test_ds, labels, label2id, id2label = load_emotion_data()
    num_labels = len(labels)
    print(f"Found {num_labels} labels: {labels}")

    print(f"\nLoading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("\nTokenizing datasets...")
    train_enc, val_enc, test_enc = tokenize_datasets(train_ds, val_ds, test_ds, tokenizer)

    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",          # <── CHANGED from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=100,
        save_total_limit=2,
        push_to_hub=False,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print("Validation metrics:", eval_results)

    print("\nEvaluating on test set...")
    test_pred = trainer.predict(test_enc)
    logits = test_pred.predictions
    preds = np.argmax(logits, axis=-1)
    labels_true = test_pred.label_ids

    print("\nClassification report (test):")
    print(
        classification_report(
            labels_true,
            preds,
            target_names=labels,
            digits=4,
        )
    )

    print(f"\nSaving model and tokenizer to {OUTPUT_DIR} ...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Also save label mapping in the model folder for easy loading later
    label_map_path_src = DATA_DIR / "emotion_labels.json"
    label_map_path_dst = OUTPUT_DIR / "emotion_labels.json"
    with open(label_map_path_src, "r", encoding="utf-8") as f_src, open(
        label_map_path_dst, "w", encoding="utf-8"
    ) as f_dst:
        f_dst.write(f_src.read())

    print("\nAll done. Model ready for inference!")


if __name__ == "__main__":
    main()

