from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------
# 1. Paths & config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = DATA_DIR

MODEL_DIR = MODELS_DIR / "emotion_distilbert"

# Inference settings
MAX_LEN = 128             # max tokens per chunk
BATCH_SIZE = 32           # adjust if memory issues
MAX_CHARS_PER_CHUNK = 400 # join short sentences into ~paragraphs

# For first test, limit number of scripts so it runs quickly
MAX_SCRIPTS = None          # set to None later for full run


# ----------------------------
# 2. Utils: sentence & chunk splitting
# ----------------------------
def split_into_sentences(text: str):
    """
    Very simple sentence splitter using punctuation.
    Not perfect for scripts, but works reasonably as a first pass.
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on ., ?, ! followed by space
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Clean and filter empty
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_sentences(sentences, max_chars=400):
    """
    Group consecutive sentences into chunks up to ~max_chars.
    This keeps context for the model while respecting max token length.
    """
    chunks = []
    current = []
    current_len = 0

    for s in sentences:
        if not s:
            continue
        if current and (current_len + len(s) + 1 > max_chars):
            chunks.append(" ".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += len(s) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ----------------------------
# 3. Load model, tokenizer, labels
# ----------------------------
def load_model_and_tokenizer():
    print(f"[INFO] Looking for model in: {MODEL_DIR}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found at {MODEL_DIR}. "
            f"Make sure you ran train_emotion_model.py successfully."
        )

    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model.to(device)
    model.eval()

    # Load label mapping
    label_map_path = MODEL_DIR / "emotion_labels.json"
    if not label_map_path.exists():
        label_map_path = DATA_DIR / "emotion_labels.json"
    print(f"[INFO] Loading label mapping from: {label_map_path}")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)

    labels = label_info["labels"]  # ordered list of label strings
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}
    id2label = {int(k): v for k, v in label_info["id2label"].items()}

    print(f"[INFO] Labels: {labels}")
    return tokenizer, model, device, labels, label2id, id2label


# ----------------------------
# 4. Inference over all scripts
# ----------------------------
def infer_on_scripts():
    print("[INFO] Starting inference on scripts...")

    scripts_path = DATA_DIR / "movie_scripts_clean.parquet"
    print(f"[INFO] Looking for scripts at: {scripts_path}")
    if not scripts_path.exists():
        raise FileNotFoundError(f"Scripts file not found at {scripts_path}")

    df_scripts = pd.read_parquet(scripts_path)
    print(f"[INFO] Loaded {len(df_scripts)} scripts total.")

    if MAX_SCRIPTS is not None:
        df_scripts = df_scripts.iloc[:MAX_SCRIPTS].copy()
        print(f"[INFO] Limiting to first {MAX_SCRIPTS} scripts for this run.")

    if df_scripts.empty:
        print("[WARN] No scripts found after limiting. Exiting.")
        return

    tokenizer, model, device, labels, label2id, id2label = load_model_and_tokenizer()

    all_rows = []
    softmax = torch.nn.Softmax(dim=-1)

    print(f"[INFO] Running inference on {len(df_scripts)} scripts...")

    for _, row in tqdm(df_scripts.iterrows(), total=len(df_scripts), desc="Scripts"):
        script_id = int(row["script_id"])
        title = str(row["title"])
        script_text = str(row["script_text"])

        # 1) Split into sentences then chunks
        sentences = split_into_sentences(script_text)
        if not sentences:
            print(f"[WARN] Script {script_id} ('{title}') has no sentences after splitting.")
            continue

        chunks = chunk_sentences(sentences, max_chars=MAX_CHARS_PER_CHUNK)
        if not chunks:
            print(f"[WARN] Script {script_id} ('{title}') has no chunks after chunking.")
            continue

        # 2) Run in batches
        chunk_indices = list(range(len(chunks)))
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            batch_idxs = chunk_indices[i : i + BATCH_SIZE]

            encodings = tokenizer(
                batch_chunks,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encodings)
                logits = outputs.logits
                probs = softmax(logits)  # shape: (batch_size, num_labels)

            probs_np = probs.cpu().numpy()
            pred_ids = probs_np.argmax(axis=-1)

            for chunk_idx, text_chunk, pred_id, prob_vec in zip(
                batch_idxs, batch_chunks, pred_ids, probs_np
            ):
                pred_label = id2label[int(pred_id)]

                row_out = {
                    "script_id": script_id,
                    "title": title,
                    "chunk_idx": int(chunk_idx),
                    "text_chunk": text_chunk,
                    "pred_label": pred_label,
                }

                # Add probs per label
                for label, p in zip(labels, prob_vec):
                    row_out[f"{label}_prob"] = float(p)

                all_rows.append(row_out)

    if not all_rows:
        print("[WARN] No chunks processed — resulting dataframe would be empty.")
        return

    df_out = pd.DataFrame(all_rows)
    print(f"[INFO] Created dataframe with {len(df_out)} chunk rows.")

    out_path = OUTPUT_DIR / "script_chunks_with_emotions.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"[INFO] Saved chunk-level emotions to {out_path}")


# ----------------------------
# 5. Entry point with error logging
# ----------------------------
if __name__ == "__main__":
    print("=== Running infer_emotions_on_scripts.py ===")
    try:
        infer_on_scripts()
    except Exception as e:
        print("[ERROR] Exception during inference:")
        import traceback
        traceback.print_exc()

