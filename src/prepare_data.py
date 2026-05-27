from pathlib import Path
import json

import pandas as pd


# ----------------------------
# 1. Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # go up from src/
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

PROC_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 2. Prepare emotions dataset
# ----------------------------
def prepare_emotion_dataset():
    emotions_raw_dir = RAW_DIR / "emotions"

    # Handle both val.txt and validation.txt naming
    split_to_filename = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
    }

    # If val.txt doesn't exist but validation.txt does, switch
    if not (emotions_raw_dir / split_to_filename["val"]).exists():
        if (emotions_raw_dir / "validation.txt").exists():
            split_to_filename["val"] = "validation.txt"

    dfs = []
    for split, fname in split_to_filename.items():
        path = emotions_raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

        # Dataset format: text;emotion
        df = pd.read_csv(
            path,
            names=["text", "emotion"],
            sep=";",
            header=None,
        )
        df["split"] = split

        # Basic cleaning
        df["text"] = df["text"].astype(str).str.strip()
        df["emotion"] = df["emotion"].astype(str).str.strip()

        # Save per-split processed file
        out_path = PROC_DIR / f"emotions_{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {split} split to {out_path} with {len(df)} rows.")

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Create label ↔ id mapping
    labels = sorted(all_df["emotion"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    mapping = {"labels": labels, "label2id": label2id, "id2label": id2label}
    mapping_path = PROC_DIR / "emotion_labels.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved label mapping to {mapping_path}")
    print("Labels:", labels)


# ----------------------------
# 3. Prepare movie scripts
# ----------------------------
def prepare_scripts_dataset():
    scripts_raw_dir = RAW_DIR / "scripts"

    # TODO: change this filename if yours is different
    parquet_path = scripts_raw_dir / "movie_scripts.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Expected scripts file not found at {parquet_path}. "
            "Rename this in prepare_scripts_dataset() if your file has a different name."
        )

    df = pd.read_parquet(parquet_path)

    # Try to normalize column names to ['script_id', 'title', 'script_text']
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}  # map lowercase → original

    # Common possibilities
    title_col = None
    script_col = None
    for cand in ["title", "name", "movie", "movie_name"]:
        if cand in cols:
            title_col = cols[cand]
            break

    for cand in ["script", "text", "content", "body"]:
        if cand in cols:
            script_col = cols[cand]
            break

    if title_col is None or script_col is None:
        raise ValueError(
            f"Could not automatically find title/script columns in {parquet_path}. "
            f"Found columns: {list(df.columns)}"
        )

    df_out = pd.DataFrame(
        {
            "script_id": range(len(df)),
            "title": df[title_col].astype(str),
            "script_text": df[script_col].astype(str),
        }
    )

    out_path = PROC_DIR / "movie_scripts_clean.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"Saved cleaned movie scripts to {out_path} with {len(df_out)} rows.")


# ----------------------------
# 4. Run both preparations
# ----------------------------
if __name__ == "__main__":
    print("=== Preparing emotion dataset ===")
    prepare_emotion_dataset()
    print("\n=== Preparing movie scripts dataset ===")
    prepare_scripts_dataset()
    print("\nAll done.")

