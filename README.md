# Story Emotion Atlas

A small NLP project for modeling the emotional arc of movie scripts.

The project:

- fine-tunes a DistilBERT emotion classifier on a labeled emotion dataset
- applies that classifier to chunks of movie scripts
- aggregates chunk predictions into pseudo-scenes
- optionally estimates character-level emotional arcs from dialogue blocks
- visualizes the results in a Streamlit dashboard

## Project Structure

```text
.
├── app.py
├── data/
│   ├── raw/
│   │   ├── emotions/
│   │   └── scripts/
│   └── processed/
├── models/
└── src/
    ├── prepare_data.py
    ├── train_emotion_model.py
    ├── infer_emotions_on_scripts.py
    ├── build_scene_emotions.py
    └── infer_character_emotions.py
```

## What Each Script Does

- `src/prepare_data.py`
  Prepares the labeled emotion dataset and normalizes the raw movie scripts dataset.

- `src/train_emotion_model.py`
  Fine-tunes `distilbert-base-uncased` on the emotion dataset and saves the trained model.

- `src/infer_emotions_on_scripts.py`
  Splits each script into text chunks and predicts emotion probabilities per chunk.

- `src/build_scene_emotions.py`
  Groups chunk-level predictions into approximate scenes and creates scene-level outputs for plotting.

- `src/infer_character_emotions.py`
  Parses dialogue-like speaker blocks and predicts character-level emotion arcs.

- `app.py`
  Main Streamlit dashboard with overview, scene inspector, character explorer, transitions, and script finder.

## Requirements

- Python 3.11 or 3.12 recommended
- enough disk space for raw data, processed parquet files, and model checkpoints
- a working internet connection during first model download and package installation

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Expected Data Layout

Place your raw files here:

```text
data/raw/emotions/train.txt
data/raw/emotions/val.txt
data/raw/emotions/test.txt
data/raw/scripts/movie_scripts.parquet
```

Notes:

- The emotion files are expected to be in `text;emotion` format.
- `prepare_data.py` also supports `validation.txt` instead of `val.txt`.
- The scripts parquet file should contain a title-like column and a script-text-like column. The code tries to detect common names automatically.

## Data Sources

- Emotion classification dataset:
  local text files stored under `data/raw/emotions/`

- Movie scripts source:
  `https://imsdb.com/`

The raw scripts directory is intentionally ignored by Git because script exports and derived parquet files are large and can easily exceed normal GitHub-friendly repository size limits.

## End-to-End Pipeline

Run the project in this order:

### 1. Prepare datasets

```bash
python src/prepare_data.py
```

This creates:

- `data/processed/emotions_train.parquet`
- `data/processed/emotions_val.parquet`
- `data/processed/emotions_test.parquet`
- `data/processed/emotion_labels.json`
- `data/processed/movie_scripts_clean.parquet`

### 2. Train the emotion classifier

```bash
python src/train_emotion_model.py
```

This saves the fine-tuned model under:

```text
models/emotion_distilbert/
```

### 3. Run chunk-level inference on scripts

```bash
python src/infer_emotions_on_scripts.py
```

This creates:

- `data/processed/script_chunks_with_emotions.parquet`

### 4. Build scene-level emotion outputs

```bash
python src/build_scene_emotions.py
```

This creates:

- `data/processed/script_scenes_meta.parquet`
- `data/processed/script_scenes_long.parquet`

### 5. Run character-level inference

```bash
python src/infer_character_emotions.py
```

This creates:

- `data/processed/character_segments_with_emotions.parquet`

## Run the Dashboard

```bash
streamlit run app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

## Dashboard Features

`app.py` includes:

- emotion overview timeline
- scene heatmap
- scene inspector
- character explorer
- emotion transition Sankey diagram
- emotion-based script finder

## Important Notes

- Scene boundaries are heuristic, not screenplay-structure-accurate.
- Character parsing is also heuristic and may miss or merge some speaker labels depending on script formatting.
- Large processed files and model artifacts are intentionally ignored by Git via `.gitignore`.
- `data/raw/scripts/` is also ignored by Git; add your local script dataset there before running the pipeline.

## Suggested Git Workflow

After creating your Git repo:

```bash
git init
git add .
git commit -m "Initial commit"
```

Because `data/processed/`, `data/raw/scripts/`, and `models/` are ignored, the repo will stay lightweight unless you intentionally change `.gitignore`.
