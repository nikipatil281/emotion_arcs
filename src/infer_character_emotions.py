from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------
# Paths & config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

SCRIPTS_PATH = DATA_DIR / "movie_scripts_clean.parquet"
MODEL_DIR = MODELS_DIR / "emotion_distilbert"

OUTPUT_PATH = DATA_DIR / "character_segments_with_emotions.parquet"

NUM_SCENES_TARGET = 40
MAX_LEN = 128
BATCH_SIZE = 32
MAX_SCRIPTS = None   # for testing; set to None for full run


# ----------------------------
# Helper: detect speaker lines
# ----------------------------
SCENE_PREFIXES = ("INT.", "EXT.", "FADE", "CUT TO", "DISSOLVE", "SUPER:", "OMIT:")


def normalize_character_name(raw_name: str):
    name = str(raw_name).strip()
    if not name:
        return None
    if name.startswith("(") and name.endswith(")"):
        return None

    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\([^)]*\)", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\bCONT'?D\b\.?", "", name).strip(" -/")
    name = re.sub(r"\bV\.?O\.?\b", "", name).strip(" -/")
    name = re.sub(r"\bO\.?S\.?\b", "", name).strip(" -/")
    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        return None
    if len(name) > 30:
        return None
    if not re.match(r"^[A-Z0-9][A-Z0-9 .'\-]*$", name):
        return None

    throwaway = {
        "BEAT",
        "PAUSE",
        "SILENCE",
        "MOMENT",
        "CONTINUED",
        "CONTD",
        "CUT TO",
        "FADE OUT",
    }
    if name in throwaway:
        return None

    return name

def looks_like_speaker(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(s.startswith(p) for p in SCENE_PREFIXES):
        return False
    # All caps-ish, not too long
    if len(s) > 30:
        return False
    # uppercase letters, digits, spaces and punctuation like .'-()
    if not re.match(r"^[A-Z0-9 .'\-()]+$", s):
        return False
    return True


def parse_script_to_segments(script_text: str):
    """
    Parse script into a list of {character, text} dialogue segments.
    Non-dialogue narration is ignored for now.
    """
    segments = []
    if not isinstance(script_text, str):
        script_text = str(script_text)

    lines = script_text.splitlines()

    current_char = None
    buffer = []

    def flush_segment():
        nonlocal buffer, current_char
        if current_char and buffer:
            text = " ".join([b.strip() for b in buffer if b.strip()])
            if text:
                segments.append(
                    {"character": current_char.strip(), "text": text}
                )
        buffer = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            # blank line: end of dialogue block
            flush_segment()
            current_char = None
            continue

        if looks_like_speaker(line):
            # new speaker
            flush_segment()
            current_char = normalize_character_name(line)
        else:
            if current_char:
                buffer.append(line)

    # last one
    flush_segment()
    return segments


def assign_scene_indices(num_segments: int, num_scenes_target: int = NUM_SCENES_TARGET):
    if num_segments == 0:
        return []
    target_scenes = min(num_scenes_target, num_segments)
    segs_per_scene = max(1, num_segments // target_scenes)
    scene_idx = np.arange(num_segments) // segs_per_scene
    scene_idx = np.clip(scene_idx, 0, target_scenes - 1)
    return scene_idx


# ----------------------------
# Load model & labels
# ----------------------------
def load_model_and_tokenizer():
    print(f"[INFO] Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model.to(device)
    model.eval()

    label_map_path = MODEL_DIR / "emotion_labels.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)

    labels = label_info["labels"]
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}
    id2label = {int(k): v for k, v in label_info["id2label"].items()}

    return tokenizer, model, device, labels, label2id, id2label


# ----------------------------
# Main: infer character emotions
# ----------------------------
def infer_character_emotions():
    print("=== Character-level emotion inference ===")

    if not SCRIPTS_PATH.exists():
        raise FileNotFoundError(f"Scripts file not found at {SCRIPTS_PATH}")

    df_scripts = pd.read_parquet(SCRIPTS_PATH)
    print(f"[INFO] Loaded {len(df_scripts)} scripts.")

    if MAX_SCRIPTS is not None:
        df_scripts = df_scripts.iloc[:MAX_SCRIPTS].copy()
        print(f"[INFO] Limiting to first {MAX_SCRIPTS} scripts for this run.")

    tokenizer, model, device, labels, label2id, id2label = load_model_and_tokenizer()
    softmax = torch.nn.Softmax(dim=-1)

    all_rows = []

    for _, row in tqdm(df_scripts.iterrows(), total=len(df_scripts), desc="Scripts"):
        script_id = int(row["script_id"])
        title = str(row["title"])
        script_text = str(row["script_text"])

        segments = parse_script_to_segments(script_text)

        if not segments:
            print(f"[WARN] No dialogue segments found for script {script_id} ({title}).")
            continue

        # assign indices & scene_idx
        for idx, seg in enumerate(segments):
            seg["segment_idx"] = idx

        scene_idx = assign_scene_indices(len(segments))
        for seg, s_idx in zip(segments, scene_idx):
            seg["scene_idx"] = int(s_idx)

        # run in batches
        segment_texts = [s["text"] for s in segments]
        segment_indices = [s["segment_idx"] for s in segments]
        segment_chars = [s["character"] for s in segments]
        segment_scene_idx = [s["scene_idx"] for s in segments]

        for i in range(0, len(segment_texts), BATCH_SIZE):
            batch_texts = segment_texts[i : i + BATCH_SIZE]
            batch_idxs = segment_indices[i : i + BATCH_SIZE]
            batch_chars = segment_chars[i : i + BATCH_SIZE]
            batch_scene_idx = segment_scene_idx[i : i + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**enc)
                probs = softmax(outputs.logits)

            probs_np = probs.cpu().numpy()
            pred_ids = probs_np.argmax(axis=-1)

            for seg_idx, char, s_idx, text, pred_id, prob_vec in zip(
                batch_idxs, batch_chars, batch_scene_idx, batch_texts, pred_ids, probs_np
            ):
                row_out = {
                    "script_id": script_id,
                    "script_title": title,
                    "character": char,
                    "segment_idx": int(seg_idx),
                    "scene_idx": int(s_idx),
                    "text": text,
                    "pred_label": id2label[int(pred_id)],
                }
                for label, p in zip(labels, prob_vec):
                    row_out[f"{label}_prob"] = float(p)
                all_rows.append(row_out)

    if not all_rows:
        print("[WARN] No character segments processed.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved character-level emotions to: {OUTPUT_PATH}")
    print(f"[INFO] Rows: {len(df_out)}")


if __name__ == "__main__":
    try:
        infer_character_emotions()
    except Exception as e:
        print("[ERROR] Exception during character emotion inference:")
        import traceback
        traceback.print_exc()
