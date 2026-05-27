from pathlib import Path
import numpy as np
import pandas as pd


# ----------------------------
# 1. Paths & config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

CHUNK_EMO_PATH = DATA_DIR / "script_chunks_with_emotions.parquet"

# Target number of scenes per script (approximate)
NUM_SCENES_TARGET = 40

# Max characters to keep in scene text excerpt
SCENE_EXCERPT_CHARS = 400


# ----------------------------
# 2. Core logic
# ----------------------------
def assign_scene_indices(df_chunks_one_script: pd.DataFrame) -> pd.DataFrame:
    """
    Given all chunks for a single script (sorted by chunk_idx),
    assign a pseudo scene index to each chunk so that we get
    roughly NUM_SCENES_TARGET scenes per script.
    """
    df = df_chunks_one_script.sort_values("chunk_idx").reset_index(drop=True)
    num_chunks = len(df)

    if num_chunks == 0:
        df["scene_idx"] = []
        return df

    target_scenes = min(NUM_SCENES_TARGET, num_chunks)
    chunks_per_scene = max(1, num_chunks // target_scenes)

    scene_indices = np.arange(num_chunks) // chunks_per_scene
    scene_indices = np.clip(scene_indices, 0, target_scenes - 1)

    df["scene_idx"] = scene_indices
    return df


def build_scene_level_emotions():
    print("=== Building scene-level emotion data ===")
    print(f"[INFO] Reading chunk-level emotions from: {CHUNK_EMO_PATH}")

    if not CHUNK_EMO_PATH.exists():
        raise FileNotFoundError(
            f"Chunk-level emotion file not found at {CHUNK_EMO_PATH}. "
            f"Run infer_emotions_on_scripts.py first."
        )

    df_chunks = pd.read_parquet(CHUNK_EMO_PATH)
    print(f"[INFO] Loaded {len(df_chunks)} chunk rows.")

    # Ensure required columns exist
    required_cols = {
        "script_id",
        "title",
        "chunk_idx",
        "text_chunk",
        "pred_label",
        "anger_prob",
        "fear_prob",
        "joy_prob",
        "love_prob",
        "sadness_prob",
        "surprise_prob",
    }
    missing = required_cols - set(df_chunks.columns)
    if missing:
        raise ValueError(f"Missing required columns in chunk dataframe: {missing}")

    # Assign scene_idx within each script
    print("[INFO] Assigning scene indices per script...")
    df_with_scenes = (
        df_chunks
        .groupby(["script_id"], group_keys=False)
        .apply(assign_scene_indices)
    )

    # ------------------------
    # 2A. Build scene meta
    # ------------------------
    print("[INFO] Aggregating scene metadata and emotion intensities...")

    # Scene-level emotion intensities: average probs across chunks
    emotion_cols = ["anger_prob", "fear_prob", "joy_prob", "love_prob", "sadness_prob", "surprise_prob"]

    agg_funcs = {col: "mean" for col in emotion_cols}

    # For excerpt, we take first chunk's text in the scene and truncate
    def first_text_excerpt(texts):
        if len(texts) == 0:
            return ""
        txt = str(texts.iloc[0])
        if len(txt) > SCENE_EXCERPT_CHARS:
            txt = txt[:SCENE_EXCERPT_CHARS] + "..."
        return txt

    scene_groups = df_with_scenes.groupby(["script_id", "title", "scene_idx"])

    df_scene_emotions = scene_groups[emotion_cols].agg(agg_funcs).reset_index()

    df_scene_meta = scene_groups["text_chunk"].apply(first_text_excerpt).reset_index(name="text_excerpt")

    # Determine dominant emotion per scene
    def dominant_emotion(row):
        vals = row[emotion_cols].values
        idx = np.argmax(vals)
        return emotion_cols[idx].replace("_prob", "")

    df_scene_emotions["dominant_emotion"] = df_scene_emotions.apply(dominant_emotion, axis=1)

    # Merge meta (excerpt) into emotions dataframe
    df_scene_full = df_scene_emotions.merge(
        df_scene_meta,
        on=["script_id", "title", "scene_idx"],
        how="left",
    )

    # ------------------------
    # 2B. Save wide + long formats
    # ------------------------
    # Wide format with one row per scene
    scenes_meta_path = DATA_DIR / "script_scenes_meta.parquet"
    df_scene_full.to_parquet(scenes_meta_path, index=False)
    print(f"[INFO] Saved scene-level meta + intensities (wide) to: {scenes_meta_path}")
    print(f"[INFO] Scene rows: {len(df_scene_full)}")

    # Long format for plotting: one row per (scene, emotion)
    records = []
    for _, row in df_scene_full.iterrows():
        for emo_col in emotion_cols:
            emotion = emo_col.replace("_prob", "")
            intensity = row[emo_col]
            records.append(
                {
                    "script_id": row["script_id"],
                    "script_title": row["title"],
                    "scene_idx": row["scene_idx"],
                    "emotion": emotion,
                    "intensity": float(intensity),
                }
            )

    df_long = pd.DataFrame(records)
    scenes_long_path = DATA_DIR / "script_scenes_long.parquet"
    df_long.to_parquet(scenes_long_path, index=False)
    print(f"[INFO] Saved scene-level long format to: {scenes_long_path}")
    print(f"[INFO] Long rows: {len(df_long)}")

    print("=== Done building scene-level data ===")


# ----------------------------
# 3. Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        build_scene_level_emotions()
    except Exception as e:
        print("[ERROR] Exception while building scene emotions:")
        import traceback
        traceback.print_exc()

