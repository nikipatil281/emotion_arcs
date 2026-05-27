from pathlib import Path
import re

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Story Emotion Atlas",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

SCENES_LONG_PATH = DATA_DIR / "script_scenes_long.parquet"
SCENES_META_PATH = DATA_DIR / "script_scenes_meta.parquet"
CHAR_PATH = DATA_DIR / "character_segments_with_emotions.parquet"

EMOTION_ORDER = ["anger", "fear", "joy", "love", "sadness", "surprise"]
EMOTION_COLORS = {
    "anger": "#b55233",
    "fear": "#435a78",
    "joy": "#d6a64f",
    "love": "#b86a7d",
    "sadness": "#5f6f86",
    "surprise": "#5b8d75",
}
PLOT_BG = "#f7f1e8"
PAPER_BG = "#fffaf4"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --paper: #fffaf4;
            --panel: rgba(255, 250, 244, 0.90);
            --ink: #2f241d;
            --muted: #6f5c4f;
            --line: rgba(91, 72, 58, 0.16);
            --accent: #a7663a;
            --accent-soft: rgba(167, 102, 58, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 166, 79, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(91, 141, 117, 0.14), transparent 28%),
                linear-gradient(180deg, #f4ecdf 0%, #efe6d7 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(214, 166, 79, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(91, 141, 117, 0.14), transparent 28%),
                linear-gradient(180deg, #f4ecdf 0%, #efe6d7 100%);
        }

        [data-testid="stHeader"] {
            display: none;
        }

        [data-testid="stDecoration"] {
            background: linear-gradient(90deg, #d6a64f 0%, #b86a7d 50%, #5b8d75 100%);
        }

        .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
            color: var(--ink);
        }

        h1, h2, h3 {
            font-family: "Fraunces", Georgia, serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        section[data-testid="stSidebar"] {
            background: rgba(248, 241, 232, 0.94);
            border-right: 1px solid var(--line);
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.4rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stMarkdown,
        .stCaption {
            color: var(--ink) !important;
        }

        button[data-testid="stWidgetLabelHelp"] {
            position: relative !important;
            width: 1.2rem !important;
            height: 1.2rem !important;
            min-width: 1.2rem !important;
            color: transparent !important;
        }

        button[data-testid="stWidgetLabelHelp"] svg {
            opacity: 0 !important;
        }

        button[data-testid="stWidgetLabelHelp"]::before {
            content: "";
            position: absolute;
            inset: 0;
            border: 2px solid #9a5a32;
            border-radius: 999px;
            box-sizing: border-box;
        }

        button[data-testid="stWidgetLabelHelp"]::after {
            content: "?";
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #9a5a32;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1;
        }

        div[data-testid="stTooltipContent"] {
            background: rgba(255, 249, 240, 0.98) !important;
            border: 1px solid rgba(91, 72, 58, 0.12) !important;
            color: var(--ink) !important;
        }

        div[data-testid="stTooltipContent"] * {
            color: var(--ink) !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="input"] > div,
        .stMultiSelect [data-baseweb="tag"] {
            background: rgba(255, 250, 244, 0.92) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="popover"] *,
        li[role="option"],
        ul[role="listbox"] li,
        [role="listbox"] [role="option"] {
            color: var(--ink) !important;
            background: transparent !important;
        }

        div[data-baseweb="popover"] {
            background: rgba(255, 250, 244, 0.98) !important;
            border: 1px solid var(--line) !important;
        }

        div[data-baseweb="menu"] {
            background: rgba(255, 250, 244, 0.98) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }

        div[data-baseweb="menu"] > div,
        div[data-baseweb="popover"] > div {
            background: rgba(255, 250, 244, 0.98) !important;
            color: var(--ink) !important;
        }

        li[aria-selected="true"],
        [role="option"][aria-selected="true"] {
            background: rgba(167, 102, 58, 0.12) !important;
            color: var(--ink) !important;
        }

        li[role="option"]:hover,
        [role="option"]:hover {
            background: rgba(167, 102, 58, 0.08) !important;
            color: var(--ink) !important;
        }

        .stSlider [data-baseweb="slider"] > div[data-testid="stTickBarMin"],
        .stSlider [data-baseweb="slider"] > div[data-testid="stTickBarMax"] {
            background: rgba(167, 102, 58, 0.18) !important;
        }

        .stSlider [role="slider"] {
            background: var(--accent) !important;
            border: 2px solid rgba(255, 250, 244, 0.95) !important;
        }

        .stDataFrame, .stTable {
            background: rgba(255, 250, 244, 0.9);
            border-radius: 18px;
        }

        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"],
        [data-testid="stTable"] * {
            color: var(--ink) !important;
            background-color: rgba(255, 250, 244, 0.92) !important;
        }

        [data-testid="stDataFrame"] [role="grid"],
        [data-testid="stDataFrame"] [role="row"],
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="gridcell"] {
            border-color: rgba(91, 72, 58, 0.10) !important;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,250,244,0.95), rgba(245,234,219,0.88));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 36px rgba(73, 49, 30, 0.08);
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 52rem;
            margin-top: 0.5rem;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 120px;
            box-shadow: 0 8px 24px rgba(73, 49, 30, 0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.6rem;
        }

        .metric-value {
            font-family: "Fraunces", Georgia, serif;
            font-size: 1.9rem;
            line-height: 1.1;
        }

        .metric-note {
            color: var(--muted);
            margin-top: 0.45rem;
            font-size: 0.92rem;
        }

        div[data-testid="stTabs"] {
            margin-top: 0.3rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 1px solid rgba(91, 72, 58, 0.12);
            padding: 0 0 0.2rem 0;
        }

        div[data-testid="stTabs"] button {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.2rem 0 0.65rem 0;
            color: var(--muted);
            box-shadow: none;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ink);
            border-bottom: 3px solid var(--accent);
        }

        div[data-testid="stTabs"] button:hover {
            color: var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_character_name(raw_name: str) -> str | None:
    if raw_name is None:
        return None

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


def base_layout(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="IBM Plex Sans, Avenir Next, sans-serif", color="#2f241d"),
        legend=dict(
            title=dict(font=dict(color="#2f241d")),
            font=dict(color="#2f241d", size=13),
            bgcolor="rgba(255, 250, 244, 0.90)",
            bordercolor="rgba(91, 72, 58, 0.12)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#fff8ef",
            bordercolor="rgba(91, 72, 58, 0.24)",
            font=dict(
                family="IBM Plex Sans, Avenir Next, sans-serif",
                color="#2f241d",
                size=13,
            ),
        ),
        margin=dict(l=28, r=28, t=26, b=28),
        height=height,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color="#2f241d", size=13),
            title=dict(font=dict(color="#2f241d", size=14)),
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(91, 72, 58, 0.12)",
        zeroline=False,
        color="#2f241d",
        tickfont=dict(color="#2f241d", size=13),
        title_font=dict(color="#2f241d", size=14),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(91, 72, 58, 0.12)",
        zeroline=False,
        color="#2f241d",
        tickfont=dict(color="#2f241d", size=13),
        title_font=dict(color="#2f241d", size=14),
    )
    return fig


def smooth_emotion_series(series: pd.Series, window: int) -> pd.Series:
    smoothed = series.rolling(window, center=True, min_periods=1).mean()
    if window >= 5:
        smoothed = smoothed.rolling(window, center=True, min_periods=1).mean()
    return smoothed


@st.cache_data
def load_scene_data():
    df_long = pd.read_parquet(SCENES_LONG_PATH)
    df_meta = pd.read_parquet(SCENES_META_PATH)

    df_long["scene_idx"] = df_long["scene_idx"].astype(int)
    df_meta["scene_idx"] = df_meta["scene_idx"].astype(int)

    return df_long, df_meta


@st.cache_data
def load_char_data():
    if not CHAR_PATH.exists():
        return None

    df_char = pd.read_parquet(CHAR_PATH)
    df_char["scene_idx"] = df_char["scene_idx"].astype(int)

    if "character" in df_char.columns:
        df_char["character_raw"] = df_char["character"]
        df_char["character"] = df_char["character"].apply(normalize_character_name)
        df_char = df_char[df_char["character"].notna()].copy()

    return df_char


def get_script_data(df_long, df_meta, selected_script: str):
    df_long_script = df_long[df_long["script_title"] == selected_script].copy()
    df_meta_script = df_meta[df_meta["title"] == selected_script].copy()

    if df_long_script.empty or df_meta_script.empty:
        st.error("No data found for this script. Did you run inference + scene-building?")
        st.stop()

    df_long_script = df_long_script.sort_values(["scene_idx", "emotion"])
    df_meta_script = df_meta_script.sort_values("scene_idx")
    return df_long_script, df_meta_script


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(selected_script: str, df_long_script: pd.DataFrame, df_meta_script: pd.DataFrame) -> None:
    dom_counts = df_meta_script["dominant_emotion"].value_counts(normalize=True)
    top_dom = dom_counts.index[0]
    top_dom_pct = dom_counts.iloc[0]
    unique_dom = df_meta_script["dominant_emotion"].nunique()
    avg_intensity = (
        df_long_script.groupby("emotion")["intensity"].mean().sort_values(ascending=False)
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Story Emotion Atlas</div>
            <h1 style="margin:0;">{selected_script}</h1>
            <div class="hero-copy">
                A scene-by-scene emotional reading of the script, built from a DistilBERT
                classifier and lightweight screenplay parsing heuristics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        render_metric_card("Total Scenes", str(len(df_meta_script)), "Approximate narrative beats extracted from chunked script text.")
    with col_b:
        render_metric_card("Most Frequent Mood", top_dom.title(), f"{top_dom_pct:.0%} of scenes lean most strongly this way.")
    with col_c:
        render_metric_card("Arc Diversity", f"{unique_dom}/{len(EMOTION_ORDER)}", f"Strongest average signal: {avg_intensity.index[0].title()}.")


def render_sidebar(all_scripts) -> str:
    st.sidebar.title("Navigator")
    selected_script = st.sidebar.selectbox("Choose a script", sorted(all_scripts))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Use the overview for the big picture, scene inspector for close reading, and character explorer for dialogue-driven arcs."
    )
    return selected_script


def make_emotion_summary_figure(df_long_script: pd.DataFrame) -> go.Figure:
    df_overall = df_long_script.groupby("emotion", as_index=False)["intensity"].mean()
    df_overall["emotion"] = pd.Categorical(df_overall["emotion"], EMOTION_ORDER, ordered=True)
    df_overall = df_overall.sort_values("emotion")

    fig = px.bar(
        df_overall,
        x="emotion",
        y="intensity",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        labels={"emotion": "Emotion", "intensity": "Average Intensity"},
    )
    fig.update_layout(showlegend=False)
    return base_layout(fig, height=340)


def make_timeline_figure(df_long_script: pd.DataFrame, selected_emotions, smooth_window: int) -> go.Figure:
    df_plot = df_long_script[df_long_script["emotion"].isin(selected_emotions)].copy()
    df_plot = df_plot.sort_values(["emotion", "scene_idx"]).copy()
    df_plot["smooth_intensity"] = (
        df_plot.groupby("emotion")["intensity"]
        .transform(lambda s: smooth_emotion_series(s, smooth_window))
    )

    fig = px.line(
        df_plot,
        x="scene_idx",
        y="smooth_intensity",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        labels={
            "scene_idx": "Scene Index",
            "smooth_intensity": "Smoothed Intensity",
            "emotion": "Emotion",
        },
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(hovermode="x unified", legend_title_text="")
    return base_layout(fig, height=420)


def make_heatmap_figure(df_long_script: pd.DataFrame, selected_emotions) -> go.Figure:
    df_pivot = (
        df_long_script[df_long_script["emotion"].isin(selected_emotions)]
        .pivot_table(index="scene_idx", columns="emotion", values="intensity", aggfunc="mean")
        .reindex(columns=selected_emotions)
        .sort_index()
    )

    fig = px.imshow(
        df_pivot.T,
        aspect="auto",
        color_continuous_scale=["#f8efe2", "#d8b579", "#a7663a", "#6f3f2f"],
        labels=dict(x="Scene Index", y="Emotion", color="Intensity"),
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            len=0.75,
            tickfont=dict(color="#2f241d", size=13),
            title=dict(font=dict(color="#2f241d", size=14)),
        )
    )
    return base_layout(fig, height=340)


def make_scene_bar_figure(df_scene_emotions: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df_scene_emotions,
        x="emotion",
        y="intensity",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        labels={"emotion": "Emotion", "intensity": "Intensity"},
    )
    fig.update_layout(showlegend=False)
    return base_layout(fig, height=320)


def make_dominant_strip_figure(df_meta_script: pd.DataFrame) -> go.Figure:
    dom_df = df_meta_script[["scene_idx", "dominant_emotion"]].copy()
    dom_df = dom_df.sort_values("scene_idx").reset_index(drop=True)
    dom_df["lane"] = 0

    fig = px.scatter(
        dom_df,
        x="scene_idx",
        y="lane",
        color="dominant_emotion",
        color_discrete_map=EMOTION_COLORS,
        labels={"scene_idx": "Scene Index", "dominant_emotion": "Dominant Emotion"},
    )
    fig.update_traces(marker=dict(size=14, symbol="square"))
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(legend_title_text="")
    return base_layout(fig, height=240)


def make_character_arc_figure(df_char_sel: pd.DataFrame) -> go.Figure:
    emo_cols = [f"{emotion}_prob" for emotion in EMOTION_ORDER]
    scene_agg = (
        df_char_sel.groupby("scene_idx")[emo_cols]
        .mean()
        .reset_index()
        .sort_values("scene_idx")
    )
    df_char_long = scene_agg.melt(
        id_vars="scene_idx",
        value_vars=emo_cols,
        var_name="emotion",
        value_name="intensity",
    )
    df_char_long["emotion"] = df_char_long["emotion"].str.replace("_prob", "", regex=False)

    fig = px.line(
        df_char_long,
        x="scene_idx",
        y="intensity",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        labels={"scene_idx": "Scene Index", "intensity": "Emotion Intensity"},
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(hovermode="x unified", legend_title_text="")
    return base_layout(fig, height=420)


def make_character_profile_figure(df_char_sel: pd.DataFrame) -> go.Figure:
    emo_cols = [f"{emotion}_prob" for emotion in EMOTION_ORDER]
    df_profile = df_char_sel[emo_cols].mean().reset_index()
    df_profile.columns = ["emotion", "intensity"]
    df_profile["emotion"] = df_profile["emotion"].str.replace("_prob", "", regex=False)

    fig = px.bar(
        df_profile,
        x="emotion",
        y="intensity",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
    )
    fig.update_layout(showlegend=False)
    return base_layout(fig, height=320)


def make_sankey_figure(df_meta_script: pd.DataFrame, sankey_height: int) -> go.Figure:
    dom_df = df_meta_script[["scene_idx", "dominant_emotion"]].copy()
    dom_df = dom_df.sort_values("scene_idx").reset_index(drop=True)

    sources = dom_df["dominant_emotion"].iloc[:-1].tolist()
    targets = dom_df["dominant_emotion"].iloc[1:].tolist()
    trans_df = pd.DataFrame({"source": sources, "target": targets}).value_counts().reset_index(name="count")

    labels = sorted(set(trans_df["source"]) | set(trans_df["target"]))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=12,
                    thickness=14,
                    line=dict(width=0.4, color="rgba(47, 36, 29, 0.3)"),
                    label=labels,
                    color=[EMOTION_COLORS.get(label, "#a7663a") for label in labels],
                ),
                link=dict(
                    source=[label_to_idx[s] for s in trans_df["source"]],
                    target=[label_to_idx[t] for t in trans_df["target"]],
                    value=trans_df["count"].tolist(),
                    color="rgba(167, 102, 58, 0.22)",
                ),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        font=dict(family="IBM Plex Sans, Avenir Next, sans-serif", color="#2f241d"),
        hoverlabel=dict(
            bgcolor="#fff8ef",
            bordercolor="rgba(91, 72, 58, 0.24)",
            font=dict(
                family="IBM Plex Sans, Avenir Next, sans-serif",
                color="#2f241d",
                size=13,
            ),
        ),
        margin=dict(l=20, r=20, t=20, b=70),
        height=sankey_height,
    )
    return fig


def render_overview_tab(df_long_script: pd.DataFrame, all_emotions) -> None:
    st.markdown("### Emotion Overview")

    controls_col, summary_col = st.columns([1.3, 1])
    with controls_col:
        selected_emotions = st.multiselect(
            "Emotions to display",
            all_emotions,
            default=all_emotions,
            help="These control the timeline and heatmap together.",
        )
    with summary_col:
        smooth_window = st.slider(
            "Smoothing window",
            min_value=1,
            max_value=9,
            step=2,
            value=3,
            help="Higher values show broader arc shape instead of scene-to-scene spikes.",
        )

    if not selected_emotions:
        st.info("Select at least one emotion to see the overview.")
        return

    st.plotly_chart(make_timeline_figure(df_long_script, selected_emotions, smooth_window), use_container_width=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(make_heatmap_figure(df_long_script, selected_emotions), use_container_width=True)
    with right:
        st.plotly_chart(make_emotion_summary_figure(df_long_script), use_container_width=True)


def render_scene_tab(df_long_script: pd.DataFrame, df_meta_script: pd.DataFrame) -> None:
    st.markdown("### Scene Inspector")

    min_scene = int(df_meta_script["scene_idx"].min())
    max_scene = int(df_meta_script["scene_idx"].max())
    scene_idx_selected = st.slider(
        "Select scene index",
        min_value=min_scene,
        max_value=max_scene,
        value=min_scene,
        key="scene_slider",
    )

    scene_meta = df_meta_script[df_meta_script["scene_idx"] == scene_idx_selected]
    if scene_meta.empty:
        st.warning("No metadata found for this scene index.")
        return

    scene_meta = scene_meta.iloc[0]
    df_scene_emotions = df_long_script[df_long_script["scene_idx"] == scene_idx_selected].copy()

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            f"#### Scene {scene_idx_selected} · {scene_meta['dominant_emotion'].title()} leads"
        )
        st.caption("Excerpt from the first chunk associated with this pseudo-scene.")
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 220px;">
                <div style="line-height:1.75; color:#3d3028;">{scene_meta['text_excerpt']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if df_scene_emotions.empty:
            st.info("No emotion data for this scene.")
        else:
            st.plotly_chart(make_scene_bar_figure(df_scene_emotions), use_container_width=True)

    st.markdown("#### Dominant Emotion Strip")
    st.plotly_chart(make_dominant_strip_figure(df_meta_script), use_container_width=True)


def render_character_tab(df_char: pd.DataFrame | None, selected_script: str) -> None:
    st.markdown("### Character Explorer")

    if df_char is None:
        st.info("Character-level data not found. Run `python src/infer_character_emotions.py` first.")
        return

    df_char_script = df_char[df_char["script_title"] == selected_script].copy()
    if df_char_script.empty:
        st.info("No character dialogue found for this script after speaker cleanup.")
        return

    char_counts = df_char_script["character"].value_counts()
    characters = char_counts.index.tolist()
    selected_char = st.selectbox(
        "Choose character",
        characters,
        help="Speaker labels are normalized, so `ELSA (CONT'D)` rolls up under `ELSA`.",
    )

    df_char_sel = df_char_script[df_char_script["character"] == selected_char].copy()
    avg_emo = df_char_sel[[f"{emotion}_prob" for emotion in EMOTION_ORDER]].mean().sort_values(ascending=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Dialogue Segments", str(len(df_char_sel)), "Only parsed dialogue blocks are counted here.")
    with col2:
        render_metric_card("Dominant Emotion", avg_emo.index[0].replace("_prob", "").title(), f"{avg_emo.iloc[0]:.2f} average model score.")
    with col3:
        render_metric_card("Scenes Appearing In", str(df_char_sel['scene_idx'].nunique()), "Count of pseudo-scenes where the character speaks.")

    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(make_character_arc_figure(df_char_sel), use_container_width=True)
    with right:
        st.plotly_chart(make_character_profile_figure(df_char_sel), use_container_width=True)


def render_transitions_tab(df_meta_script: pd.DataFrame) -> None:
    st.markdown("### Emotion Transitions")

    if len(df_meta_script) < 2:
        st.info("Not enough scenes to build transitions.")
        return

    sankey_height = st.slider(
        "Diagram height",
        min_value=320,
        max_value=900,
        value=520,
        step=40,
        help="Increase this if labels start to feel cramped.",
    )
    st.plotly_chart(make_sankey_figure(df_meta_script, sankey_height), use_container_width=True)


def render_script_finder_tab(df_meta: pd.DataFrame, all_emotions) -> None:
    st.markdown("### Emotion-Based Script Finder")
    st.caption("Find scripts where one emotion dominates a large share of scenes.")

    filter_emotion = st.selectbox(
        "Select an emotion to filter by",
        all_emotions,
        index=all_emotions.index("joy") if "joy" in all_emotions else 0,
        key="filter_emotion_select",
    )
    min_share = st.slider(
        f"Minimum % of scenes where {filter_emotion} dominates",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
    )

    scene_counts = (
        df_meta.groupby(["script_id", "title"])["scene_idx"]
        .nunique()
        .reset_index(name="total_scenes")
    )
    dom_counts = (
        df_meta.groupby(["script_id", "title", "dominant_emotion"])["scene_idx"]
        .nunique()
        .reset_index(name="scenes_with_emotion")
    )
    diversity = (
        df_meta.groupby(["script_id", "title"])["dominant_emotion"]
        .nunique()
        .reset_index(name="emotion_diversity")
    )

    dom_stats = dom_counts.merge(scene_counts, on=["script_id", "title"], how="left")
    dom_stats = dom_stats.merge(diversity, on=["script_id", "title"], how="left")
    dom_stats["share"] = dom_stats["scenes_with_emotion"] / dom_stats["total_scenes"]

    filtered = dom_stats[dom_stats["dominant_emotion"] == filter_emotion].copy()
    filtered = filtered[filtered["share"] * 100 >= min_share].sort_values("share", ascending=False)

    if filtered.empty:
        st.info("No scripts meet this threshold. Try lowering the minimum percentage.")
        return

    show_df = filtered.copy()
    show_df["share_%"] = (show_df["share"] * 100).round(1)
    show_df = show_df[
        ["title", "total_scenes", "scenes_with_emotion", "share_%", "emotion_diversity"]
    ].rename(
        columns={
            "title": "Script Title",
            "total_scenes": "Total Scenes",
            "scenes_with_emotion": f"Scenes w/ {filter_emotion.title()}",
            "share_%": f"{filter_emotion.title()} Dominance (%)",
            "emotion_diversity": "Dominant Emotion Diversity",
        }
    )
    st.dataframe(show_df, use_container_width=True, hide_index=True)


def main() -> None:
    apply_theme()

    df_long, df_meta = load_scene_data()
    df_char = load_char_data()

    all_scripts = df_long["script_title"].unique().tolist()
    all_emotions = [emotion for emotion in EMOTION_ORDER if emotion in df_long["emotion"].unique()]

    selected_script = render_sidebar(all_scripts)
    df_long_script, df_meta_script = get_script_data(df_long, df_meta, selected_script)

    render_header(selected_script, df_long_script, df_meta_script)

    tab_overview, tab_scene, tab_char, tab_trans, tab_filter = st.tabs(
        [
            "Overview",
            "Scene Inspector",
            "Character Explorer",
            "Transitions",
            "Script Finder",
        ]
    )

    with tab_overview:
        render_overview_tab(df_long_script, all_emotions)
    with tab_scene:
        render_scene_tab(df_long_script, df_meta_script)
    with tab_char:
        render_character_tab(df_char, selected_script)
    with tab_trans:
        render_transitions_tab(df_meta_script)
    with tab_filter:
        render_script_finder_tab(df_meta, all_emotions)


if __name__ == "__main__":
    main()
