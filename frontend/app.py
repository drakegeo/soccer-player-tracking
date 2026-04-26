import base64
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from pipelines.config import MODELS_DIR, CLASS_NAMES, CONF_THRESHOLD
from src.model_training.inference import count_players_per_frame

def _bg_css() -> str:
    img_path = ROOT / "data" / "app" / "soccer-ball-background-display-with-analytics-statistics-graphs_1061279-993.avif"
    encoded = base64.b64encode(img_path.read_bytes()).decode()
    return f"url('data:image/avif;base64,{encoded}')"

st.set_page_config(page_title="Soccer Player Tracker", layout="centered")

_CSS = """
<style>
/* ── Background ── */
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.72), rgba(0,10,0,0.80)), BG_IMAGE;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ── */
.block-container {
    padding-top: 2.5rem;
    max-width: 780px;
}

/* ── Title ── */
h1 {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* ── Caption / body text ── */
p, .stCaption, label, .stMarkdown {
    color: #c8d8c8 !important;
}

/* ── Upload box ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #3a7a3a;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.85; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stMetricLabel"]  { color: #a0bfa0 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important; }

/* ── White team card ── */
.white-card [data-testid="stMetric"] {
    border-left: 4px solid #ffffff;
}

/* ── Red team card ── */
.red-card [data-testid="stMetric"] {
    border-left: 4px solid #e53935;
}

/* ── Section headers ── */
h2, h3 {
    color: #e8f5e8 !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(255,255,255,0.08);
    color: #c8f0c8;
    border: 1px solid #3a7a3a;
    border-radius: 8px;
    width: 100%;
    font-weight: 600;
}
.stDownloadButton > button:hover {
    background: rgba(255,255,255,0.14);
}

/* ── Success / error messages ── */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
"""

st.markdown(_CSS.replace("BG_IMAGE", _bg_css()), unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# ⚽ Soccer Player Tracker")
st.caption("Upload a match video to detect and count players per team using YOLOv8.")
st.markdown("---")

# ── Model check ─────────────────────────────────────────────────────────────
best_weights = MODELS_DIR / "soccer_player_detector" / "weights" / "best.pt"
if not best_weights.exists():
    st.error("No trained model found. Run the training pipeline first.")
    st.stop()

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Drop a video file here", type=["mp4", "avi", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    output_path = str(ROOT / "models" / "output_annotated.mp4")

    st.markdown("")
    if st.button("Run Inference", type="primary"):
        with st.spinner("Processing video — this may take a minute..."):
            results = count_players_per_frame(
                model_path=str(best_weights),
                source=input_path,
                class_names=CLASS_NAMES,
                conf_threshold=CONF_THRESHOLD,
                output_video_path=output_path,
            )

        st.success(f"Done — {len(results)} frames processed.")
        st.markdown("---")

        # ── Stats ────────────────────────────────────────────────────────────
        white = [r["white_team"] for r in results]
        red   = [r["red_team"]   for r in results]

        st.subheader("Player Count Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.metric("⬜ White Team — avg", f"{sum(white) / len(white):.1f}")
            st.metric("⬜ White Team — max", max(white))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="red-card">', unsafe_allow_html=True)
            st.metric("🟥 Red Team — avg", f"{sum(red) / len(red):.1f}")
            st.metric("🟥 Red Team — max", max(red))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Video preview ─────────────────────────────────────────────────
        st.subheader("Annotated Video")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇ Download Annotated Video",
                data=f,
                file_name="annotated.mp4",
                mime="video/mp4",
            )
