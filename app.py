from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import io
import helper
import settings

st.set_page_config(
    page_title="EcoVision — Waste Detection",
    page_icon="♻️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f12; }

section[data-testid="stSidebar"] {
    background: #111418 !important;
    border-right: 1px solid #1e2228;
}
section[data-testid="stSidebar"] * { color: #c8d0dc !important; }

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 1.4rem 0 2rem;
    border-bottom: 1px solid #1e2228; margin-bottom: 1.5rem;
}
.sidebar-brand .icon { font-size: 28px; line-height: 1; }
.sidebar-brand .name {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 20px; color: #e8edf2 !important; letter-spacing: -0.3px;
}
.sidebar-brand .version {
    font-size: 11px; color: #4ade80 !important;
    letter-spacing: 0.8px; font-weight: 500;
}
.info-card {
    background: #181c22; border: 1px solid #1e2228;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.info-card .label {
    font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
    color: #556070 !important; font-weight: 500; margin-bottom: 4px;
}
.info-card .value { font-size: 14px; font-weight: 500; color: #c8d0dc !important; }

.hero { padding: 3rem 0 1.8rem; max-width: 680px; }
.hero .tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.25);
    color: #4ade80; font-size: 11px; font-weight: 500;
    letter-spacing: 1px; text-transform: uppercase;
    padding: 5px 12px; border-radius: 100px; margin-bottom: 20px;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: 46px; font-weight: 800;
    line-height: 1.08; letter-spacing: -1.5px; color: #e8edf2; margin: 0 0 14px;
}
.hero h1 span { color: #4ade80; }
.hero p {
    font-size: 15px; line-height: 1.7; color: #7a8694;
    font-weight: 300; margin: 0; max-width: 520px;
}
.badges-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 1.8rem 0 2.2rem; }
.badge {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 15px; border-radius: 100px;
    font-size: 13px; font-weight: 500;
}
.badge-recyclable    { background: rgba(233,192,78,0.10); border: 1px solid rgba(233,192,78,0.28); color: #E9C04E; }
.badge-nonrecyclable { background: rgba(94,128,173,0.10); border: 1px solid rgba(94,128,173,0.28); color: #8ab0d8; }
.badge-hazardous     { background: rgba(194,84,85,0.10);  border: 1px solid rgba(194,84,85,0.28);  color: #e07879; }
.badge-dot { width: 7px; height: 7px; border-radius: 50%; }
.dot-y { background: #E9C04E; }
.dot-b { background: #8ab0d8; }
.dot-r { background: #e07879; }

.detection-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
}
.detection-title {
    font-family: 'Syne', sans-serif; font-size: 16px;
    font-weight: 600; color: #c8d0dc; letter-spacing: -0.3px;
}
.live-dot {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; letter-spacing: 0.8px; text-transform: uppercase;
    color: #4ade80; font-weight: 500;
}
.live-dot::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80; animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
.sec-header {
    font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700;
    color: #c8d0dc; letter-spacing: -0.3px;
    border-left: 3px solid #4ade80; padding-left: 12px;
    margin: 2.2rem 0 1rem;
}
.kpi-card {
    background: #111418; border: 1px solid #1e2228;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 4px;
}
.kpi-label {
    font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
    color: #3a4452; font-weight: 500; margin-bottom: 7px;
}
.kpi-value {
    font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800;
    color: #e8edf2; letter-spacing: -0.8px; line-height: 1;
}
.stRecyclable {
    background: rgba(233,192,78,0.08); border: 1px solid rgba(233,192,78,0.22);
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; border-radius: 10px;
    font-size: 14px; font-weight: 500; color: #E9C04E;
}
.stNonRecyclable {
    background: rgba(94,128,173,0.08); border: 1px solid rgba(94,128,173,0.22);
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; border-radius: 10px;
    font-size: 14px; font-weight: 500; color: #8ab0d8;
}
.stHazardous {
    background: rgba(194,84,85,0.08); border: 1px solid rgba(194,84,85,0.25);
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; border-radius: 10px;
    font-size: 14px; font-weight: 500; color: #e07879;
}
.ai-response-card {
    background: linear-gradient(135deg, #111c18 0%, #111418 100%);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 14px; padding: 1.4rem 1.6rem; margin: 1rem 0;
    color: #c8d0dc; font-size: 14px; line-height: 1.75;
}
.ai-response-card .ai-label {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700;
    color: #4ade80; letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 10px;
}
.voice-card {
    background: #111418; border: 1px solid #1e2228;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin: 1rem 0;
}
.tab-label {
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700;
    color: #4ade80; letter-spacing: 0.5px;
}
.stButton > button {
    background: #4ade80 !important; color: #0d0f12 !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 14px !important; padding: 11px 26px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #22c55e !important;
    box-shadow: 0 6px 20px rgba(74,222,128,0.22) !important;
}
.empty-state {
    text-align: center; padding: 3rem 0;
    color: #3a4452; font-size: 14px;
}
.dash-footer {
    font-size: 11px; color: #2a3340;
    border-top: 1px solid #1e2228;
    padding: 1.5rem 0; margin-top: 2.5rem; letter-spacing: 0.4px;
}
hr { border-color: #1e2228 !important; }
</style>
""", unsafe_allow_html=True)

WASTE_PALETTE = {
    "Recyclable":     "#E9C04E",
    "Non-Recyclable": "#8ab0d8",
    "Hazardous":      "#e07879",
}
PLOTLY_BASE = dict(
    plot_bgcolor="#111418", paper_bgcolor="#111418",
    font=dict(family="DM Sans", color="#7a8694"),
    margin=dict(t=16, b=16, l=8, r=8),
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ('detection_history', []),
    ('webcam_running', False),
    ('ai_response', ''),
    ('last_detected_items', []),
    ('photo_detected_items', []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="icon">♻️</div>
        <div>
            <div class="name">EcoVision</div>
            <div class="version">● LIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-card"><div class="label">Model</div><div class="value">YOLOv8 Detection</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><div class="label">Mode</div><div class="value">Webcam + Photo Upload</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><div class="label">AI Assistant</div><div class="value">Claude Vision + Voice</div></div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;color:#3a4452;margin-bottom:8px;letter-spacing:0.5px;text-transform:uppercase;'>API Keys</p>", unsafe_allow_html=True)

    # Read from secrets.toml if available (for cloud deploy) — safe fallback
    try:
        _ant_secret = st.secrets.get("ANTHROPIC_API_KEY", "")
        _oai_secret = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        _ant_secret = ""
        _oai_secret = ""

    anthropic_key = st.text_input(
        "Anthropic API Key", type="password",
        value=_ant_secret,
        placeholder="sk-ant-...",
        help="Required for AI assistant + voice features"
    )
    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        value=_oai_secret,
        placeholder="sk-...",
        help="Required for microphone speech-to-text (Whisper)"
    )
    st.markdown("<p style='font-size:11px;color:#3a4452;line-height:1.6;margin-top:8px;'>Keys are never stored. Used only for this session.</p>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="tag">♻ AI-Powered Detection</div>
    <h1>Intelligent<br>Waste <span>Segregation</span></h1>
    <p>Use your webcam or upload a photo, ask questions by voice or text, and get instant AI-powered waste disposal guidance.</p>
</div>
<div class="badges-row">
    <div class="badge badge-recyclable"><div class="badge-dot dot-y"></div>Recyclable</div>
    <div class="badge badge-nonrecyclable"><div class="badge-dot dot-b"></div>Non-Recyclable</div>
    <div class="badge badge-hazardous"><div class="badge-dot dot-r"></div>Hazardous</div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
def get_counts():
    h = st.session_state.get('detection_history', [])
    return (
        len(h),
        sum(1 for e in h if e['category'] == 'Recyclable'),
        sum(1 for e in h if e['category'] == 'Non-Recyclable'),
        sum(1 for e in h if e['category'] == 'Hazardous'),
    )

def kpi(col, label, value):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>""", unsafe_allow_html=True)

total, r_count, n_count, h_count = get_counts()
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "📦 Total Detected", total)
kpi(k2, "♻ Recyclable",      r_count)
kpi(k3, "🗑 Non-Recyclable", n_count)
kpi(k4, "⚠ Hazardous",       h_count)

# ── Load YOLO model ───────────────────────────────────────────────────────────
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model: {model_path}")
    st.error(ex)
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS — Webcam  |  Photo Upload
# ══════════════════════════════════════════════════════════════════════════════
tab_webcam, tab_photo = st.tabs(["🎥 Live Webcam", "📸 Photo Upload"])

# ── TAB 1 — Webcam ────────────────────────────────────────────────────────────
with tab_webcam:
    st.markdown("""
    <div class="detection-header" style="margin-top:1.2rem">
        <div class="detection-title">Live Camera Feed</div>
        <div class="live-dot">Live</div>
    </div>
    """, unsafe_allow_html=True)
    helper.play_webcam(model)

# ── TAB 2 — Photo Upload ─────────────────────────────────────────────────────
with tab_photo:
    st.markdown("<div class='sec-header' style='margin-top:1.2rem'>Upload a Photo</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_bgr, (640, int(640 * (9 / 16))))

        # Run YOLO
        allowed = helper._get_allowed_classes(model)
        res     = model.predict(img_resized, conf=0.6, classes=allowed)
        annotated, detected_set = helper._draw_boxes(img_resized, res, model.names)

        # Show annotated image
        col_img, col_info = st.columns([3, 2])
        with col_img:
            st.image(annotated, channels="BGR", use_container_width=True)

        with col_info:
            if detected_set:
                recyclable_items, non_recyclable_items, hazardous_items = helper.classify_waste_type(detected_set)

                if recyclable_items:
                    items = ", ".join(helper.remove_dash_from_class_name(i) for i in recyclable_items)
                    st.markdown(f"<div class='stRecyclable'>♻ Recyclable<br><small>{items}</small></div>", unsafe_allow_html=True)
                if non_recyclable_items:
                    items = ", ".join(helper.remove_dash_from_class_name(i) for i in non_recyclable_items)
                    st.markdown(f"<div class='stNonRecyclable'>🗑 Non-Recyclable<br><small>{items}</small></div>", unsafe_allow_html=True)
                if hazardous_items:
                    items = ", ".join(helper.remove_dash_from_class_name(i) for i in hazardous_items)
                    st.markdown(f"<div class='stHazardous'>⚠ Hazardous<br><small>{items}</small></div>", unsafe_allow_html=True)

                # Save to history
                helper._save_to_history(recyclable_items, non_recyclable_items, hazardous_items)
                st.session_state['photo_detected_items'] = [
                    helper.remove_dash_from_class_name(i) for i in detected_set
                ]
            else:
                st.markdown("<div class='empty-state'>No waste items detected</div>", unsafe_allow_html=True)
                st.session_state['photo_detected_items'] = []

        # ── AI Assistant for uploaded photo ──────────────────────────────────
        st.markdown("<div class='sec-header'>🤖 Ask AI About This Image</div>", unsafe_allow_html=True)

        # Store image bytes for Claude
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        img_bytes_for_claude = img_encoded.tobytes()

        voice_tab, text_tab = st.tabs(["🎤 Voice Question", "⌨️ Type Question"])

        # ── Voice input ───────────────────────────────────────────────────────
        with voice_tab:
            st.markdown("<div class='voice-card'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#7a8694;font-size:13px;margin-bottom:12px;'>Record your question and AI will answer about this image.</p>", unsafe_allow_html=True)

            try:
                from audio_recorder_streamlit import audio_recorder
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#4ade80",
                    neutral_color="#3a4452",
                    icon_size="2x",
                )
            except ImportError:
                st.warning("audio-recorder-streamlit not installed. Add it to requirements.txt")
                audio_bytes = None

            if audio_bytes and openai_key and anthropic_key:
                with st.spinner("🎙 Transcribing..."):
                    question = helper.speech_to_text(audio_bytes, openai_key)
                st.markdown(f"<p style='color:#c8d0dc;font-size:13px;'>🗣 <em>{question}</em></p>", unsafe_allow_html=True)

                with st.spinner("🤖 Claude is analyzing..."):
                    response = helper.ask_claude_about_image(
                        img_bytes_for_claude, question,
                        st.session_state['photo_detected_items'],
                        anthropic_key
                    )
                st.session_state['ai_response'] = response

            elif audio_bytes and not openai_key:
                st.warning("⚠ Add OpenAI API key in sidebar for speech-to-text.")
            elif audio_bytes and not anthropic_key:
                st.warning("⚠ Add Anthropic API key in sidebar for AI response.")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Text input ────────────────────────────────────────────────────────
        with text_tab:
            question_text = st.text_input(
                "Ask a question about this waste item",
                placeholder="How do I dispose of this battery?",
                label_visibility="collapsed"
            )
            if st.button("Ask AI 🤖", key="ask_photo_text"):
                if not anthropic_key:
                    st.warning("⚠ Add Anthropic API key in sidebar.")
                elif question_text.strip():
                    with st.spinner("🤖 Claude is analyzing..."):
                        response = helper.ask_claude_about_image(
                            img_bytes_for_claude, question_text,
                            st.session_state['photo_detected_items'],
                            anthropic_key
                        )
                    st.session_state['ai_response'] = response

        # ── Show AI Response + TTS ────────────────────────────────────────────
        if st.session_state.get('ai_response'):
            st.markdown(f"""
            <div class="ai-response-card">
                <div class="ai-label">🤖 EcoVision AI</div>
                {st.session_state['ai_response']}
            </div>
            """, unsafe_allow_html=True)

            # Text-to-Speech button
            if st.button("🔊 Listen to Response"):
                with st.spinner("Generating audio..."):
                    audio_buf = helper.text_to_speech(st.session_state['ai_response'])
                st.audio(audio_buf, format="audio/mp3")

# ══════════════════════════════════════════════════════════════════════════════
# Charts + History
# ══════════════════════════════════════════════════════════════════════════════
history = st.session_state.get('detection_history', [])
total, r_count, n_count, h_count = get_counts()

st.markdown("<hr>", unsafe_allow_html=True)
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("<div class='sec-header'>Detection Breakdown</div>", unsafe_allow_html=True)

    if history:
        df         = pd.DataFrame(history)
        cat_counts = df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']

        fig_bar = px.bar(
            cat_counts, x='Category', y='Count',
            color='Category', color_discrete_map=WASTE_PALETTE, text='Count',
        )
        fig_bar.update_traces(textposition='outside', marker_line_width=0)
        fig_bar.update_layout(
            **PLOTLY_BASE, showlegend=False,
            xaxis=dict(showgrid=False, title=None, color='#3a4452'),
            yaxis=dict(showgrid=True, gridcolor='#1e2228', title='Items', color='#3a4452'),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            values=cat_counts['Count'], names=cat_counts['Category'],
            color=cat_counts['Category'], color_discrete_map=WASTE_PALETTE, hole=0.55,
        )
        fig_pie.update_layout(
            **PLOTLY_BASE, showlegend=True,
            legend=dict(font=dict(color='#7a8694', size=12), bgcolor='rgba(0,0,0,0)'),
        )
        fig_pie.update_traces(marker=dict(line=dict(color='#0d0f12', width=2)))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.markdown("<div class='empty-state'>Charts will appear after detection</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='sec-header'>Session History</div>", unsafe_allow_html=True)

    if history:
        cat_css = {
            'Recyclable':     ('stRecyclable',    '♻'),
            'Non-Recyclable': ('stNonRecyclable', '🗑'),
            'Hazardous':      ('stHazardous',     '⚠'),
        }
        for entry in reversed(history):
            css, icon = cat_css.get(entry['category'], ('stRecyclable', '•'))
            st.markdown(
                f"<div class='{css}'>"
                f"<span>{icon} &nbsp;<strong>{entry['item'].title()}</strong> · {entry['category']}</span>"
                f"<span style='font-size:11px;opacity:0.5;margin-left:8px;'>{entry['time']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("🗑 Clear History"):
            st.session_state['detection_history'] = []
            st.session_state['ai_response'] = ''
            st.rerun()
    else:
        st.markdown("<div class='empty-state'>No items detected yet</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-footer">
    EcoVision Waste Detection System &nbsp;·&nbsp; Powered by YOLOv8 + Claude AI &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
