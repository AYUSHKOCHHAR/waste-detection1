from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
from datetime import datetime
import anthropic
import base64
import tempfile
import os
from gtts import gTTS
from io import BytesIO
import openai


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(model_path):
    return YOLO(model_path)


# ── Waste classification ──────────────────────────────────────────────────────
def classify_waste_type(detected_items):
    recyclable_items     = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items      = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_recyclable_items, hazardous_items


def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")


# ── History ───────────────────────────────────────────────────────────────────
def _save_to_history(recyclable_items, non_recyclable_items, hazardous_items):
    if 'detection_history' not in st.session_state:
        st.session_state['detection_history'] = []

    timestamp = datetime.now().strftime("%H:%M:%S")
    existing  = [e['item'] for e in st.session_state['detection_history']]

    for item in recyclable_items:
        name = remove_dash_from_class_name(item)
        if name not in existing:
            st.session_state['detection_history'].append(
                {'item': name, 'category': 'Recyclable', 'time': timestamp})
            existing.append(name)

    for item in non_recyclable_items:
        name = remove_dash_from_class_name(item)
        if name not in existing:
            st.session_state['detection_history'].append(
                {'item': name, 'category': 'Non-Recyclable', 'time': timestamp})
            existing.append(name)

    for item in hazardous_items:
        name = remove_dash_from_class_name(item)
        if name not in existing:
            st.session_state['detection_history'].append(
                {'item': name, 'category': 'Hazardous', 'time': timestamp})
            existing.append(name)


# ── Text-to-Speech ────────────────────────────────────────────────────────────
def text_to_speech(text: str) -> BytesIO:
    """Convert text to speech using gTTS and return audio bytes."""
    tts = gTTS(text=text, lang='en', slow=False)
    audio_buf = BytesIO()
    tts.write_to_fp(audio_buf)
    audio_buf.seek(0)
    return audio_buf


# ── Speech-to-Text (OpenAI Whisper) ──────────────────────────────────────────
def speech_to_text(audio_bytes: bytes, openai_api_key: str) -> str:
    """Transcribe audio bytes using OpenAI Whisper API."""
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f
            )
        os.unlink(tmp_path)
        return transcript.text
    except Exception as e:
        return f"[Transcription error: {e}]"


# ── Claude Vision API ─────────────────────────────────────────────────────────
def ask_claude_about_image(
    image_bytes: bytes,
    question: str,
    detected_items: list,
    anthropic_api_key: str,
) -> str:
    """Send image + question to Claude and get waste disposal advice."""
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Build context from YOLO detections
        if detected_items:
            items_str = ", ".join(detected_items)
            context = f"Our YOLO model has already detected these waste items in the image: {items_str}. "
        else:
            context = "No specific waste items were detected by our YOLO model. "

        system_prompt = """You are EcoVision, an expert AI waste management assistant. 
Your job is to help users properly dispose of waste items.
When analyzing images:
1. Identify all waste items visible
2. Classify them as Recyclable, Non-Recyclable, or Hazardous
3. Give clear, actionable disposal instructions
4. Be concise but informative (3-5 sentences max)
5. Always end with an eco-tip"""

        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{context}User question: {question}",
                        },
                    ],
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude API error: {e}"


def ask_claude_text_only(
    question: str,
    detected_items: list,
    anthropic_api_key: str,
) -> str:
    """Ask Claude a text-only question about detected waste items."""
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        if detected_items:
            items_str = ", ".join(detected_items)
            context = f"The following waste items were detected: {items_str}. "
        else:
            context = ""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are EcoVision, an expert waste management AI.
{context}
User question: {question}

Give a concise answer (3-5 sentences) with disposal advice and an eco-tip.""",
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude API error: {e}"


# ── YOLO frame processing ──────────────────────────────────────────────────────
def _get_allowed_classes(model):
    """Return list of class IDs excluding 'person'."""
    return [k for k, v in model.names.items() if v.lower() != 'person']


def _draw_boxes(image, res, names):
    """Draw colored bounding boxes on image, skipping person class."""
    annotated = image.copy()
    if not res[0].boxes:
        return annotated, set()

    detected = set()
    for box in res[0].boxes:
        cls_id   = int(box.cls[0])
        cls_name = names[cls_id]
        if cls_name.lower() == 'person':
            continue

        detected.add(cls_name)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = f"{remove_dash_from_class_name(cls_name)} {conf:.2f}"

        if cls_name in settings.RECYCLABLE:
            color = (78, 192, 233)
        elif cls_name in settings.HAZARDOUS:
            color = (85, 84, 194)
        else:
            color = (173, 128, 94)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated, detected


def _display_detected_frames(model, st_frame, image):
    image = cv2.resize(image, (640, int(640 * (9 / 16))))

    if 'unique_classes'             not in st.session_state:
        st.session_state['unique_classes'] = set()
    if 'recyclable_placeholder'     not in st.session_state:
        st.session_state['recyclable_placeholder']     = st.sidebar.empty()
    if 'non_recyclable_placeholder' not in st.session_state:
        st.session_state['non_recyclable_placeholder'] = st.sidebar.empty()
    if 'hazardous_placeholder'      not in st.session_state:
        st.session_state['hazardous_placeholder']      = st.sidebar.empty()
    if 'last_detection_time'        not in st.session_state:
        st.session_state['last_detection_time'] = 0

    # Auto-clear sidebar after 3s
    if time.time() - st.session_state['last_detection_time'] > 3:
        st.session_state['recyclable_placeholder'].empty()
        st.session_state['non_recyclable_placeholder'].empty()
        st.session_state['hazardous_placeholder'].empty()

    allowed = _get_allowed_classes(model)
    res     = model.predict(image, conf=0.6, classes=allowed)
    names   = model.names

    annotated, new_classes = _draw_boxes(image, res, names)

    if new_classes != st.session_state['unique_classes']:
        st.session_state['unique_classes'] = new_classes
        recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(new_classes)

        if recyclable_items:
            items_str = ", ".join(remove_dash_from_class_name(i) for i in recyclable_items)
            st.session_state['recyclable_placeholder'].markdown(
                f"<div class='stRecyclable'>♻ Recyclable: {items_str}</div>",
                unsafe_allow_html=True)
        if non_recyclable_items:
            items_str = ", ".join(remove_dash_from_class_name(i) for i in non_recyclable_items)
            st.session_state['non_recyclable_placeholder'].markdown(
                f"<div class='stNonRecyclable'>🗑 Non-Recyclable: {items_str}</div>",
                unsafe_allow_html=True)
        if hazardous_items:
            items_str = ", ".join(remove_dash_from_class_name(i) for i in hazardous_items)
            st.session_state['hazardous_placeholder'].markdown(
                f"<div class='stHazardous'>⚠ Hazardous: {items_str}</div>",
                unsafe_allow_html=True)

        _save_to_history(recyclable_items, non_recyclable_items, hazardous_items)
        st.session_state['last_detection_time'] = time.time()

    st_frame.image(annotated, channels="BGR")


# ── Webcam loop ───────────────────────────────────────────────────────────────
def play_webcam(model):
    source_webcam = settings.WEBCAM_PATH

    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button('▶ Start Detection', use_container_width=True):
            st.session_state['webcam_running'] = True
            st.session_state['unique_classes'] = set()
    with col2:
        if st.button('⏹ Stop Camera', use_container_width=True):
            st.session_state['webcam_running'] = False

    if st.session_state['webcam_running']:
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            if not vid_cap.isOpened():
                st.error("❌ Camera not found. Check WEBCAM_PATH in settings.py")
                st.session_state['webcam_running'] = False
                return

            st_frame = st.empty()
            while vid_cap.isOpened() and st.session_state.get('webcam_running', False):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                else:
                    st.warning("⚠ Could not read frame from camera.")
                    break
            vid_cap.release()

        except Exception as e:
            st.sidebar.error("Error: " + str(e))
            st.session_state['webcam_running'] = False

        st.rerun()
