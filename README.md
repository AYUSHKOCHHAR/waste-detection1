# ♻️ EcoVision — AI Waste Detection

Real-time waste detection and classification using YOLOv8 + Claude AI.

## Features
- 🎥 **Live Webcam** detection (local only)
- 📸 **Photo Upload** with YOLO + Claude Vision analysis
- 🎤 **Voice Questions** via microphone (OpenAI Whisper)
- 🤖 **AI Answers** powered by Claude (Anthropic)
- 🔊 **Text-to-Speech** responses (gTTS)
- 📊 **Charts & History** of detected items

## Waste Categories
| Category | Items |
|---|---|
| ♻ Recyclable | cardboard box, can, plastic bottle, plastic bottle cap, reuseable paper |
| 🗑 Non-Recyclable | plastic bag, straw, plastic cup, snack bag, scrap plastic, etc. |
| ⚠ Hazardous | battery, chemical spray can, light bulb, paint bucket, etc. |

## Setup

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd waste-detection
pip install -r requirements.txt
```

### 2. Add your model
Place your trained YOLOv8 model at:
```
weights/best.pt
```

### 3. Add API Keys
Either enter them in the sidebar at runtime, **or** create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
```

### 4. Run
```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push to GitHub (use Git LFS for `best.pt` if >100MB)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Add API keys in **Secrets** tab (Settings → Secrets)

> ⚠️ Webcam tab only works locally. Use Photo Upload tab on cloud.

## Project Structure
```
waste-detection/
├── app.py               # Main Streamlit app
├── helper.py            # YOLO + Claude + Voice helpers
├── settings.py          # Paths and waste category lists
├── requirements.txt     # Python dependencies
├── packages.txt         # System apt packages
├── weights/
│   └── best.pt          # Your trained YOLOv8 model (add manually)
└── .streamlit/
    └── config.toml      # Theme config
```
