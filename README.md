🧠 CLASSROOM — AI-Powered Speech Processing Pipeline

This project extracts text, topics, and insights from video/audio sources of offline classrooms to help teachers getting feedback using yt-dlp, Gradio, and LLMs (Gemini / Ollama / OpenAI / Groq).
It supports YouTube downloads, transcription, and topic extraction with optional local or remote LLM inference.

📁 Folder Structure
speech2/
│
├── app.py
├── requirements.txt
├── secrets/
│   ├── .env
│   └── config.json
├── workspace/
├── outputs/
└── README.md


⚠️ Never commit the secrets/ folder to GitHub — it is ignored via .gitignore.

🔐 Secrets Setup
1. Create secrets/.env
mkdir -p secrets
touch secrets/.env

Example .env file
#API Keys

GROQ_API_KEY=your_key

#OpenRouter API (Required for OpenRouter models)
OPENROUTER_API_KEY=your_key
GEMINI_API_KEY=your_key
#OpenAI API
OPENAI_API_KEY=your_key
#Gmail credentials (for sending reports)
GMAIL_USER=your mail
GMAIL_APP_PASSWORD=gmailpassword for sending result in g mail

#Email debugging (optional)
SMTP_DEBUG=0



#Whisper model size (tiny, base, small, medium, large)
WHISPER_SIZE=small

#Compute type (auto, int8, int8_float16, float16, float32)
WHISPER_COMPUTE=auto

#Enable word timestamps (0 or 1)
WORD_TS=0

#Prefer GStreamer over FFmpeg (0 or 1)
PREFER_GSTREAMER=1


#Chunk size for long audio (seconds)
CHUNK_SEC=900

#Maximum duration for direct processing (seconds)
MAX_DIRECT_SEC=1200

#Always segment audio (0 or 1)
ALWAYS_SEGMENT=0

#Default translation setting (0 or 1)
TRANSLATE_DEFAULT=1

#Characters to include in prompt for transcript
PROMPT_TRANSCRIPT_CHARS=1000

# ===========================================
# VISUAL ANALYSIS
# ===========================================

#Sample rate for vision analysis (seconds)
VISION_SAMPLE_SEC=1.5

#Hand detection IoU threshold
HAND_IOU_THRESH=0.35

#Maximum board snapshots to save
MAX_BOARD_FRAMES=8

#Board detection thresholds
BOARD_WHITE_PCT=0.22
BOARD_DARK_PCT=0.22

# ===========================================
# GROQ CONFIGURATION
# ===========================================

#Groq model settings
GROQ_MODEL=openai/gpt-oss-120b
GROQ_MAX_TOKENS=900
GROQ_TEMP=0.2

#Groq Scout model settings
GROQ_SCOUT_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_SCOUT_MAX_TOKENS=900
GROQ_SCOUT_TEMP=0.2

# ===========================================
# OLLAMA CONFIGURATION (Local LLM)
# ===========================================

#Ollama server URL
OLLAMA_URL=http://127.0.0.1:11434

#Ollama model settings
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMP=0.0
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=800
OLLAMA_TIMEOUT=600

# ===========================================
# GENERAL SETTINGS
# ===========================================

#Provider timeout (seconds)
PROVIDER_TIMEOUT=120

# ===========================================
# GRADIO SERVER CONFIGURATION
# ===========================================

#Server settings
GRADIO_SERVER_PORT=7860
GRADIO_HOST=127.0.0.1
GRADIO_SHARE=0

#Ollama overrides
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_NUM_PREDICT=256
OLLAMA_NUM_CTX=4096
OLLAMA_TEMP=0.2

#Ollama overrides
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_NUM_PREDICT=256
OLLAMA_NUM_CTX=4096
OLLAMA_TEMP=0.2

# Ollama overrides
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_NUM_PREDICT=256
OLLAMA_NUM_CTX=4096
OLLAMA_TEMP=0.2

# Ollama overrides
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_NUM_PREDICT=256
OLLAMA_NUM_CTX=4096
OLLAMA_TEMP=0.2

2. Create secrets/config.json
{
    "GROQ_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "GMAIL_USER":"",
    "GMAIL_APP_PASSWORD":"",
    "OLLAMA_URL":"http://127.0.0.1:11434",
    "OLLAMA_MODEL":"qwen2.5:7b-instruct",
    "OLLAMA_TEMP":"0.7",
    "OLLAMA_NUM_CTX":"4096",
    "OLLAMA_NUM_PREDICT":"512",
    "OLLAMA_TIMEOUT":"120"
}

⚙️ Ollama Setup (Local LLM)

If you want to use a local LLM instead of OpenAI/Groq:

Install Ollama

curl -fsSL https://ollama.com/install.sh | sh


Pull a model

ollama pull qwen2.5:7b-instruct


Run the Ollama server

ollama serve


(It runs on http://127.0.0.1:11434 by default.)

🧩 Installation
1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt  ##if it shows package is outdated upgrade :)


If you face build issues with packages like av or numpy, try:

pip install --no-cache-dir -r requirements.txt

▶️ Run the Project
python app.py


Then open the Gradio link shown in the terminal (usually http://127.0.0.1:7860).
