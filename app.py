import os
import re
import json
import uuid
import time
import shutil
import subprocess
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import requests
import gradio as gr
from faster_whisper import WhisperModel

# NEW: OpenAI + Gemini SDKs
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ────────────────────────────────────────────────────────────────────────────────
# QUICK ENV SETUP (examples)
#   # Groq:
#   #   GROQ_API_KEY, GROQ_MODEL (default openai/gpt-oss-120b), GROQ_SCOUT_MODEL
#   #
#   # OpenRouter:
#   #   OPENROUTER_API_KEY
#   #   - deepseek/deepseek-r1-distill-llama-70b
#   #   - google/gemma-2-9b-it
#   #
#   # OpenAI (GPT-4o-mini or other):
#   #   OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini)
#   #
#   # Google Gemini (2.0 Flash by default):
#   #   GEMINI_API_KEY, GEMINI_MODEL (default gemini-2.0-flash)
#   #
#   # Gmail (app-level email only):
#   #   GMAIL_USER, GMAIL_APP_PASSWORD  (App Password required)
# ────────────────────────────────────────────────────────────────────────────────

# Optional helpers
try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# --------------------------------------------------------------------------------
# App-level email config (hard-coded list; not exposed in UI)
APP_EMAIL_ENABLED = True
APP_EMAIL_RECIPIENTS = [
    "vibhamishra.outr@gmail.com",
    "vibhamishra0907@gmail.com",
]
# --------------------------------------------------------------------------------

# ===================== Paths & .env loader =====================
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "workspace"
AUDIO_DIR = WORK_DIR / "audio"
UPLOAD_DIR = WORK_DIR / "uploads"
BOARD_DIR  = WORK_DIR / "boards"
for p in (WORK_DIR, AUDIO_DIR, UPLOAD_DIR, BOARD_DIR):
    p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(UPLOAD_DIR))

def _load_env():
    if load_dotenv:
        load_dotenv(BASE_DIR / ".env", override=False)
        load_dotenv(BASE_DIR / "secrets" / ".env", override=True)
    cfg = BASE_DIR / "secrets" / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        for k, v in data.items():
            os.environ.setdefault(k, str(v))

_load_env()

# ===================== Whisper config =====================
WHISPER_SIZE   = os.getenv("WHISPER_SIZE", "small")
COMPUTE_TYPE   = os.getenv("WHISPER_COMPUTE", "auto")
WORD_TS        = os.getenv("WORD_TS", "0") == "1"
PREFER_GST     = os.getenv("PREFER_GSTREAMER", "1") != "0"

CHUNK_SEC      = int(os.getenv("CHUNK_SEC", "900"))
MAX_DIRECT_SEC = int(os.getenv("MAX_DIRECT_SEC", "1200"))
ALWAYS_SEGMENT = os.getenv("ALWAYS_SEGMENT", "0") == "1"

TRANSLATE_DEFAULT       = os.getenv("TRANSLATE_DEFAULT", "1") == "1"
PROMPT_TRANSCRIPT_CHARS = int(os.getenv("PROMPT_TRANSCRIPT_CHARS", "1000"))

# ===== Visual analysis knobs =====
VISION_SAMPLE_SEC = float(os.getenv("VISION_SAMPLE_SEC", "1.5"))
HAND_IOU_THRESH   = float(os.getenv("HAND_IOU_THRESH", "0.35"))
MAX_BOARD_FRAMES  = int(os.getenv("MAX_BOARD_FRAMES", "8"))
BOARD_WHITE_PCT   = float(os.getenv("BOARD_WHITE_PCT", "0.22"))
BOARD_DARK_PCT    = float(os.getenv("BOARD_DARK_PCT", "0.22"))

# ===================== Gmail SMTP ENV =====================
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SMTP_DEBUG = os.getenv("SMTP_DEBUG", "0") == "1"

# ===================== Groq ENV =====================
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))
GROQ_TEMP       = float(os.getenv("GROQ_TEMP", "0.2"))

# Groq Llama-4-Scout
GROQ_SCOUT_MODEL      = os.getenv("GROQ_SCOUT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_SCOUT_MAX_TOKENS = int(os.getenv("GROQ_SCOUT_MAX_TOKENS", "900"))
GROQ_SCOUT_TEMP       = float(os.getenv("GROQ_SCOUT_TEMP", "0.2"))

# ===================== OpenAI ENV (NEW) =====================
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "900"))
OPENAI_TEMP       = float(os.getenv("OPENAI_TEMP", "0.2"))

# ===================== Gemini ENV =====================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMP        = float(os.getenv("GEMINI_TEMP", "0.2"))
GEMINI_MAX_TOKENS  = int(os.getenv("GEMINI_MAX_TOKENS", "1200"))
GEMINI_CHUNK_CHARS = int(os.getenv("GEMINI_CHUNK_CHARS", "7000"))
GEMINI_OVERLAP     = int(os.getenv("GEMINI_OVERLAP", "500"))

# ===================== Ollama (feedback only) =====================
def _ollama_base_url() -> str:
    raw = (os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") or "").strip()
    raw = raw.rstrip("/")
    if raw.endswith("/api"):
        raw = raw[:-4]
    u = urlparse(raw)
    scheme = u.scheme or "http"
    host   = u.hostname or "127.0.0.1"
    port   = u.port or 11434
    return f"{scheme}://{host}:{port}"

def _ollama_probe_or_raise():
    r = requests.get(f"{_ollama_base_url()}/api/version", timeout=(5, 10))
    r.raise_for_status()

# ===================== OpenRouter ENV =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODELS = {
    "or_deepseek_r1d_70b": "deepseek/deepseek-r1-distill-llama-70b",
    "or_gemma2_9b_it":     "google/gemma-2-9b-it",
}

# Shared provider timeout
PROVIDER_TIMEOUT = int(os.getenv("PROVIDER_TIMEOUT", "120"))

# ====== local analysis helpers ======
from analyze import analyze_transcript, extract_topics

# ===================== Instantiate Whisper =====================
model = WhisperModel(
    WHISPER_SIZE,
    device="auto",
    compute_type=COMPUTE_TYPE,
    download_root=str(WORK_DIR / "models"),
)


# ===================== Text chunk util (Gemini map-reduce) =====================
def _chunk_text(text: str, target_chars: int = 7000, overlap: int = 500):
    text = text or ""
    if len(text) <= target_chars:
        return [text]
    chunks, i = [], 0
    step = max(target_chars - overlap, 1000)
    while i < len(text):
        chunks.append(text[i:i+target_chars])
        i += step
    return chunks

# ===================== Small utils =====================
def _which(bin_name: str) -> Optional[str]:
    return shutil.which(bin_name)

def _ffprobe_duration(path: Path) -> float:
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return 0.0
    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    try:
        return float((res.stdout or b"").decode().strip())
    except Exception:
        return 0.0

def _ffprobe_streams(path: Path) -> dict:
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return {}
    res = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    try:
        return json.loads((res.stdout or b"{}").decode())
    except Exception:
        return {}

def _has_audio_stream(path: Path) -> bool:
    info = _ffprobe_streams(path)
    return bool(info.get("streams"))

def _normalize_uploaded(vpath):
    if not vpath:
        return None
    if isinstance(vpath, str):
        return vpath
    if isinstance(vpath, (list, tuple)) and v:
        return vpath[0]
    if isinstance(vpath, dict):
        return vpath.get("name") or vpath.get("path")
    return None

def _mmss(sec: float) -> str:
    sec = max(float(sec or 0), 0.0)
    m = int(sec // 60); s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def _mmss_to_sec(s: str) -> float:
    try:
        mm, ss = s.strip().split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0.0

def _trim_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2, 200)
    return text[:half] + "\n...\n" + text[-half:]

# ===================== Audio extraction =====================
def _gst_extract(in_path: Path, out_wav: Path) -> None:
    gst = _which("gst-launch-1.0")
    if not gst:
        raise FileNotFoundError("gst-launch-1.0 not found")
    v = in_path.resolve().as_posix()
    a = out_wav.resolve().as_posix()
    cmd = [
        gst, "-q",
        "filesrc", f"location={v}",
        "!", "decodebin",
        "!", "audioconvert",
        "!", "audioresample",
        "!", "audio/x-raw,channels=1,rate=16000,format=S16LE",
        "!", "wavenc",
        "!", "filesink", f"location={a}",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        err = (res.stderr or b"").decode(errors="ignore")
        raise RuntimeError(f"GStreamer failed: {err[:800]}")

def _ffmpeg_extract(in_path: Path, out_wav: Path) -> None:
    ffm = _which("ffmpeg")
    if not ffm:
        raise FileNotFoundError("ffmpeg not found")
    if not _has_audio_stream(in_path):
        raise RuntimeError(f"No audio stream found in file: {in_path}")
    cmd = [
        ffm, "-hide_banner", "-nostdin", "-y",
        "-i", str(in_path),
        "-vn", "-sn", "-dn",
        "-map", "0:a:0?",
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        err = (res.stderr or b"").decode(errors="ignore")[:800]
        raise RuntimeError(err)

def extract_audio(video_path: Path) -> Path:
    out_wav = AUDIO_DIR / f"{video_path.stem}.wav"
    try:
        if PREFER_GST:
            try:
                _gst_extract(video_path, out_wav)
            except Exception as e:
                print("[extract] GStreamer failed; trying FFmpeg:", e)
                _ffmpeg_extract(video_path, out_wav)
        else:
            try:
                _ffmpeg_extract(video_path, out_wav)
            except Exception as e:
                print("[extract] FFmpeg failed; trying GStreamer:", e)
                _gst_extract(video_path, out_wav)
        return out_wav
    except Exception as e:
        print("[extract] Both extractors failed; switching to long-mode. Reason:", e)
        os.environ["ALWAYS_SEGMENT"] = "1"
        return out_wav

# ===================== URL handling =====================
def _is_http_url(s: Optional[str]) -> bool:
    if not s:
        return False
    try:
        u = urlparse(s.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def _download_media_from_url(url: str, dest_dir: Path) -> Path:
    url = url.strip()
    file_id = uuid.uuid4().hex
    exts = (".mp4", ".mkv", ".mov", ".webm", ".mp3", ".m4a", ".wav", ".aac", ".flac")
    base_no_qs = url.split("?", 1)[0].lower()

    # Direct-file URL
    if any(base_no_qs.endswith(e) for e in exts):
        suffix = Path(base_no_qs).suffix
        dst = dest_dir / f"{file_id}{suffix}"
        with requests.get(url, stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        if not dst.exists() or dst.stat().st_size == 0:
            raise gr.Error("Direct download failed or empty file.")
        return dst

    # Platforms (YouTube/Drive/etc.) via yt-dlp
    if YoutubeDL is None:
        raise gr.Error("This URL isn’t a direct media file. Install yt-dlp: pip install yt-dlp")
    outtmpl = str(dest_dir / f"{file_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        files = list(dest_dir.glob(f"{file_id}.*"))
        if not files:
            guess = Path(ydl.prepare_filename(info))
            if guess.exists():
                return guess
            raise gr.Error("Download succeeded but cannot find the output file.")
    return files[0]

# ===================== Long-media segmentation =====================
def segment_media_to_wavs(src_media: Path, chunk_sec: int = CHUNK_SEC) -> List[Path]:
    ffm = _which("ffmpeg")
    if not ffm:
        raise FileNotFoundError("ffmpeg not found")
    if not _has_audio_stream(src_media):
        raise RuntimeError(f"No audio stream found in file: {src_media}")

    out_dir = AUDIO_DIR / f"{src_media.stem}_chunks"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_dir / f"{src_media.stem}_%04d.wav"
    cmd = [
        ffm, "-hide_banner", "-loglevel", "error",
        "-i", str(src_media),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "segment", "-segment_time", str(chunk_sec),
        str(pattern),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg segmentation failed: {(res.stderr or b'').decode(errors='ignore')[:800]}")
    parts = sorted(out_dir.glob(f"{src_media.stem}_*.wav"))
    if not parts:
        raise RuntimeError("Segmentation produced no chunks.")
    return parts

# ===================== Prompt builder for feedback =====================
def _build_feedback_prompt(transcript_text: str, segments: List[dict]) -> Tuple[str, str]:
    t_snip = _trim_middle(transcript_text or "", PROMPT_TRANSCRIPT_CHARS)
    lines = []
    for s in segments[:60]:
        txt = (s.get("text","" ).strip().replace("\n"," "))[:90]
        if txt:
            lines.append(f"{_mmss(s.get('start',0))}–{_mmss(s.get('end',0))} {txt}")

    system_msg = (
        "You are “TeachCoach”. Only pedagogy-improvement advice (no audio/production; no praise). "
        "Detect topics; then produce:\n"
        "0) Detected Topics (bullets)\n"
        "1) Top 5 Improvements (fix + why + how)\n"
        "2) Minute-by-minute Fixes “(mm:ss) → fix”\n"
        "3) Add These Examples (2–3)\n"
        "4) Ask These Questions (4)\n"
        "5) Next-Class Plan (10 steps)\n"
        "Be concrete with timestamps. Return markdown only."
    )
    user_msg = f"Transcript (truncated):\n{t_snip}\n\nSegments:\n" + "\n".join(lines)
    return system_msg, user_msg

# ===================== Whisper transcription =====================
def transcribe_wav_chunk(path: Path, language_hint: Optional[str], initial_prompt: Optional[str],
                         prime: bool, task_mode: str = "transcribe"):
    seg_iter, info = model.transcribe(
        str(path),
        task=task_mode,
        language=None if not language_hint or language_hint == "auto" else language_hint,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        initial_prompt=(initial_prompt if prime else None),
        word_timestamps=WORD_TS,
    )
    segs, texts = [], []
    for s in seg_iter:
        t = (s.text or "").strip()
        segs.append({"start": float(getattr(s, "start", 0.0) or 0.0),
                     "end": float(getattr(s, "end", 0.0) or 0.0),
                     "text": t})
        if t:
            texts.append(t)
    return segs, " ".join(texts), float(getattr(info, "duration", 0.0) or 0.0)

def transcribe_long(src_media: Path, language_hint: Optional[str], initial_prompt: Optional[str],
                    task_mode: str = "transcribe") -> Tuple[List[dict], str, float]:
    parts = segment_media_to_wavs(src_media, CHUNK_SEC)
    all_segments: List[Dict[str, Any]] = []
    full_text: List[str] = []
    total_off = 0.0
    for i, wav_part in enumerate(parts):
        segs, text, dur = transcribe_wav_chunk(wav_part, language_hint, initial_prompt, i == 0, task_mode)
        for s in segs:
            all_segments.append({
                "start": s["start"] + total_off,
                "end": s["end"] + total_off,
                "text": s["text"]
            })
        if text:
            full_text.append(text)
        total_off += max(dur, float(CHUNK_SEC))
    duration = all_segments[-1]["end"] if all_segments else total_off
    return all_segments, " ".join(full_text), duration

def transcribe_short(wav_path: Path, language_hint: Optional[str], initial_prompt: Optional[str],
                     task_mode: str = "transcribe") -> Tuple[List[dict], str, float]:
    segments_iter, info = model.transcribe(
        str(wav_path),
        task=task_mode,
        language=None if not language_hint or language_hint == "auto" else language_hint,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        initial_prompt=(initial_prompt or None),
        word_timestamps=WORD_TS,
    )
    segments: List[dict] = []
    full_text = []
    for s in segments_iter:
        seg = {
            "start": float(getattr(s, "start", 0.0) or 0.0),
            "end": float(getattr(s, "end", 0.0) or 0.0),
            "text": (s.text or "").strip()
        }
        segments.append(seg)
        if seg["text"]:
            full_text.append(seg["text"])
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return segments, " ".join(full_text), duration

# ===================== Provider feedbacks =====================
def openrouter_feedback_model(transcript_text: str, segments: List[dict], model_id: str) -> str:
    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY env var")

    system_msg, user_msg = _build_feedback_prompt(transcript_text, segments)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Lecture Analyzer",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role":"system","content":system_msg},
            {"role":"user","content":user_msg}
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=PROVIDER_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content"," ").strip()
    if not text:
        raise RuntimeError(f"OpenRouter empty response: {j}")
    return text

def ollama_feedback(transcript_text: str, segments: List[dict]) -> str:
    _ollama_probe_or_raise()
    base       = _ollama_base_url()
    model_id   = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    temp       = float(os.getenv("OLLAMA_TEMP", "0.0"))
    num_ctx    = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    num_predict= int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
    timeout    = int(os.getenv("OLLAMA_TIMEOUT", "600"))

    t_snip = _trim_middle(transcript_text or "", PROMPT_TRANSCRIPT_CHARS)
    lines = []
    for s in segments[:60]:
        txt = (s.get("text","").strip().replace("\n"," "))[:90]
        if txt:
            lines.append(f"{_mmss(s.get('start',0))}–{_mmss(s.get('end',0))} {txt}")

    system_msg = (
        "You are “TeachCoach”… (pedagogy advice only). "
        "0) Topics  1) Top 5 Improvements  2) (mm:ss)→fix  3) Examples  4) Questions(4)  5) Next-Class Plan(10)."
    )
    user_msg = f"Transcript (truncated):\n{t_snip}\n\nSegments:\n" + "\n".join(lines)
    merged = f"[SYSTEM]\n{system_msg}\n[/SYSTEM]\n[USER]\n{user_msg}\n[/USER]\n"

    payload = {
        "model": model_id,
        "prompt": merged,
        "stream": False,
        "keep_alive": "2h",
        "options": {"temperature": temp, "num_ctx": num_ctx, "num_predict": num_predict, "mirostat": 0},
    }

    r = requests.post(f"{base}/api/generate", json=payload, timeout=(10, timeout))
    if r.status_code == 404:
        raise gr.Error("Ollama 404 on /api/generate. Ensure OLLAMA_URL does not include /api.")
    r.raise_for_status()
    content = (r.json().get("response") or "").strip()
    if not content:
        raise gr.Error("Ollama returned empty feedback.")
    return content

def make_groq():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    maybe_proxies = None
    if os.getenv("FORCE_GROQ_PROXIES_JSON"):
        import json as _j
        try:
            maybe_proxies = _j.loads(os.getenv("FORCE_GROQ_PROXIES_JSON"))
        except Exception:
            maybe_proxies = None
    try:
        if maybe_proxies:
            return Groq(api_key=api_key, proxies=maybe_proxies)
        return Groq(api_key=api_key)
    except TypeError:
        return Groq(api_key=api_key)

def groq_feedback(transcript_text: str, segments: List[dict]) -> str:
    if Groq is None:
        raise RuntimeError("groq SDK not installed. Run: pip install groq")
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY env var")
    client = make_groq()
    system_msg, user_msg = _build_feedback_prompt(transcript_text, segments)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=GROQ_TEMP,
        max_tokens=GROQ_MAX_TOKENS,
        stream=False,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned empty text")
    return text

def groq_scout_feedback(transcript_text: str, segments: List[dict]) -> str:
    if Groq is None:
        raise RuntimeError("groq SDK not installed. Run: pip install groq")
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY env var")
    client = make_groq()
    system_msg, user_msg = _build_feedback_prompt(transcript_text, segments)
    resp = client.chat.completions.create(
        model=GROQ_SCOUT_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=GROQ_SCOUT_TEMP,
        max_tokens=GROQ_SCOUT_MAX_TOKENS,
        stream=False,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq (Llama-4-Scout) returned empty text")
    return text

# NEW: OpenAI feedback
def openai_feedback(transcript_text: str, segments: List[dict]) -> str:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY env var")
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_msg, user_msg = _build_feedback_prompt(transcript_text, segments)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=OPENAI_TEMP,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned empty text")
    return text

# UPDATED: Gemini feedback (map-reduce for long transcripts)
def gemini_feedback(transcript_text: str, segments: list[dict]) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai SDK not installed. Run: pip install google-generativeai")
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    model_g = genai.GenerativeModel(GEMINI_MODEL)

    system_msg, user_msg = _build_feedback_prompt(transcript_text, segments)
    parts = _chunk_text(user_msg, GEMINI_CHUNK_CHARS, GEMINI_OVERLAP)
    partials: list[str] = []
    for i, ch in enumerate(parts, 1):
        resp = model_g.generate_content(
            [{"text": system_msg},{"text": f"(Part {i} of {len(parts)})\n{ch}"}],
            generation_config={"temperature": GEMINI_TEMP, "max_output_tokens": GEMINI_MAX_TOKENS},
        )
        partials.append((getattr(resp, "text", None) or "").strip())
    if len(partials) == 1:
        return partials[0]
    resp = model_g.generate_content(
        [
            {"text": "Combine and deduplicate the partial analyses below into ONE cohesive report, "
                      "strictly following the same 'TeachCoach' rubric with timestamps kept where present."},
            {"text": "\n\n---\n\n".join(partials)}
        ],
        generation_config={"temperature": GEMINI_TEMP, "max_output_tokens": GEMINI_MAX_TOKENS},
    )
    return (getattr(resp, "text", None) or "").strip()

# ===================== Heuristic Q&A (local) =====================
Q_RE = re.compile(
    r"(?:^|[\s\"“])(?:why|what|how|when|where|which|can|could|should|would|is|are|do|does)\b.*?\?",
    re.IGNORECASE
)

def qna_heuristic(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    questions = []
    sid_counter = 0
    last_sid = None
    last_q_end = -1e9
    for i, s in enumerate(segments):
        txt = (s.get("text","") or "").strip()
        if not txt:
            continue
        if "?" in txt or Q_RE.search(txt):
            sid = None
            if (s["start"] - last_q_end) <= 90 and last_sid is not None:
                sid = last_sid
            if not sid:
                sid_counter += 1
                sid = f"s{sid_counter}"
            last_sid = sid
            last_q_end = s["end"]
            answered = False
            for j in range(i+1, min(i+12, len(segments))):
                s2 = segments[j]
                if (s2["start"] - s["end"]) > 120:
                    break
                txt2 = (s2.get("text","") or "").strip()
                if len(txt2.split()) >= 6 and "?" not in txt2:
                    answered = True
                    break
            questions.append({
                "t_start": float(s.get("t_start", s.get("start",0.0))),
                "t_end": float(s.get("t_end", s.get("end",0.0))),
                "question": txt,
                "student_id": sid,
                "answered": answered,
            })
    return {"questions": questions}

def _items_from_heuristic(segments: List[dict]) -> List[dict]:
    h = qna_heuristic(segments)
    items: List[dict] = []
    for q in h.get("questions", []):
        items.append({
            "t": _mmss(q.get("t_start", 0.0)),
            "speaker": "student",
            "text": q.get("question", ""),
            "answered": bool(q.get("answered", False)),
            "student_id": (q.get("student_id") or "s?")
        })
    return items

# ===================== Gmail SMTP helpers =====================
def send_gmail_smtp(to_email: str, subject: str, body_text: str, body_html: Optional[str] = None) -> None:
    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        raise RuntimeError("Set GMAIL_USER and GMAIL_APP_PASSWORD env vars first.")
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body_text or "")
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        if SMTP_DEBUG:
            s.set_debuglevel(1)
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)

def send_report_to_recipients(recipients: List[str], subject: str, body_text: str, body_html: Optional[str] = None) -> None:
    for addr in (recipients or []):
        a = (addr or "").strip()
        if not a:
            continue
        try:
            send_gmail_smtp(a, subject, body_text, body_html)
            print(f"[email] sent → {a}")
        except Exception as e:
            print(f"[email] failed → {a}: {e}")

# ===================== ADAPTIVE WEIGHTING (NEW) =====================

WEIGHTS_FILE = os.environ.get("MODEL_WEIGHTS_FILE", str(BASE_DIR / "model_weights.json"))

# Provider base weights (least → most)
PROVIDER_BASE_WEIGHTS = {
    "openrouter": 1.0,   # least
    "groq":       2.0,
    "ollama":     3.0,
    "gpt":        5.0,   # most
    "gemini":     5.0,   # most (same as GPT)
}

BONUS_SCALE = 1.25        # strength of vote effect
BONUS_MIN   = -2.0
BONUS_MAX   =  2.0

def _load_weights() -> Dict[str, Any]:
    if not os.path.exists(WEIGHTS_FILE):
        return {"votes": {}, "version": 1}
    try:
        return json.loads(Path(WEIGHTS_FILE).read_text())
    except Exception:
        return {"votes": {}, "version": 1}

def _save_weights(data: Dict[str, Any]) -> None:
    Path(WEIGHTS_FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False))

def _provider_for_model(model_key: str) -> str:
    # normalize key to provider bucket
    if model_key in ("openai",):
        return "gpt"
    if model_key in ("gemini",):
        return "gemini"
    if model_key in ("groq", "scout"):
        return "groq"
    if model_key.startswith("or_"):
        return "openrouter"
    if model_key in ("ollama",):
        return "ollama"
    # fallback: assume lower preference
    return "openrouter"

def _base_weight(model_key: str) -> float:
    return PROVIDER_BASE_WEIGHTS.get(_provider_for_model(model_key), 1.0)

def _vote_tuple(weights_store: Dict[str, Any], key: str) -> Tuple[int, int]:
    v = weights_store.get("votes", {}).get(key, {})
    return int(v.get("up", 0)), int(v.get("down", 0))

def _bonus_from_votes(up: int, down: int) -> float:
    # Smooth, spam-resistant; ln ratio with clamp
    import math
    bonus = math.log((up + 1) / (down + 1.0)) * BONUS_SCALE
    return float(max(BONUS_MIN, min(BONUS_MAX, bonus)))

def effective_weight(weights_store: Dict[str, Any], model_key: str) -> float:
    base = _base_weight(model_key)
    up, down = _vote_tuple(weights_store, model_key)
    bonus = _bonus_from_votes(up, down)
    return base + bonus

def register_vote(model_key: str, upvote: bool) -> Dict[str, Any]:
    store = _load_weights()
    store.setdefault("votes", {})
    rec = store["votes"].setdefault(model_key, {"up": 0, "down": 0})
    if upvote:
        rec["up"] = int(rec.get("up", 0)) + 1
    else:
        rec["down"] = int(rec.get("down", 0)) + 1
    _save_weights(store)
    return store

def ranking_md(weights_store: Dict[str, Any], available_models: List[str]) -> str:
    rows = []
    for k in available_models:
        up, down = _vote_tuple(weights_store, k)
        wt = effective_weight(weights_store, k)
        rows.append((k, wt, up, down, _provider_for_model(k), _base_weight(k)))
    rows.sort(key=lambda r: r[1], reverse=True)
    name_map = {
        "openai": "OpenAI · GPT",
        "gemini": "Google · Gemini",
        "groq": "Groq (GPT-OSS-120B)",
        "scout": "Groq (Llama-4-Scout)",
        "ollama": "Ollama (local)",
        "or_deepseek_r1d_70b": "OpenRouter · DeepSeek R1D 70B",
        "or_gemma2_9b_it": "OpenRouter · Gemma2-9B-IT",
    }
    lines = ["**Model Ranking (higher is preferred)**",
             "",
             "| Rank | Model | Provider | Base | Bonus(votes) | Up | Down | Eff. Weight |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for i, (k, wt, up, down, prov, base) in enumerate(rows, 1):
        bonus = wt - base
        label = name_map.get(k, k)
        lines.append(f"| {i} | {label} | {prov} | {base:.2f} | {bonus:+.2f} | {up} | {down} | {wt:.2f} |")
    return "\n".join(lines)

# ===================== Multi-provider selector (with ordering) =====================
# mode one of:
#   'groq' | 'scout' | 'ollama' |
#   'or_deepseek_r1d_70b' | 'or_gemma2_9b_it' |
#   'openai' | 'gemini' |
#   'both' (groq + ollama) | 'all' (all seven)

def get_feedbacks(transcript_text: str, segments: List[dict], mode: str) -> dict:
    m = (mode or "groq").lower()
    if m == "all":
        want = {"groq","scout","ollama","or_deepseek_r1d_70b","or_gemma2_9b_it","openai","gemini"}
    elif m == "both":
        want = {"groq","ollama"}
    else:
        want = {m}

    calls = {}
    with ThreadPoolExecutor(max_workers=len(want) or 1) as pool:
        if "groq" in want:
            calls["groq"] = pool.submit(groq_feedback, transcript_text, segments)
        if "scout" in want:
            calls["scout"] = pool.submit(groq_scout_feedback, transcript_text, segments)
        if "ollama" in want:
            calls["ollama"] = pool.submit(ollama_feedback, transcript_text, segments)
        if "openai" in want:
            calls["openai"] = pool.submit(openai_feedback, transcript_text, segments)
        if "gemini" in want:
            calls["gemini"] = pool.submit(gemini_feedback, transcript_text, segments)
        for key, model_id in OPENROUTER_MODELS.items():
            if key in want:
                calls[key] = pool.submit(openrouter_feedback_model, transcript_text, segments, model_id)

        out = {}
        for name, fut in calls.items():
            try:
                out[name] = fut.result()
            except Exception as e:
                out[name] = f"[{name} failed: {e}]"

        # ORDER by effective weight (desc)
        store = _load_weights()
        ordered = sorted(out.keys(), key=lambda k: effective_weight(store, k), reverse=True)
        out["_ordered_keys"] = ordered
        return out

# ===================== Helpers for Q&A table =====================
def qna_rows_from_items(items: List[dict]) -> List[List[Any]]:
    rows = []
    for it in items:
        tsec = _mmss_to_sec(it.get("t","00:00"))
        spk  = (it.get("speaker") or "").lower()
        sid  = it.get("student_id") or ("S?" if spk == "student" else "T")
        rows.append([round(tsec,2), round(tsec,2), sid, bool(it.get("answered", False)), it.get("text",""), "", spk])
    return rows

def qna_summary_from_items(items: List[dict]) -> Dict[str, Any]:
    total_stu = sum(1 for it in items if (it.get("speaker") or "").lower() == "student")
    total_tea = sum(1 for it in items if (it.get("speaker") or "").lower() == "teacher")
    uniq_students = len({(it.get("student_id") or "").lower()
                         for it in items if (it.get("speaker") or "").lower()=="student" and it.get("student_id")})
    answered_est = sum(1 for it in items if (it.get("speaker") or "").lower()=="student" and bool(it.get("answered")))
    return {
        "total_items": len(items),
        "total_student_questions": total_stu,
        "total_teacher_questions": total_tea,
        "unique_students_est": uniq_students,
        "answered": answered_est,
        "unanswered": max(total_stu - answered_est, 0)
    }

# ===================== Pipeline =====================
def process(
     src_file: Optional[str],
     video_url: Optional[str],
     language_hint: str,
     initial_prompt: str,
     translate_to_en: bool,
     analyze_visuals_flag: bool,
     feedback_mode: str,
 ) -> Dict[str, Any]:

    def _normalize(v):
        if isinstance(v, list) and v:
            return v[0]
        return v

    src_file = _normalize(_normalize_uploaded(src_file))
    chosen: Optional[Path] = None

    if src_file and Path(src_file).exists():
        chosen = Path(src_file)
    elif _is_http_url(video_url):
        try:
            chosen = _download_media_from_url(video_url.strip(), WORK_DIR)
        except Exception as e:
            raise gr.Error(f"Could not fetch URL: {e}")

    if not chosen:
        raise gr.Error("No valid media provided (upload a file or paste a video URL).")

    file_id = uuid.uuid4().hex
    dst_vid = WORK_DIR / f"{file_id}{chosen.suffix.lower()}"
    for _ in range(20):
        try:
            with open(chosen, "rb") as r, open(dst_vid, "wb") as w:
                shutil.copyfileobj(r, w, 1024 * 1024)
            break
        except PermissionError:
            time.sleep(0.25)
    else:
        raise gr.Error("Could not read the file (Windows locked it). Close players and try again.")

    media_len = _ffprobe_duration(dst_vid)
    use_long = ALWAYS_SEGMENT or (media_len and media_len > MAX_DIRECT_SEC)
    task_mode = "translate" if translate_to_en else "transcribe"

    if use_long:
        segments, transcript_text, duration_sec = transcribe_long(dst_vid, language_hint, initial_prompt, task_mode)
    else:
        wav = extract_audio(dst_vid)
        if os.getenv("ALWAYS_SEGMENT", "0") == "1":
            segments, transcript_text, duration_sec = transcribe_long(dst_vid, language_hint, initial_prompt, task_mode)
        else:
            segments, transcript_text, duration_sec = transcribe_short(wav, language_hint, initial_prompt, task_mode)

    _ = analyze_transcript(transcript_text, segments, duration_sec)
    _ = extract_topics(transcript_text)

    items = _items_from_heuristic(segments)
    q_rows = qna_rows_from_items(items)
    q_summary = qna_summary_from_items(items)

    m = (feedback_mode or "").lower()
    if m == "both":
        mode = "both"
    elif m in {"scout","ollama","groq","or_deepseek_r1d_70b","or_gemma2_9b_it","all","openai","gemini"}:
        mode = m
    else:
        mode = "groq"

    feedback_map = get_feedbacks(transcript_text, segments, mode)

    # ORDERED primary by adaptive weight
    ordered_keys = feedback_map.pop("_ordered_keys", [])
    store = _load_weights()

    primary = "*No feedback produced.*"
    for k in ordered_keys:
        text = feedback_map.get(k)
        if text and not str(text).startswith("[") :
            primary = text
            break
    used_engine = ", ".join(ordered_keys) if ordered_keys else ", ".join(sorted(feedback_map.keys()))

    # visuals: append to each feedback
    visuals: Dict[str, Any] = {}
    if analyze_visuals_flag:
        visuals = analyze_visuals(dst_vid, file_id)
        vr = visuals
        hand_line = f"- *Students raised hands (unique est.)*: {vr.get('hand_raise_unique', 0)}"
        board_line = f"- *Board snapshots*: {len(vr.get('board_snapshots', []))}"
        board_text = vr.get("board_text", "")
        add_section = "\n\n### Classroom Visuals\n" + hand_line + "\n" + board_line
        if board_text:
            add_section += "\n- *Board OCR (condensed)*:\n\n" + (board_text[:2000]) + ("\n..." if len(board_text)>2000 else "\n")
        for k in list(feedback_map.keys()):
            feedback_map[k] = (feedback_map[k] or "") + add_section
        primary = (primary or "") + add_section

    # Email
    if APP_EMAIL_ENABLED and APP_EMAIL_RECIPIENTS:
        parts = []
        titles = {
            "openai": "OpenAI · GPT",
            "gemini": "Google · Gemini",
            "groq": "Groq (GPT-OSS-120B)",
            "scout": "Groq (Llama-4-Scout)",
            "or_deepseek_r1d_70b": "OpenRouter · DeepSeek R1-Distill-Llama-70B",
            "or_gemma2_9b_it": "OpenRouter · Gemma2-9B-IT",
            "ollama": "Ollama (local)",
        }
        for k in ordered_keys:
            if k in feedback_map and feedback_map[k]:
                parts.append(f"### {titles.get(k,k)}\n{feedback_map[k]}")
        if not parts and primary:
            parts = [primary]
        subject = f"Lecture report — {file_id}"
        body_join = "\n\n".join(parts)
        text_body = (
            "Lecture Feedback\n================\n" + body_join +
            "\n\nQ&A Summary\n-----------\n" +
            f"Total items: {q_summary.get('total_items',0)} | "
            f"Student: {q_summary.get('total_student_questions',0)} | "
            f"Teacher: {q_summary.get('total_teacher_questions',0)} | "
            f"Answered(est): {q_summary.get('answered',0)} | "
            f"Unanswered(est): {q_summary.get('unanswered',0)} | "
            f"Unique students(est): {q_summary.get('unique_students_est',0)}\n"
        )
        html_body = (
            "<h2>Lecture Feedback</h2>"
            f"<div style='white-space:pre-wrap'>{body_join}</div>"
            "<h3>Q&amp;A Summary</h3><p>"
            f"Total items: {q_summary.get('total_items',0)} | "
            f"Student: {q_summary.get('total_student_questions',0)} | "
            f"Teacher: {q_summary.get('total_teacher_questions',0)} | "
            f"Answered(est): {q_summary.get('answered',0)} | "
            f"Unanswered(est): {q_summary.get('unanswered',0)} | "
            f"Unique students(est): {q_summary.get('unique_students_est',0)}</p>"
        )
        try:
            send_report_to_recipients(APP_EMAIL_RECIPIENTS, subject, text_body, html_body)
        except Exception as e:
            print("[email] batch send failed:", e)

    # Build ranking markdown and voting model list
    available_models = ordered_keys or list(feedback_map.keys())
    rank_md = ranking_md(store, available_models)

    return {
        "file_id": file_id,
        "duration_sec": duration_sec,
        "transcript_text": transcript_text,
        "segments": segments,
        "paragraph_primary": primary,
        "feedback_map": feedback_map,
        "ordered_keys": ordered_keys,
        "feedback_engine": used_engine,
        "qna_items": items,
        "qna_rows": q_rows,
        "qna_summary": q_summary,
        "visuals": visuals,
        "ranking_md": rank_md,
        "available_models": available_models,
    }

# ===================== Visual analysis (hand-raise + board) =====================
def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    iw = max(0, xB - xA); ih = max(0, yB - yA)
    inter = iw * ih
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def _is_board_like(frame_bgr) -> bool:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]; s = hsv[:,:,1]
    white_mask = (v > 225) & (s < 30)
    dark_mask  = (v < 50)
    white_pct = white_mask.mean()
    dark_pct  = dark_mask.mean()
    return (white_pct >= BOARD_WHITE_PCT) or (dark_pct >= BOARD_DARK_PCT)

def _ocr_text(frame_bgr) -> str:
    if pytesseract is None:
        return ""
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        txt = pytesseract.image_to_string(gray, lang="eng")
        return txt.strip()
    except Exception:
        return ""

def analyze_visuals(video_path: Path, file_id: str) -> Dict[str, Any]:
    out = {"hand_raise_unique": 0, "hand_raise_events": [], "board_snapshots": [], "board_text": ""}
    if cv2 is None:
        out["board_text"] = "[cv2 not installed — visuals skipped]"
        return out
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        out["board_text"] = "[could not open video — visuals skipped]"
        return out

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * VISION_SAMPLE_SEC), 1)

    pose = None
    if YOLO is not None:
        try:
            pose = YOLO("yolov8n-pose.pt")
        except Exception as e:
            print("[visuals] YOLO load failed:", e)

    next_id = 1
    tracks = []
    raised_ids = set()
    saved_boards = 0
    board_texts = []

    frame_idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        ret, frame = (False, None)
        if frame_idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            curr = []
            if pose is not None:
                try:
                    res = pose(frame, verbose=False)
                    for r in res:
                        if getattr(r, "keypoints", None) is None:
                            continue
                        for kp, box in zip(r.keypoints.xy, r.boxes.xyxy):
                            k = kp.cpu().numpy()
                            x1,y1,x2,y2 = [float(v) for v in box.cpu().numpy()]
                            bbox = (x1,y1,x2,y2)
                            def get(i):
                                if i < k.shape[0]:
                                    return float(k[i,0]), float(k[i,1])
                                return None
                            Ls, Lw, Rs, Rw = get(5), get(7), get(6), get(8)
                            raised = False
                            def above(a,b): return (a is not None and b is not None) and (a[1] < b[1] - 8)
                            if above(Lw, Ls) or above(Rw, Rs):
                                raised = True
                            curr.append({"bbox":bbox, "raised":raised})
                except Exception as e:
                    print("[visuals] pose failed:", e)

            if curr:
                for c in curr:
                    best_iou, best_j = 0.0, -1
                    for j, tr in enumerate(tracks):
                        iou = _iou(c["bbox"], tr["bbox"])
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= HAND_IOU_THRESH and best_j >= 0:
                        tracks[best_j]["bbox"] = c["bbox"]
                        if c["raised"]:
                            tracks[best_j]["raised"] = True
                            raised_ids.add(tracks[best_j]["id"])
                    else:
                        tid = next_id; next_id += 1
                        tracks.append({"id":tid, "bbox":c["bbox"], "raised":c["raised"]})
                        if c["raised"]:
                            raised_ids.add(tid)

                if len(tracks) > 256:
                    tracks = tracks[-128:]

                hand_cnt = sum(1 for c in curr if c["raised"])
                if hand_cnt > 0:
                    out["hand_raise_events"].append({"t": ts, "count": hand_cnt})

            took = False
            if saved_boards < MAX_BOARD_FRAMES and _is_board_like(frame):
                snap_path = BOARD_DIR / f"{file_id}_board_{saved_boards+1:02d}.jpg"
                try:
                    cv2.imwrite(str(snap_path), frame)
                    out["board_snapshots"].append(str(snap_path))
                    saved_boards += 1
                    took = True
                except Exception as e:
                    print("[visuals] save board failed:", e)

            if took:
                text = _ocr_text(frame)
                if text:
                    board_texts.append(text)

        frame_idx += 1

    cap.release()
    out["hand_raise_unique"] = int(len(raised_ids))
    out["board_text"] = ("\n\n".join(board_texts)).strip()
    return out

# ===================== UI =====================
LANG_OPTIONS = ["auto", "en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]

def _choose_engine_choice(choice: Optional[str], default="groq") -> str:
    if not isinstance(choice, str):
        return default
    s = choice.strip().lower()
    mapping = {
        "groq (gpt-oss-120b)": "groq",
        "— preferred": "groq",
        "groq (llama-4-scout)": "scout",
        "ollama": "ollama",
        "openrouter · deepseek r1-distill-llama-70b": "or_deepseek_r1d_70b",
        "openrouter · gemma2-9b-it": "or_gemma2_9b_it",
        "openai · gpt-4o-mini": "openai",
        "google · gemini": "gemini",
        "both": "both",
        "all": "all",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return default

with gr.Blocks(title="Lecture Analyzer — Whisper + Heuristic Q&A + Multi-Feedback + Visuals + Adaptive Ranking") as demo:
    gr.Markdown("# Lecture Analyzer")

    with gr.Row():
        with gr.Column(scale=1):
            video_url_tb = gr.Textbox(
                label="Video URL (http/https)",
                placeholder="https://... (YouTube, Google Drive, direct .mp4, etc.)"
            )
            inp = gr.File(label="Upload lecture video", file_types=["video"], type="filepath")

            lang = gr.Dropdown(LANG_OPTIONS, value="auto", label="Language (hint)")
            iprompt = gr.Textbox(
                label="Initial prompt / topic hint (optional)",
                lines=2,
                placeholder="e.g., Class 10 Arithmetic Progression: define AP, nth term, sum, examples"
            )
            translate_toggle = gr.Checkbox(
                label="Translate non-English → English (Whisper translate)",
                value=TRANSLATE_DEFAULT
            )

            gr.Markdown("### Feedback engine")
            feedback_engine_dd = gr.Radio(
                choices=[
                    "Groq (gpt-oss-120b) — preferred",
                    "Groq (Llama-4-Scout)",
                    "OpenAI · GPT-4o-mini",
                    "Google · Gemini",
                    "Ollama (local)",
                    "OpenRouter · DeepSeek R1-Distill-Llama-70B",
                    "OpenRouter · Gemma2-9B-IT",
                    "Both (Groq + Ollama)",
                    "All (compare)"
                ],
                value="Groq (gpt-oss-120b) — preferred",
                label="Choose model(s) for feedback"
            )

            gr.Markdown("### Visuals (optional)")
            vis_on = gr.Checkbox(label="Analyze video for hand-raises + board snapshots (beta)", value=True)

            run = gr.Button("Analyze", variant="primary")

    with gr.Tab("Transcript"):
        transcript_box = gr.Textbox(label="Transcript (English if translated)", lines=12)

    with gr.Tab("Segments"):
        segtbl = gr.Dataframe(headers=["start", "end", "text"], datatype=["number", "number", "str"])

    with gr.Tab("Teaching Feedback"):
        with gr.Tabs():
            with gr.TabItem("Primary (Top by Weight)"):
                feedback_primary_box = gr.Textbox(label="Primary feedback (adaptive order)", lines=16)
                engine_md = gr.Markdown()
            with gr.TabItem("OpenAI · GPT"):
                fb_openai = gr.Textbox(label="OpenAI feedback", lines=14)
            with gr.TabItem("Google · Gemini"):
                fb_gemini = gr.Textbox(label="Gemini feedback", lines=14)
            with gr.TabItem("Groq (GPT-OSS-120B)"):
                fb_groq = gr.Textbox(label="Groq feedback", lines=14)
            with gr.TabItem("Groq (Llama-4-Scout)"):
                fb_scout = gr.Textbox(label="Groq (Llama-4-Scout) feedback", lines=14)
            with gr.TabItem("Ollama"):
                fb_ollama = gr.Textbox(label="Ollama feedback", lines=14)
            with gr.TabItem("OpenRouter · DeepSeek R1-Distill-Llama-70B"):
                fb_or_deepseek = gr.Textbox(label="OpenRouter · DeepSeek R1-Distill-Llama-70B feedback", lines=14)
            with gr.TabItem("OpenRouter · Gemma2-9B-IT"):
                fb_or_gemma = gr.Textbox(label="OpenRouter · Gemma2-9B-IT feedback", lines=14)

    with gr.Tab("Q&A Stats"):
        qna_summary_md = gr.Markdown()
        qna_tbl = gr.Dataframe(
            headers=["t_start","t_end","student_id","answered","question","answer_span","notes"],
            datatype=["number","number","str","bool","str","str","str"],
            wrap=True
        )

    with gr.Tab("Visuals"):
        visuals_md = gr.Markdown()
        board_gallery = gr.Gallery(label="Board snapshots", columns=3, height=300)

    # NEW: Ranking & Voting
    with gr.Tab("Model Ranking & Votes"):
        ranking_md_box = gr.Markdown(value="Run analysis to see current ranking…")
        with gr.Row():
            vote_model_dd = gr.Dropdown(choices=[], label="Model to vote", interactive=True)
        with gr.Row():
            up_btn = gr.Button("👍 Thumbs Up", variant="primary")
            down_btn = gr.Button("👎 Thumbs Down", variant="secondary")
        vote_result_md = gr.Markdown()

    with gr.Tab("Raw JSON"):
        rawjson_box = gr.Code(label="Full Response (JSON)", language="json")

    # STATE: last available models
    available_models_state = gr.State([])

    def ui_process(vpath, video_url, language_hint, initial_prompt,
                   translate_to_en, feedback_engine_choice,
                   vis_flag, progress=gr.Progress()):
        progress(0.05, desc="Preparing & copying file…")

        fe = _choose_engine_choice(feedback_engine_choice, default="groq")
        if fe in ("both", "all"):
            fe_mode = "both" if fe == "both" else "all"
        else:
            fe_mode = fe

        out = process(
            vpath, video_url, language_hint, initial_prompt,
            translate_to_en, vis_flag, fe_mode
        )

        progress(0.85, desc="Formatting results…")

        segs = out.get("segments") or []
        seg_rows = [[round(s.get("start", 0), 2), round(s.get("end", 0), 2), s.get("text", "")] for s in segs]

        primary = out.get("paragraph_primary", "") or ""
        fb_map = out.get("feedback_map", {}) or {}
        engine_used = out.get("feedback_engine","none")
        engine_note = f"*Feedback Engines (ordered):* {out.get('ordered_keys', [])}  |  *Q&A Engine:* heuristic (local)"

        # Q&A
        rows = out.get("qna_rows") or []
        summ = out.get("qna_summary") or {}
        summary_md = (
            f"*Total items:* {summ.get('total_items',0)}  |  "
            f"*Student questions:* {summ.get('total_student_questions',0)}  |  "
            f"*Teacher questions:* {summ.get('total_teacher_questions',0)}  |  "
            f"*Answered (est):* {summ.get('answered',0)}  |  "
            f"*Unanswered (est):* {summ.get('unanswered',0)}  |  "
            f"*Unique students (est):* {summ.get('unique_students_est',0)}"
        )

        # Visuals
        v = out.get("visuals") or {}
        hr_u = v.get("hand_raise_unique", 0)
        evs  = v.get("hand_raise_events", [])
        snaps= v.get("board_snapshots", [])
        v_md = ""
        if vis_flag:
            v_md = f"*Hand-raises (unique students, est.):* {hr_u}  \n"
            if evs:
                times = ", ".join(_mmss(e.get('t',0)) for e in evs[:12])
                more = " …" if len(evs) > 12 else ""
                v_md += f"Frames with raised hands @ {times}{more}  \n"
            v_md += f"*Board snapshots saved:* {len(snaps)}"
        gallery_imgs = snaps

        # Ranking + vote dropdown
        rank_md = out.get("ranking_md", "—")
        choices = out.get("available_models", [])
        dd_update = gr.update(choices=choices, value=(choices[0] if choices else None))

        progress(0.98, desc="Done")
        return (
            out.get("transcript_text"),
            seg_rows,
            primary,
            engine_note,
            fb_map.get("openai",""),
            fb_map.get("gemini",""),
            fb_map.get("groq",""),
            fb_map.get("scout",""),
            fb_map.get("ollama",""),
            fb_map.get("or_deepseek_r1d_70b",""),
            fb_map.get("or_gemma2_9b_it",""),
            summary_md,
            rows,
            v_md,
            gallery_imgs,
            rank_md,
            dd_update,
            json.dumps(out, ensure_ascii=False, indent=2),
            choices,  # state
        )

    run.click(
        ui_process,
        inputs=[inp, video_url_tb, lang, iprompt, translate_toggle, feedback_engine_dd, vis_on],
        outputs=[
            transcript_box, segtbl,
            feedback_primary_box, engine_md,
            # tabs
            fb_openai, fb_gemini,
            fb_groq, fb_scout, fb_ollama,
            fb_or_deepseek, fb_or_gemma,
            qna_summary_md, qna_tbl,
            visuals_md, board_gallery,
            ranking_md_box,      # Ranking tab
            vote_model_dd,       # update choices/value
            rawjson_box,
            available_models_state,  # keep for voting
        ],
        api_name=False,
        show_api=False,
    )

    # --- Voting handlers ---
    def do_vote(selected_key: str, up: bool, available_models: List[str]):
        if not selected_key:
            return ("Select a model first.", ranking_md(_load_weights(), available_models))
        store = register_vote(selected_key, upvote=up)
        msg = f"Recorded {'👍 upvote' if up else '👎 downvote'} for **{selected_key}**."
        md = ranking_md(store, available_models)
        return (msg, md)

    up_btn.click(
        do_vote,
        inputs=[vote_model_dd, gr.State(True), available_models_state],
        outputs=[vote_result_md, ranking_md_box],
    )
    down_btn.click(
        do_vote,
        inputs=[vote_model_dd, gr.State(False), available_models_state],
        outputs=[vote_result_md, ranking_md_box],
    )

# ===================== Gradio launch =====================
def pick_free_port(preferred: Optional[str] = None, start: int = 7860, end: int  = 7890) -> Optional[int]:
    def free(p: int) -> bool:
        import socket as _s
        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) != 0
    if preferred:
        try:
            pp = int(preferred)
            if free(pp):
                return pp
        except Exception:
            pass
    for p in range(start, end + 1):
        if free(p):
            return p
    return None

if __name__ == "__main__":
    env_port = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    port = pick_free_port(env_port)
    share_flag = bool(int(os.getenv("GRADIO_SHARE", "0")))
    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    try:
        demo.queue(max_size=4).launch(
            show_api=False,
            share=share_flag,
            server_name=host,
            server_port=port,
        )
    except Exception as e:
        print("[gradio] launch error:", e)
        demo.queue(max_size=4).launch(
            show_api=False,
            share=False,
            server_name="127.0.0.1",
            server_port=None
        )
