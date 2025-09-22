# app.py
import os
import uuid
import datetime
import traceback
import requests
import time
import re
import base64
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# File parsing
import docx
from PyPDF2 import PdfReader

# ---------- CONFIG ----------
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# sometimes we need request.url_root while inside helpers; import alias
from flask import request as flask_request

# ---------- Small helpers to save images & build URLs ----------
def save_image_bytes_and_get_url(img_bytes, ext="png"):
    """
    Save raw bytes to uploads/ and return an absolute URL to that file.
    """
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    with open(path, "wb") as f:
        f.write(img_bytes)
    # Build an absolute URL so the frontend (on a different origin) fetches from backend correctly
    base = (flask_request.url_root or "").rstrip("/") if flask_request else os.environ.get("BACKEND_URL", "")
    if not base:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    return f"{base}/uploads/{fname}"


def save_base64_and_return_url(b64_str):
    """
    Given either a data:...base64, or raw base64 string decode and save to uploads/, return absolute URL or None
    """
    if not b64_str or not isinstance(b64_str, str):
        return None
    s = b64_str.strip()
    # if data:...base64,... split
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    s = s.replace("\n", "").replace("\r", "")
    try:
        img_bytes = base64.b64decode(s)
    except Exception as e:
        print("[save_base64] decode failed:", e)
        return None
    # detect extension by signature
    ext = "png"
    if img_bytes.startswith(b"\x89PNG"):
        ext = "png"
    elif img_bytes[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif img_bytes[:4] == b"RIFF":
        ext = "webp"
    return save_image_bytes_and_get_url(img_bytes, ext=ext)


def ensure_absolute_url(s):
    """
    Normalize a returned image identifier into an absolute HTTP(S) URL that the frontend can fetch.
    - if already http(s) -> return as-is
    - if starts with '/uploads/' -> prefix with request.url_root or BACKEND_URL
    - if data:... -> save and return uploaded file URL
    - if looks like base64 -> save and return uploaded file URL
    - otherwise return original string (may be handled by frontend)
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("data:"):
        return save_base64_and_return_url(s)
    if s.startswith("/uploads/"):
        base = (flask_request.url_root or "").rstrip("/") if flask_request else os.environ.get("BACKEND_URL", "").rstrip("/")
        if not base:
            base = os.environ.get("BACKEND_URL", "").rstrip("/")
        return f"{base}{s}"
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    # raw base64 heuristic
    candidate = s.replace("\n", "").replace("\r", "")
    if len(candidate) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
        return save_base64_and_return_url(candidate)
    return s

# CORS
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
except Exception:
    pass

# LLM (Together)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "your_key_here")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")

# Hugging Face image inference (set HF_API_TOKEN in your env)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", None)
# Default to SDXL Base model (working, inference enabled)
HF_MODEL = os.environ.get("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))

# ---------- Stable Horde (fallback / alternative) ----------
STABLE_HORDE_API_KEY = os.environ.get("STABLE_HORDE_API_KEY", "0000000000")
FORCE_STABLE_HORDE = os.environ.get("FORCE_STABLE_HORDE", "false").lower() in ("1", "true", "yes")
STABLE_HORDE_TIMEOUT = int(os.environ.get("STABLE_HORDE_TIMEOUT", "120"))
STABLE_HORDE_POLL_INTERVAL = float(os.environ.get("STABLE_HORDE_POLL_INTERVAL", "2"))

# (call_stablehorde kept mostly as-is) -- it may return data:image/... base64 strings
# We'll post-process those later in generate_via_preferred_provider


def call_stablehorde(prompt, api_key=STABLE_HORDE_API_KEY, timeout=STABLE_HORDE_TIMEOUT, poll_interval=STABLE_HORDE_POLL_INTERVAL):
    """
    Submit an async generation to Stable Horde and poll until done.
    Returns a list of data URLs (data:image/png;base64,...) or raises RuntimeError.
    """
    if not prompt:
        raise RuntimeError("Empty prompt for Stable Horde")

    submit_url = "https://stablehorde.net/api/v2/generate/async"
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "params": {
            "steps": 20,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512
        }
    }

    try:
        r = requests.post(submit_url, json=payload, headers=headers, timeout=30)
    except Exception as e:
        raise RuntimeError(f"Stable Horde submit failed: {e}")

    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise RuntimeError(f"Stable Horde submit error {r.status_code}: {body}")

    try:
        job = r.json()
    except Exception:
        raise RuntimeError("Stable Horde submit: failed to parse JSON response")

    # job id detection (API shapes vary)
    job_id = job.get("id") or job.get("request_id") or job.get("job_id") or job.get("requestUUID")
    if not job_id:
        # sometimes images are returned immediately
        if isinstance(job, dict) and job.get("images"):
            out = []
            for it in job["images"]:
                if isinstance(it, dict) and it.get("img"):
                    out.append("data:image/png;base64," + it["img"])
                elif isinstance(it, str) and it.startswith("http"):
                    out.append(it)
            if out:
                return out
        raise RuntimeError(f"Stable Horde submit: no job id in response: {job}")

    check_url = f"https://stablehorde.net/api/v2/generate/check/{job_id}"
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            s = requests.get(check_url, headers=headers, timeout=30)
            if s.status_code != 200:
                last_err = f"check returned {s.status_code}: {s.text[:200]}"
                time.sleep(poll_interval)
                continue
            status = s.json()
        except Exception as e:
            last_err = f"poll exception: {e}"
            time.sleep(poll_interval)
            continue

        # finished detection
        if status.get("done") or status.get("finished") or status.get("status") == "done":
            images = []
            if isinstance(status, dict):
                if status.get("images") and isinstance(status["images"], list):
                    for it in status["images"]:
                        if isinstance(it, dict) and it.get("img"):
                            images.append("data:image/png;base64," + it["img"])
                        elif isinstance(it, str):
                            if it.startswith("http"):
                                images.append(it)
                            else:
                                images.append("data:image/png;base64," + it)
                if status.get("generations") and isinstance(status["generations"], list):
                    for g in status["generations"]:
                        if isinstance(g, dict) and g.get("img"):
                            images.append("data:image/png;base64," + g["img"])
                if status.get("result") and isinstance(status["result"], list):
                    for it in status["result"]:
                        if isinstance(it, dict) and it.get("img"):
                            images.append("data:image/png;base64," + it["img"])
            images = [i for i in images if i]
            if images:
                return images
            raise RuntimeError(f"Stable Horde finished but returned no images: {status}")

        time.sleep(poll_interval)

    raise RuntimeError(f"Stable Horde generation timed out after {timeout}s. Last error: {last_err}")


# wrapper that chooses provider (HF preferred unless forced off)
def generate_via_preferred_provider(prompt, language="en"):
    """
    Returns (images_list, provider_name)
    Tries Hugging Face first (if HF_API_TOKEN present and not forced off), otherwise uses Stable Horde.
    Ensures returned image strings are absolute URLs that frontend can fetch.
    """
    final_prompt = f"{prompt} (language: {language})"
    # prefer HF unless forced off
    if HF_API_TOKEN and not FORCE_STABLE_HORDE:
        try:
            imgs = call_hf_image(final_prompt)
            provider = "huggingface"
        except Exception as e:
            print("[image] Hugging Face failed, falling back to Stable Horde:", e)
            try:
                imgs = call_stablehorde(final_prompt)
                provider = "stablehorde"
            except Exception as e2:
                raise RuntimeError(f"Hugging Face failed: {e}; Stable Horde failed: {e2}")
    else:
        try:
            imgs = call_stablehorde(final_prompt)
            provider = "stablehorde"
        except Exception as e:
            raise RuntimeError(f"Stable Horde failed: {e}")

    # Normalize images into absolute URLs where possible
    normalized = []
    for it in imgs:
        try:
            u = ensure_absolute_url(it)
            normalized.append(u or it)
        except Exception:
            normalized.append(it)
    return normalized, provider


# log model + token presence at startup (do NOT log token value)
print(f"[startup] HF_MODEL = {HF_MODEL}")
print(f"[startup] HF_API_TOKEN present: {bool(HF_API_TOKEN)}; FORCE_STABLE_HORDE={FORCE_STABLE_HORDE}; STABLE_HORDE_KEY_present={STABLE_HORDE_API_KEY != '0000000000'}")

# Optional Web Search (Tavily). Set env: TAVILY_API_KEY
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_URL = os.environ.get("TAVILY_URL", "https://api.tavily.com/search")

# ---------- AGENT CONFIG (STRICT bullet rules) ----------
AGENTS = {
    "general": {
        "system": (
            "You are BharatAI, a helpful assistant. "
            "Always respond ONLY using bullet points or numbered lists. "
            "Never write paragraphs."
        ),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 600,
    },
    "docs": {
        "system": (
            "You are BharatAI, a retrieval assistant. "
            "Answer ONLY from the provided documents. "
            "If no answer is present, say so. "
            "Always respond in bullet points or numbered lists ONLY. "
            "Never use paragraphs."
        ),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "temperature": 0.2,
        "max_tokens": 700,
    },
    "web": {
        "system": (
            "You are BharatAI, a research assistant. "
            "Summarize search results in 3–6 bullet points with inline citations like [1], [2]. "
            "At the end, add clickable source links. "
            "Do NOT write paragraphs — bullet points only."
        ),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "temperature": 0.4,
        "max_tokens": 700,
    },
    "code": {
        "system": (
            "You are BharatAI, a senior software engineer. "
            "Always start with runnable code (inside triple backticks). "
            "Then explain in short bullet points. "
            "Never use long paragraphs."
        ),
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "temperature": 0.25,
        "max_tokens": 700,
    }
}
DEFAULT_AGENT_KEY = "general"


# ---------- HELPERS ----------
... (rest of your file unchanged) ...
