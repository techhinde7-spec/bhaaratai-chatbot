# app.py — Final clean version for Bhaaratai backend
# Integrated Hugging Face image & video generation endpoints

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
from flask_cors import CORS

# ---------- APP & CONFIG ----------
app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000"
]

# Enable CORS
CORS(app, origins=FRONTEND_ORIGINS, supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "apikey", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, apikey, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response


# ---------- BASIC ROUTES ----------
@app.route("/")
def home():
    return jsonify({"status": "Bhaaratai Backend Running ✅"})

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()})


# ---------- HELPER FUNCTIONS ----------
def save_bytes_and_get_url(b: bytes, content_type: str = None, ext_hint: str = None):
    """Save raw bytes to uploads/ and return an absolute URL to that file."""
    ext = "bin"
    if ext_hint:
        ext = ext_hint
    elif content_type and "/" in content_type:
        ext = content_type.split("/")[1].split(";")[0]
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    with open(path, "wb") as f:
        f.write(b)
    base = request.host_url.rstrip("/")
    return f"{base}/uploads/{fname}"


def save_base64_and_return_url(b64_str):
    if not b64_str or not isinstance(b64_str, str):
        return None
    s = b64_str.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    try:
        b = base64.b64decode(s)
    except Exception:
        return None
    return save_bytes_and_get_url(b, content_type="image/png", ext_hint="png")


def ensure_absolute_url(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("data:"):
        return save_base64_and_return_url(s)
    if re.match(r"^https?://", s, re.I):
        return s
    if len(s) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return save_base64_and_return_url(s)
    return s


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)


# ---------- HUGGING FACE CONFIG ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))


# ---------- HUGGING FACE HELPERS ----------
def call_hf_image(prompt):
    """Call Hugging Face image generation API."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": prompt, "options": {"wait_for_model": True}}

    for attempt in range(HF_RETRIES):
        print(f"[HF-IMAGE] Attempt {attempt+1}/{HF_RETRIES}")
        resp = requests.post(url, json=data, headers=headers, timeout=HF_TIMEOUT)
        if resp.status_code == 200:
            ctype = resp.headers.get("content-type", "")
            if "image" in ctype:
                return [save_bytes_and_get_url(resp.content, ctype, "png")]
            try:
                j = resp.json()
                if isinstance(j, dict) and "image" in j:
                    return [save_base64_and_return_url(j["image"])]
            except Exception:
                pass
        elif resp.status_code in (202, 429, 503):
            time.sleep(3)
            continue
        else:
            raise RuntimeError(f"Hugging Face Image error {resp.status_code}: {resp.text}")
    raise RuntimeError("Hugging Face Image failed after retries")


def call_hf_video(prompt):
    """Call Hugging Face text-to-video model."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    url = f"https://api-inference.huggingface.co/models/{HF_VIDEO_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": prompt, "options": {"wait_for_model": True}}

    for attempt in range(HF_RETRIES):
        print(f"[HF-VIDEO] Attempt {attempt+1}/{HF_RETRIES}")
        resp = requests.post(url, json=data, headers=headers, timeout=HF_TIMEOUT)
        if resp.status_code == 200:
            ctype = resp.headers.get("content-type", "")
            if "video" in ctype or "octet-stream" in ctype:
                return [save_bytes_and_get_url(resp.content, ctype, "mp4")]
            try:
                j = resp.json()
                if isinstance(j, dict):
                    for k in ("b64_video", "video", "data"):
                        if k in j and isinstance(j[k], str):
                            return [save_base64_and_return_url(j[k])]
            except Exception:
                pass
        elif resp.status_code in (202, 429, 503):
            time.sleep(3)
            continue
        else:
            raise RuntimeError(f"Hugging Face Video error {resp.status_code}: {resp.text}")
    raise RuntimeError("Hugging Face Video failed after retries")


# ---------- ROUTES ----------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            message = data.get("message", "").strip()
        else:
            message = request.form.get("message", "").strip()

        if not message:
            return jsonify({"error": "missing_message"}), 400

        reply = f"You said: {message}. (Bhaaratai backend is connected and replying successfully!)"
        return jsonify({"response": reply, "timestamp": datetime.datetime.utcnow().isoformat()})
    except Exception as e:
        print("Chat route error:", e)
        return jsonify({"error": "internal_error", "details": str(e)}), 500


@app.route("/generate-image", methods=["POST"])
def generate_image():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "missing_prompt"}), 400
    try:
        images = call_hf_image(prompt)
        urls = [ensure_absolute_url(u) for u in images if u]
        return jsonify({"images": urls, "provider": "huggingface"})
    except Exception as e:
        print("Image generation error:", e)
        return jsonify({"error": "image_generation_failed", "details": str(e)}), 500


@app.route("/generate-video", methods=["POST"])
def generate_video():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "missing_prompt"}), 400
    try:
        videos = call_hf_video(prompt)
        urls = [ensure_absolute_url(u) for u in videos if u]
        return jsonify({"videos": urls, "provider": "huggingface"})
    except Exception as e:
        print("Video generation error:", e)
        return jsonify({"error": "video_generation_failed", "details": str(e)}), 500


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] HF_IMAGE_MODEL={HF_IMAGE_MODEL}, HF_VIDEO_MODEL={HF_VIDEO_MODEL}, HF_TOKEN={bool(HF_API_TOKEN)}")
    app.run(host="0.0.0.0", port=port, debug=True)
