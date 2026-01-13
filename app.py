# app.py - Final ready-to-deploy backend for BharatAI (FULL VERSION)
import os
import uuid
import datetime
import traceback
import requests
import time
import re
import base64
import json
import sqlite3
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS

# ---------- CONFIG ----------
app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

DATABASE = os.path.join(os.getcwd(), "codes.db")

FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000",
    "https://webtoolslive.com",
    "https://www.webtoolslive.com",
]

CORS(app, origins=FRONTEND_ORIGINS, supports_credentials=False)

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    response.headers["Access-Control-Allow-Origin"] = origin if origin in FRONTEND_ORIGINS else "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, apikey"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ---------- ENV ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

# ---------- DATABASE ----------
def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(DATABASE, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception=None):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()

with app.app_context():
    db = get_db()
    db.execute("""
    CREATE TABLE IF NOT EXISTS codes (
        id TEXT PRIMARY KEY,
        code TEXT UNIQUE,
        note TEXT,
        published_by TEXT,
        published_at TEXT,
        usage_count INTEGER DEFAULT 0,
        max_uses INTEGER DEFAULT 4,
        valid INTEGER DEFAULT 1
    )
    """)
    db.commit()

# ---------- FILE HELPERS ----------
def save_bytes_and_get_url(b: bytes, ext="png"):
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(b)
    return f"{request.host_url.rstrip('/')}/uploads/{fname}"

def save_base64_and_return_url(b64):
    try:
        b = base64.b64decode(b64)
        return save_bytes_and_get_url(b)
    except Exception:
        return None

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# ---------- IMAGE PROMPT ENHANCERS ----------
STYLE_PRESETS = {
    "anime": "anime style, vibrant colors, clean line art",
    "cartoon": "cartoon style, bold outlines, pixar look",
    "realistic": "photorealistic, cinematic lighting, ultra detail"
}

SIZE_PRESETS = {
    "1:1": "square composition, centered subject",
    "9:16": "portrait composition, vertical framing",
    "16:9": "landscape composition, wide cinematic framing"
}

def build_image_prompt(prompt, style=None, size=None):
    parts = [prompt]
    if style in STYLE_PRESETS:
        parts.append(STYLE_PRESETS[style])
    if size in SIZE_PRESETS:
        parts.append(SIZE_PRESETS[size])
    parts.append("high quality, sharp focus, no blur")
    return ", ".join(parts)

# ---------- HF HELPERS ----------
def hf_post_with_backoff(url, headers, payload):
    for attempt in range(HF_RETRIES):
        try:
            return requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("HF request failed after retries")

def call_hf_image(prompt, model):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN missing")

    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json"
    }

    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(resp.text)

    # 🔥 SD-3 returns JSON
    data = resp.json()

    # Try common SD-3 formats
    for key in ("image", "b64", "b64_json", "data"):
        if key in data:
            return [save_base64_and_return_url(data[key])]

    raise RuntimeError("No image found in HF response")


def call_hf_video(prompt, model):
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = hf_post_with_backoff(url, headers, payload)
    if resp.status_code == 200:
        return [save_bytes_and_get_url(resp.content, "mp4")]
    raise RuntimeError(resp.text)

# ---------- ROUTES ----------
@app.route("/")
def home():
    return jsonify({"status": "BharatAI backend running ✅"})

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.json or {}).get("message")
    if not msg:
        return jsonify({"error": "missing_message"}), 400

    if TOGETHER_API_KEY:
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "messages": [{"role": "user", "content": msg}]
        }
        res = requests.post(
            TOGETHER_URL,
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
            json=payload
        )
        return jsonify({"response": res.json()["choices"][0]["message"]["content"]})

    return jsonify({"response": msg})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    body = request.get_json() or {}
    prompt = body.get("prompt") or body.get("text")
    style = body.get("style")
    size = body.get("size")

    if not prompt:
        return jsonify({"error": "missing_prompt"}), 400

    final_prompt = build_image_prompt(prompt, style, size)
    images = call_hf_image(final_prompt, HF_IMAGE_MODEL)

    return jsonify({
        "images": images,
        "meta": {"style": style, "size": size, "prompt": final_prompt}
    })

@app.route("/generate-video", methods=["POST"])
def generate_video():
    body = request.get_json() or {}
    prompt = body.get("prompt") or body.get("text")
    if not prompt:
        return jsonify({"error": "missing_prompt"}), 400
    return jsonify({"videos": call_hf_video(prompt, HF_VIDEO_MODEL)})

# ---------- ENTRY ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("BharatAI backend started 🚀")
    app.run(host="0.0.0.0", port=port)
