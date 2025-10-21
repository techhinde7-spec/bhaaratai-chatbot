# app.py - Final ready-to-deploy backend for BharatAI
# Routes: /, /health, /chat, /generate-image, /generate-video
# Hugging Face integration (image + video), proper CORS, save outputs to /uploads

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

# ---------- CONFIG ----------
app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150 MB

# Frontend origins - adjust if needed
FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000",
]

# CORS
CORS(app, origins=FRONTEND_ORIGINS, supports_credentials=False)

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

# ---------- ENV / PROVIDER ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
# default to the SD-3 model you asked about
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")

# ---------- UTILITIES ----------
def save_bytes_and_get_url(b: bytes, content_type: str = None, ext_hint: str = None):
    """
    Save raw bytes to uploads/ and return an absolute URL to that file.
    """
    ext = "bin"
    if ext_hint:
        ext = ext_hint
    else:
        try:
            if content_type and "/" in content_type:
                ext = content_type.split("/")[1].split(";")[0]
        except Exception:
            ext = "bin"
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    with open(path, "wb") as f:
        f.write(b)
    # build base from request host or BACKEND_URL env fallback
    base = ""
    try:
        base = (request.host_url or "").rstrip("/")
    except Exception:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    return f"{base}/uploads/{fname}"

def save_base64_and_return_url(b64_str):
    """
    Accepts data:... or raw base64, decodes and writes file, returns URL or None.
    """
    if not b64_str or not isinstance(b64_str, str):
        return None
    s = b64_str.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    s = s.replace("\n", "").replace("\r", "")
    try:
        b = base64.b64decode(s)
    except Exception:
        return None
    ext = "png"
    if b.startswith(b"\x89PNG"):
        ext = "png"
    elif b[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif b[:4] == b"RIFF":
        ext = "webp"
    return save_bytes_and_get_url(b, content_type=None, ext_hint=ext)

def ensure_absolute_url(s):
    """
    If s is a data URL or base64, save and return uploads URL.
    If s is already http(s) return it. Otherwise return original string.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("data:"):
        return save_base64_and_return_url(s)
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    # if it's probably a base64 string, save it
    candidate = s.replace("\n", "").replace("\r", "")
    if len(candidate) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
        return save_base64_and_return_url(candidate)
    return s

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# ---------- HF helpers ----------
def hf_post_with_backoff(url, headers=None, json_payload=None, timeout=HF_TIMEOUT, retries=HF_RETRIES):
    backoff = [1, 2, 4, 8]
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.post(url, headers=headers or {}, json=json_payload, timeout=timeout)
            return resp
        except requests.RequestException as e:
            last_exc = e
            wait = backoff[min(attempt, len(backoff)-1)]
            time.sleep(wait)
    raise last_exc or RuntimeError("HF request failed")

def call_hf_image(prompt, model=HF_IMAGE_MODEL, timeout=HF_TIMEOUT, retries=HF_RETRIES):
    """
    Call Hugging Face image model and return list of URLs or raise.
    Uses Accept: application/json to avoid HF accept-type errors.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")
    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "BhaarataiBackend/1.0"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = hf_post_with_backoff(hf_url, headers=headers, json_payload=payload, timeout=timeout, retries=retries)

    status = resp.status_code
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if status == 200:
        # binary case
        if ctype.startswith("image/"):
            ext = ctype.split("/")[1].split(";")[0] or "png"
            return [save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint=ext)]

        # expect JSON: try to parse for base64 or url
        try:
            j = resp.json()
        except Exception:
            txt = resp.text or ""
            if isinstance(txt, str) and len(txt) > 100:
                u = save_base64_and_return_url(txt)
                if u:
                    return [u]
            raise RuntimeError("HF returned non-image and non-json response")

        # Common keys to check
        # 1) direct URL(s)
        def collect_urls(obj):
            out = []
            if isinstance(obj, str) and obj.startswith("http"):
                out.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    out += collect_urls(v)
            elif isinstance(obj, list):
                for it in obj:
                    out += collect_urls(it)
            return out

        urls = collect_urls(j)
        if urls:
            return list(dict.fromkeys(urls))

        # 2) base64 fields
        for key in ("b64_json", "b64", "image", "data"):
            val = j.get(key) if isinstance(j, dict) else None
            if isinstance(val, str) and val.strip():
                saved = save_base64_and_return_url(val)
                if saved:
                    return [saved]
            if isinstance(val, list):
                out = []
                for it in val:
                    if isinstance(it, str):
                        maybe = ensure_absolute_url(it)
                        out.append(maybe)
                if out:
                    return out

        # 3) fallback: search entire JSON for base64-like string
        def find_base64(obj):
            if isinstance(obj, str):
                cand = obj.strip()
                if len(cand) > 100 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", cand):
                    return cand
                return None
            if isinstance(obj, dict):
                for v in obj.values():
                    x = find_base64(v)
                    if x:
                        return x
            if isinstance(obj, list):
                for it in obj:
                    x = find_base64(it)
                    if x:
                        return x
            return None

        cand = find_base64(j)
        if cand:
            saved = save_base64_and_return_url(cand)
            if saved:
                return [saved]

        # nothing found
        raise RuntimeError(f"Hugging Face returned JSON but no image fields. snippet: {json.dumps(j)[:1000]}")

    elif status in (202, 429, 503):
        raise RuntimeError(f"Hugging Face transient status {status}: {resp.text}")
    else:
        # propagate HF error body
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Hugging Face image error {status}: {err}")

def call_hf_video(prompt, model=HF_VIDEO_MODEL, timeout=HF_TIMEOUT, retries=HF_RETRIES):
    """
    Call Hugging Face video model and return list of URLs or raise.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")
    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "BhaarataiBackend/1.0"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = hf_post_with_backoff(hf_url, headers=headers, json_payload=payload, timeout=timeout, retries=retries)

    status = resp.status_code
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if status == 200:
        if ctype.startswith("video/") or "octet-stream" in ctype:
            ext = "mp4"
            return [save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint=ext)]

        try:
            j = resp.json()
        except Exception:
            txt = resp.text or ""
            if isinstance(txt, str) and len(txt) > 200:
                u = save_base64_and_return_url(txt)
                if u:
                    return [u]
            raise RuntimeError("HF returned non-video and non-json response")

        # collect http urls or base64
        def collect_urls(obj):
            out = []
            if isinstance(obj, str) and obj.startswith("http"):
                out.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    out += collect_urls(v)
            elif isinstance(obj, list):
                for it in obj:
                    out += collect_urls(it)
            return out

        urls = collect_urls(j)
        if urls:
            return list(dict.fromkeys(urls))

        # base64 search
        def find_base64(obj):
            if isinstance(obj, str):
                cand = obj.strip()
                if len(cand) > 200 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", cand):
                    return cand
                return None
            if isinstance(obj, dict):
                for v in obj.values():
                    x = find_base64(v)
                    if x:
                        return x
            if isinstance(obj, list):
                for it in obj:
                    x = find_base64(it)
                    if x:
                        return x
            return None

        cand = find_base64(j)
        if cand:
            saved = save_base64_and_return_url(cand)
            if saved:
                return [saved]

        raise RuntimeError(f"Hugging Face returned JSON but no video fields. snippet: {json.dumps(j)[:1200]}")

    elif status in (202, 429, 503):
        raise RuntimeError(f"Hugging Face transient status {status}: {resp.text}")
    else:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Hugging Face video error {status}: {err}")

# ---------- ROUTES ----------
@app.route("/")
def home():
    return jsonify({"status": "Bhaaratai Backend Running ✅", "time": datetime.datetime.utcnow().isoformat()})

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
            message = (data.get("message") or data.get("text") or "").strip()
        else:
            message = (request.form.get("message") or "").strip()

        if not message:
            return jsonify({"error": "missing_message"}), 400

        if TOGETHER_API_KEY:
            payload = {
                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7,
            }
            headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
            try:
                res = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=60)
                res.raise_for_status()
                jr = res.json()
                # defensive parsing
                try:
                    reply = jr["choices"][0]["message"]["content"]
                except Exception:
                    reply = jr.get("output") or jr.get("result") or str(jr)
                return jsonify({"response": reply, "timestamp": datetime.datetime.utcnow().isoformat()})
            except Exception as e:
                print("Together API error:", e)
                traceback.print_exc()
                # fallthrough to echo

        # fallback: simple echo (useful during dev)
        return jsonify({"response": f"You said: {message}. (Bhaaratai backend echo)", "timestamp": datetime.datetime.utcnow().isoformat()})
    except Exception as e:
        print("chat error:", e)
        traceback.print_exc()
        return jsonify({"error": "internal_error", "details": str(e)}), 500

@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        model = body.get("model") or HF_IMAGE_MODEL
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400
        images = call_hf_image(prompt, model=model)
        urls = [ensure_absolute_url(u) for u in images if u]
        return jsonify({"images": urls, "provider": "huggingface"})
    except Exception as e:
        tb = traceback.format_exc()
        print("generate-image error:", e)
        print(tb)
        return jsonify({"error": "image_generation_failed", "details": str(e), "traceback": tb}), 500

@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        model = body.get("model") or HF_VIDEO_MODEL
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400
        videos = call_hf_video(prompt, model=model)
        urls = [ensure_absolute_url(u) for u in videos if u]
        return jsonify({"videos": urls, "provider": "huggingface"})
    except Exception as e:
        tb = traceback.format_exc()
        print("generate-video error:", e)
        print(tb)
        return jsonify({"error": "video_generation_failed", "details": str(e), "traceback": tb}), 500

# ---------- ENTRYPOINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] HF_IMAGE_MODEL={HF_IMAGE_MODEL} HF_VIDEO_MODEL={HF_VIDEO_MODEL} HF_API_TOKEN_SET={bool(HF_API_TOKEN)}")
    app.run(host="0.0.0.0", port=port, debug=True)
