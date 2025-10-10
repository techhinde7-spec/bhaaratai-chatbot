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
        # permissive fallback for testing — in prod prefer explicit origins
        response.headers["Access-Control-Allow-Origin"] = ",".join(FRONTEND_ORIGINS) if FRONTEND_ORIGINS else "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, apikey, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response


# ---------- BASIC ROUTES ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Bhaaratai Backend Running ✅"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()})


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)


# ---------- HELPER FUNCTIONS ----------
def _get_base_url():
    """Return a base URL to build absolute URLs for uploaded files."""
    try:
        base = request.host_url.rstrip("/")
    except Exception:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    return base


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
    base = _get_base_url()
    return f"{base}/uploads/{fname}"


def save_base64_and_return_url(b64_str):
    """Save base64 image/video and return a URL."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    s = b64_str.strip()
    if s.startswith("data:") and "," in s:
        # data:<mime>;base64,<b64>
        try:
            header, payload = s.split(",", 1)
        except Exception:
            payload = s
    else:
        payload = s
    payload = payload.replace("\n", "").replace("\r", "")
    try:
        b = base64.b64decode(payload)
    except Exception:
        return None
    # guess ext from header or bytes
    ext = "png"
    if b.startswith(b"\x89PNG"):
        ext = "png"
    elif b[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif b[:4] == b"RIFF":
        ext = "webp"
    elif b[:4] == b"PK\x03\x04":
        ext = "zip"
    return save_bytes_and_get_url(b, ext_hint=ext)


def ensure_absolute_url(s):
    """Normalize various server responses into a usable HTTP URL (or save base64 to disk)."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("data:"):
        return save_base64_and_return_url(s)
    if s.startswith("/uploads/"):
        base = _get_base_url()
        return f"{base}{s}"
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    # treat long base64 strings as base64 payloads
    cand = s.replace("\n", "").replace("\r", "")
    if len(cand) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", cand):
        return save_base64_and_return_url(cand)
    return s


# ---------- HUGGING FACE CONFIG ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))

# Optional Together API for chat
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")


# ---------- HUGGING FACE HELPERS ----------
def call_hf_image(prompt: str):
    """Call Hugging Face image generation API and return list of URLs or raise."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    hf_url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json, image/png, image/jpeg, image/webp"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    backoff = [1, 2, 4, 8]
    attempts = HF_RETRIES if HF_RETRIES > 0 else len(backoff)
    last_exc = None

    for attempt in range(attempts):
        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=HF_TIMEOUT)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if status == 200:
                # binary image returned
                if ctype.startswith("image/"):
                    url = save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint=(ctype.split("/")[1] if "/" in ctype else "png"))
                    return [url]
                # try parse json body for base64 or urls
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    # if raw base64
                    cleaned = txt.replace("\n", "").replace("\r", "")
                    if len(cleaned) > 100 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
                        url = save_base64_and_return_url(cleaned)
                        if url:
                            return [url]
                    return [txt]
                # search for image data inside JSON
                found = []
                def collect(obj):
                    if isinstance(obj, str):
                        s = obj.strip()
                        if s.startswith("http://") or s.startswith("https://"):
                            found.append(s)
                        elif s.startswith("data:"):
                            u = save_base64_and_return_url(s)
                            if u: found.append(u)
                        else:
                            c = s.replace("\n", "").replace("\r", "")
                            if len(c) > 100 and re.fullmatch(r"[A-Za-z0-9+/=]+", c):
                                u = save_base64_and_return_url(c)
                                if u: found.append(u)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            collect(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            collect(it)
                collect(j)
                found = list(dict.fromkeys(found))
                if found:
                    return found
                return [json.dumps(j)]
            elif status in (202, 429, 503):
                last_exc = RuntimeError(f"HF transient status {status}: {resp.text[:200]}")
                time.sleep(backoff[min(attempt, len(backoff)-1)])
                continue
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face image error {status}: {err}")
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff[min(attempt, len(backoff)-1)])
            continue

    raise RuntimeError(f"Hugging Face image call failed after {attempts} attempts: {last_exc}")


def call_hf_video(prompt: str):
    """Call Hugging Face text-to-video model and return list of URLs or raise."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    hf_url = f"https://api-inference.huggingface.co/models/{HF_VIDEO_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json, video/mp4, video/webm, application/octet-stream"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    backoff = [1, 2, 4, 8]
    attempts = HF_RETRIES if HF_RETRIES > 0 else len(backoff)
    last_exc = None

    for attempt in range(attempts):
        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=HF_TIMEOUT)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if status == 200:
                if ctype.startswith("video/") or "octet-stream" in ctype:
                    url = save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint=(ctype.split("/")[1] if "/" in ctype else "mp4"))
                    return [url]
                # try json
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    cleaned = txt.replace("\n", "").replace("\r", "")
                    if len(cleaned) > 200 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
                        u = save_base64_and_return_url(cleaned)
                        if u: return [u]
                    return [txt]
                # collect video fields if present
                found = []
                def collect(obj):
                    if isinstance(obj, str):
                        s = obj.strip()
                        if s.startswith("http://") or s.startswith("https://"):
                            found.append(s)
                        elif s.startswith("data:"):
                            u = save_base64_and_return_url(s)
                            if u: found.append(u)
                        else:
                            c = s.replace("\n", "").replace("\r", "")
                            if len(c) > 200 and re.fullmatch(r"[A-Za-z0-9+/=]+", c):
                                u = save_base64_and_return_url(c)
                                if u: found.append(u)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            collect(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            collect(it)
                collect(j)
                found = list(dict.fromkeys(found))
                if found:
                    return found
                return [json.dumps(j)]
            elif status in (202, 429, 503):
                last_exc = RuntimeError(f"HF transient status {status}: {resp.text[:200]}")
                time.sleep(backoff[min(attempt, len(backoff)-1)])
                continue
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face video error {status}: {err}")
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff[min(attempt, len(backoff)-1)])
            continue

    raise RuntimeError(f"Hugging Face video call failed after {attempts} attempts: {last_exc}")


# ---------- Chat helper (Together optional, HF fallback) ----------
def call_together_text(message: str, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0.7, max_tokens=512):
    """Call Together API for chat. Requires TOGETHER_API_KEY in env."""
    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY not set")
    payload = {"model": model, "messages": [{"role": "user", "content": message}], "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(TOGETHER_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Try to extract content robustly
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)


# ---------- ROUTES ----------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts JSON or form with "message".
    Uses Together API if configured, else tries Hugging Face text-inference if HF token present, else echoes.
    """
    try:
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            message = (data.get("message") or data.get("text") or "").strip()
        else:
            message = (request.form.get("message") or "").strip()

        if not message:
            return jsonify({"error": "missing_message"}), 400

        # Try Together
        if TOGETHER_API_KEY:
            try:
                reply = call_together_text(message)
                return jsonify({"response": reply, "provider": "together", "timestamp": datetime.datetime.utcnow().isoformat()})
            except Exception as e:
                print("Together API failed, falling back:", e)
                traceback.print_exc()

        # Try Hugging Face text inference (if configured)
        if HF_API_TOKEN:
            try:
                hf_text_model = os.environ.get("HF_TEXT_MODEL", None)  # optional override
                if not hf_text_model:
                    # fallback to a small default if you don't have a text model env
                    # NOTE: set HF_TEXT_MODEL env on Render if you want a specific text model
                    hf_text_model = os.environ.get("HF_MODEL", None)
                if not hf_text_model:
                    # no HF text model configured — skip to echo
                    raise RuntimeError("No HF_TEXT_MODEL configured")
                hf_url = f"https://api-inference.huggingface.co/models/{hf_text_model}"
                headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
                payload = {"inputs": message, "options": {"wait_for_model": True}}
                r = requests.post(hf_url, json=payload, headers=headers, timeout=HF_TIMEOUT)
                if r.status_code == 200:
                    try:
                        j = r.json()
                        # look for "generated_text" or similar
                        if isinstance(j, dict) and j.get("generated_text"):
                            reply = j.get("generated_text")
                        else:
                            # best-effort stringify
                            reply = j if isinstance(j, str) else json.dumps(j)
                    except Exception:
                        reply = r.text or ""
                    return jsonify({"response": reply, "provider": "huggingface-text", "timestamp": datetime.datetime.utcnow().isoformat()})
                else:
                    print("HF text model returned", r.status_code, r.text[:400])
            except Exception as e:
                print("HF text fallback failed:", e)
                traceback.print_exc()

        # Final fallback: echo
        return jsonify({"response": f"You said: {message}. (fallback echo)", "provider": "echo", "timestamp": datetime.datetime.utcnow().isoformat()})

    except Exception as e:
        print("Chat route error:", e)
        traceback.print_exc()
        return jsonify({"error": "internal_error", "details": str(e)}), 500


@app.route("/generate-image", methods=["POST"])
def generate_image():
    """
    Body JSON: { "prompt": "..." }
    Returns: { "images": [url1, url2], "provider": "huggingface" }
    """
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400

        if not HF_API_TOKEN:
            return jsonify({"error": "missing_hf_api_token"}), 500

        images = call_hf_image(prompt)
        urls = [ensure_absolute_url(u) for u in images if u]
        return jsonify({"images": urls, "provider": "huggingface"})
    except Exception as e:
        print("generate-image error:", e)
        traceback.print_exc()
        return jsonify({"error": "generation_failed", "details": str(e)}), 500


@app.route("/generate-video", methods=["POST"])
def generate_video():
    """
    Body JSON: { "prompt": "..." }
    Returns: { "videos": [url1, url2], "provider": "huggingface" }
    """
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400

        if not HF_API_TOKEN:
            return jsonify({"error": "missing_hf_api_token"}), 500

        videos = call_hf_video(prompt)
        urls = [ensure_absolute_url(u) for u in videos if u]
        return jsonify({"videos": urls, "provider": "huggingface"})
    except Exception as e:
        print("generate-video error:", e)
        traceback.print_exc()
        return jsonify({"error": "video_generation_failed", "details": str(e)}), 500


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] HF_IMAGE_MODEL={HF_IMAGE_MODEL}, HF_VIDEO_MODEL={HF_VIDEO_MODEL}, HF_TOKEN={bool(HF_API_TOKEN)}")
    app.run(host="0.0.0.0", port=port, debug=True)
