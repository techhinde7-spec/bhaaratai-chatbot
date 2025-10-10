# app.py — Final clean version for Bhaaratai backend
# Routes: /, /health, /chat, /generate-image, /generate-video
# Hugging Face integration (image + video), file saving, CORS

import os
import uuid
import datetime
import traceback
import requests
import time
import re
import base64
import json
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

# ---------- APP & CONFIG ----------
app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Allowed frontend origins — add any additional domains you host frontends on
FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000"
]

# CORS: restrict to FRONTEND_ORIGINS but allow preflight
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGINS}}, supports_credentials=False)

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # If you prefer strict, switch to: response.headers["Access-Control-Allow-Origin"] = "null"
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, apikey, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

# ---------- Basic routes ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Bhaaratai Backend Running ✅", "time": datetime.datetime.utcnow().isoformat()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()})

# ---------- Helpers to save files and produce absolute URLs ----------
def _base_url():
    # Prefer BACKEND_URL env if provided (useful on Render)
    env = os.environ.get("BACKEND_URL")
    if env:
        return env.rstrip("/")
    # request.host_url available when a request is active
    try:
        return (request.host_url or "").rstrip("/")
    except Exception:
        return ""

def save_bytes_and_get_url(b: bytes, content_type: str = None, ext_hint: str = None):
    ext = "bin"
    if ext_hint:
        ext = ext_hint.strip().lstrip(".")
    else:
        if content_type and "/" in content_type:
            ext = content_type.split("/")[1].split(";")[0]
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    with open(path, "wb") as f:
        f.write(b)
    base = _base_url()
    return f"{base}/uploads/{fname}"

def save_base64_and_return_url(b64_str: str):
    if not b64_str or not isinstance(b64_str, str):
        return None
    s = b64_str.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    s = s.replace("\n", "").replace("\r", "")
    try:
        raw = base64.b64decode(s)
    except Exception:
        return None
    # try to guess extension
    ext = "png"
    if raw[:8].startswith(b"\x89PNG"):
        ext = "png"
    elif raw[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif raw[:4] == b"RIFF":
        ext = "webp"
    return save_bytes_and_get_url(raw, ext_hint=ext)

def ensure_absolute_url(s: str):
    if not s or not isinstance(s, str):
        return None
    s2 = s.strip()
    if s2.startswith("data:"):
        return save_base64_and_return_url(s2)
    if re.match(r"^https?://", s2, re.I):
        return s2
    # large base64 string fallback
    if len(s2) > 60 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", s2):
        return save_base64_and_return_url(s2)
    return s2

@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# ---------- Provider config (env vars) ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # required for HF usage
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")  # optional (for chat LLM)

# ---------- Hugging Face helpers (robust) ----------
def call_hf_image(prompt: str, model: str = None, timeout: int = None, retries: int = None):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    model = model or HF_IMAGE_MODEL
    timeout = timeout or HF_TIMEOUT
    retries = (retries if retries is not None else HF_RETRIES)

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        # Request JSON response by default. HF may return JSON that includes base64 strings or URLs.
        "Accept": "application/json",
        "User-Agent": "Bhaaratai/Render"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    backoff = [1, 2, 4]
    last_exc = None

    for attempt in range(retries):
        try:
            print(f"[HF-IMAGE] POST {hf_url} attempt {attempt+1}/{retries}")
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            print(f"[HF-IMAGE] status={status} content-type={ctype}")

            if status == 200:
                # If the model server responded with an image directly (rare if Accept=application/json),
                # save the bytes
                if ctype.startswith("image/"):
                    return [save_bytes_and_get_url(resp.content, content_type=ctype)]
                # Try JSON parse
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    # If plain base64 returned
                    if isinstance(txt, str) and len(txt) > 50 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", txt):
                        return [save_base64_and_return_url(txt)]
                    return [txt]

                # Search JSON for common image fields
                candidates = []
                if isinstance(j, dict):
                    for k in ("b64_json", "image", "images", "data", "url", "generated_images"):
                        v = j.get(k)
                        if v:
                            if isinstance(v, str):
                                candidates.append(v)
                            elif isinstance(v, list):
                                for it in v:
                                    if isinstance(it, str):
                                        candidates.append(it)
                            elif isinstance(v, dict):
                                # nested patterns
                                for subk in ("b64_json", "url", "image"):
                                    sub = v.get(subk)
                                    if isinstance(sub, str):
                                        candidates.append(sub)
                elif isinstance(j, list):
                    for it in j:
                        if isinstance(it, str):
                            candidates.append(it)

                saved = []
                for c in candidates:
                    if not c:
                        continue
                    c = c.strip()
                    if re.match(r"^https?://", c, re.I):
                        saved.append(c)
                        continue
                    if c.startswith("data:"):
                        url = save_base64_and_return_url(c)
                        if url:
                            saved.append(url)
                            continue
                    # large base64
                    cleaned = c.replace("\n", "").replace("\r", "")
                    if len(cleaned) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
                        url = save_base64_and_return_url(cleaned)
                        if url:
                            saved.append(url)
                            continue
                if saved:
                    return saved

                # fallback: search any http urls in JSON text
                json_text = json.dumps(j)
                found_urls = re.findall(r"https?://[^\s'\"\]\}]+", json_text)
                if found_urls:
                    # unique
                    return list(dict.fromkeys(found_urls))

                # nothing useful found
                return [json_text]

            elif status in (202, 429, 503):
                # transient -> backoff & retry
                last_exc = RuntimeError(f"HF transient status {status}: {resp.text[:400]}")
                wait = backoff[min(attempt, len(backoff)-1)]
                time.sleep(wait)
                continue
            else:
                # treat as fatal
                try:
                    err_j = resp.json()
                except Exception:
                    err_j = resp.text
                raise RuntimeError(f"Hugging Face image error {status}: {err_j}")

        except requests.exceptions.RequestException as rexc:
            print("[HF-IMAGE] request exception:", rexc)
            traceback.print_exc()
            last_exc = rexc
            wait = backoff[min(attempt, len(backoff)-1)]
            time.sleep(wait)
            continue
        except Exception as exc:
            print("[HF-IMAGE] unexpected:", exc)
            traceback.print_exc()
            last_exc = exc
            break

    raise RuntimeError(f"Hugging Face image call failed after {retries} attempts: {last_exc}")

def call_hf_video(prompt: str, model: str = None, timeout: int = None, retries: int = None):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    model = model or HF_VIDEO_MODEL
    timeout = timeout or HF_TIMEOUT
    retries = (retries if retries is not None else HF_RETRIES)

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "Bhaaratai/Render"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    backoff = [1, 2, 4]
    last_exc = None

    for attempt in range(retries):
        try:
            print(f"[HF-VIDEO] POST {hf_url} attempt {attempt+1}/{retries}")
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            print(f"[HF-VIDEO] status={status} content-type={ctype}")

            if status == 200:
                if ctype.startswith("video/") or "octet-stream" in ctype:
                    return [save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint="mp4")]
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    if isinstance(txt, str) and len(txt) > 200 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", txt):
                        return [save_base64_and_return_url(txt)]
                    return [txt]

                candidates = []
                if isinstance(j, dict):
                    for k in ("b64_video", "video", "videos", "urls", "data", "url"):
                        v = j.get(k)
                        if v:
                            if isinstance(v, str):
                                candidates.append(v)
                            elif isinstance(v, list):
                                for it in v:
                                    if isinstance(it, str):
                                        candidates.append(it)
                out = []
                for c in candidates:
                    if not c:
                        continue
                    if c.startswith("data:"):
                        saved = save_base64_and_return_url(c)
                        if saved:
                            out.append(saved)
                            continue
                    if re.match(r"^https?://", c, re.I):
                        out.append(c)
                        continue
                    cleaned = c.replace("\n", "").replace("\r", "")
                    if len(cleaned) > 100 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
                        saved = save_base64_and_return_url(cleaned)
                        if saved:
                            out.append(saved)
                            continue
                if out:
                    return out

                found_urls = re.findall(r"https?://[^\s'\"\]\}]+", json.dumps(j))
                if found_urls:
                    return list(dict.fromkeys(found_urls))

                return [json.dumps(j)]

            elif status in (202, 429, 503):
                last_exc = RuntimeError(f"HF transient status {status}: {resp.text[:400]}")
                wait = backoff[min(attempt, len(backoff)-1)]
                time.sleep(wait)
                continue
            else:
                try:
                    err_j = resp.json()
                except Exception:
                    err_j = resp.text
                raise RuntimeError(f"Hugging Face video error {status}: {err_j}")

        except requests.exceptions.RequestException as rexc:
            print("[HF-VIDEO] request exception:", rexc)
            traceback.print_exc()
            last_exc = rexc
            wait = backoff[min(attempt, len(backoff)-1)]
            time.sleep(wait)
            continue
        except Exception as exc:
            print("[HF-VIDEO] unexpected:", exc)
            traceback.print_exc()
            last_exc = exc
            break

    raise RuntimeError(f"Hugging Face video call failed after {retries} attempts: {last_exc}")

# ---------- Chat route ----------
@app.route("/chat", methods=["POST"])
def chat_route():
    try:
        # Accept JSON or form
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            message = (data.get("message") or data.get("text") or "").strip()
        else:
            message = (request.form.get("message") or "").strip()

        if not message:
            return jsonify({"error": "missing_message"}), 400

        # If Together API key is configured, call it; otherwise return a helpful reply
        if TOGETHER_API_KEY:
            together_url = "https://api.together.xyz/v1/chat/completions"
            payload = {
                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7,
            }
            headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
            resp = requests.post(together_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            try:
                reply = j["choices"][0]["message"]["content"]
            except Exception:
                reply = j.get("choice", j.get("output", str(j)))
            return jsonify({"response": reply, "provider": "together", "timestamp": datetime.datetime.utcnow().isoformat()})
        else:
            # fallback echo + useful info (so frontend verifies backend connectivity)
            reply = f"You said: {message}. (Bhaaratai backend is connected and replying successfully!)"
            return jsonify({"response": reply, "provider": "echo", "timestamp": datetime.datetime.utcnow().isoformat()})

    except requests.exceptions.RequestException as rex:
        print("Together API request failed:", rex)
        traceback.print_exc()
        return jsonify({"error": "provider_error", "details": str(rex)}), 502
    except Exception as e:
        print("Chat route error:", e)
        traceback.print_exc()
        return jsonify({"error": "internal_error", "details": str(e)}), 500

# ---------- Image generation route ----------
@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        model = body.get("model") or None
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400
        imgs = call_hf_image(prompt, model=model)
        urls = [ensure_absolute_url(u) for u in imgs if u]
        return jsonify({"images": urls, "provider": "huggingface"})
    except Exception as e:
        print("generate-image error:", e)
        traceback.print_exc()
        return jsonify({"error": "image_generation_failed", "details": str(e)}), 500

# ---------- Video generation route ----------
@app.route("/generate-video", methods=["POST"])
def generate_video_route():
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        model = body.get("model") or None
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400
        vids = call_hf_video(prompt, model=model)
        urls = [ensure_absolute_url(u) for u in vids if u]
        return jsonify({"videos": urls, "provider": "huggingface"})
    except Exception as e:
        print("generate-video error:", e)
        traceback.print_exc()
        return jsonify({"error": "video_generation_failed", "details": str(e)}), 500

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] HF_IMAGE_MODEL={HF_IMAGE_MODEL} HF_VIDEO_MODEL={HF_VIDEO_MODEL} HF_TOKEN_SET={bool(HF_API_TOKEN)}")
    app.run(host="0.0.0.0", port=port, debug=True)
