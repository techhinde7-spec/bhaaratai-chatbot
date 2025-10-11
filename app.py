# app.py - Final ready-to-deploy backend for BharatAI
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
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Frontend origins - keep your production origins here
FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000"
]

# CORS
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

# ---------- ENV / PROVIDER ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")

# ---------- UTILITIES ----------
def save_bytes_and_get_url(b: bytes, content_type: str = None, ext_hint: str = None):
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
    base = (request.host_url or "").rstrip("/")
    return f"{base}/uploads/{fname}"

def save_base64_and_return_url(b64_str):
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
    # basic type guess
    ext = "png"
    if b.startswith(b"\x89PNG"):
        ext = "png"
    elif b[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    return save_bytes_and_get_url(b, content_type=None, ext_hint=ext)

def ensure_absolute_url(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("data:"):
        return save_base64_and_return_url(s)
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    if len(s) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return save_base64_and_return_url(s)
    return s

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# ---------- HUGGING FACE HELPERS ----------
def call_hf_image(prompt, model=HF_IMAGE_MODEL, timeout=HF_TIMEOUT, retries=HF_RETRIES):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")
    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json, image/png, image/jpeg, image/webp"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    backoff = [1, 2, 4]
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if status == 200:
                # binary image
                if ctype.startswith("image/"):
                    return [save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint=ctype.split("/")[1])]
                # try parse json for base64/url
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    # maybe raw base64
                    if isinstance(txt, str) and len(txt) > 100:
                        return [save_base64_and_return_url(txt)]
                    return [txt]
                # try extract known keys
                for k in ("images", "image", "b64_json", "data", "url"):
                    if k in j:
                        val = j[k]
                        if isinstance(val, list):
                            out = []
                            for it in val:
                                if isinstance(it, str):
                                    out.append(ensure_absolute_url(it))
                            return out
                        if isinstance(val, str):
                            return [ensure_absolute_url(val)]
                # search for http urls or base64 inside json
                def find_urls(obj):
                    found = []
                    if isinstance(obj, str) and re.match(r"^https?://", obj):
                        found.append(obj)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            found += find_urls(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            found += find_urls(it)
                    return found
                found = find_urls(j)
                if found:
                    return list(dict.fromkeys(found))
                # fallback: stringify
                return [json.dumps(j)]
            elif status in (202, 429, 503):
                time.sleep(backoff[min(attempt, len(backoff)-1)])
                last_exc = RuntimeError(f"HF transient {status}: {resp.text[:200]}")
                continue
            else:
                # bubble HF errors up (401/404 will appear here)
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face image error {status}: {err}")
        except requests.RequestException as req_exc:
            last_exc = req_exc
            time.sleep(backoff[min(attempt, len(backoff)-1)])
            continue
    raise RuntimeError(f"Hugging Face image call failed after {retries} attempts: {last_exc}")

def call_hf_video(prompt, model=HF_VIDEO_MODEL, timeout=HF_TIMEOUT, retries=HF_RETRIES):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")
    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json, video/mp4, application/octet-stream"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    backoff = [1, 2, 4]
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if status == 200:
                if ctype.startswith("video/") or "octet-stream" in ctype:
                    return [save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint="mp4")]
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    # maybe raw base64
                    if isinstance(txt, str) and len(txt) > 200:
                        return [save_base64_and_return_url(txt)]
                    return [txt]
                # extract likely fields
                for k in ("b64_video", "video", "videos", "data", "url", "urls"):
                    if k in j:
                        val = j[k]
                        if isinstance(val, list):
                            out=[]
                            for it in val:
                                if isinstance(it, str):
                                    out.append(ensure_absolute_url(it))
                            return out
                        if isinstance(val, str):
                            return [ensure_absolute_url(val)]
                # search for http urls in JSON
                def find_http(obj):
                    found=[]
                    if isinstance(obj, str) and re.match(r"^https?://", obj):
                        found.append(obj)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            found += find_http(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            found += find_http(it)
                    return found
                found = find_http(j)
                if found:
                    return list(dict.fromkeys(found))
                return [json.dumps(j)]
            elif status in (202, 429, 503):
                time.sleep(backoff[min(attempt, len(backoff)-1)])
                last_exc = RuntimeError(f"HF transient {status}: {resp.text[:200]}")
                continue
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face video error {status}: {err}")
        except requests.RequestException as req_exc:
            last_exc = req_exc
            time.sleep(backoff[min(attempt, len(backoff)-1)])
            continue
    raise RuntimeError(f"Hugging Face video call failed after {retries} attempts: {last_exc}")

# ---------- ROUTES ----------
@app.route("/")
def home():
    return jsonify({"status": "Bhaaratai Backend Running ✅"})

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # accept json or form
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            message = (data.get("message") or data.get("text") or "").strip()
        else:
            message = (request.form.get("message") or "").strip()
        if not message:
            return jsonify({"error": "missing_message"}), 400

        # If Together API key is present, call it
        if TOGETHER_API_KEY:
            payload = {
                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7
            }
            headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
            try:
                res = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=60)
                res.raise_for_status()
                j = res.json()
                # defensive checks
                reply = None
                if isinstance(j, dict):
                    # typical shape: choices[0].message.content
                    choices = j.get("choices")
                    if choices and isinstance(choices, list):
                        c0 = choices[0]
                        if c0 and isinstance(c0, dict):
                            msg = c0.get("message") or c0.get("content")
                            if isinstance(msg, dict):
                                reply = msg.get("content") or msg.get("text")
                            elif isinstance(msg, str):
                                reply = msg
                if not reply:
                    # fallback shapes
                    reply = j.get("result") or j.get("output") or str(j)
                return jsonify({"response": reply, "timestamp": datetime.datetime.utcnow().isoformat()})
            except Exception as e:
                print("Together call error:", e)
                traceback.print_exc()
                # fall back to echo
        # fallback simple reply (useful during dev)
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
        if not prompt:
            return jsonify({"error":"missing_prompt"}), 400
        images = call_hf_image(prompt)
        urls = [ensure_absolute_url(u) for u in images if u]
        return jsonify({"images": urls, "provider": "huggingface"})
    except Exception as e:
        tb = traceback.format_exc()
        print("generate-image error:", e)
        print(tb)
        return jsonify({"error":"image_generation_failed", "details": str(e), "traceback": tb}), 500

@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        if not prompt:
            return jsonify({"error":"missing_prompt"}), 400
        videos = call_hf_video(prompt)
        urls = [ensure_absolute_url(u) for u in videos if u]
        return jsonify({"videos": urls, "provider": "huggingface"})
    except Exception as e:
        tb = traceback.format_exc()
        print("generate-video error:", e)
        print(tb)
        return jsonify({"error":"video_generation_failed", "details": str(e), "traceback": tb}), 500

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] HF_IMAGE_MODEL={HF_IMAGE_MODEL} HF_VIDEO_MODEL={HF_VIDEO_MODEL} HF_API_TOKEN_SET={bool(HF_API_TOKEN)}")
    app.run(host="0.0.0.0", port=port, debug=True)
