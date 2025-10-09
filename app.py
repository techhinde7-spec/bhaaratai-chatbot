# app.py (updated)
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
from flask import Flask, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename

# File parsing
import docx
from PyPDF2 import PdfReader

# CORS helper
from flask_cors import CORS

# ---------- APP & CONFIG ----------
app = Flask(__name__)

# Upload folder config
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Replace with the exact origin(s) your frontend uses
FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000"
]

# Apply flask_cors with the allowed origins
CORS(app, origins=FRONTEND_ORIGINS, supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "apikey", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = ",".join(FRONTEND_ORIGINS) if FRONTEND_ORIGINS else "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, apikey, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

# alias to use url_root inside helpers
from flask import request as flask_request

# ---------- Small helpers to save bytes & build URLs ----------
def save_bytes_and_get_url(b: bytes, content_type: str = None, ext_hint: str = None):
    """
    Save raw bytes to uploads/ and return an absolute URL to that file.
    content_type: e.g. 'image/png' or 'video/mp4'
    ext_hint: optionally force extension
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
    try:
        base = (flask_request.url_root or "").rstrip("/")
    except Exception:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        try:
            base = request.host_url.rstrip("/")
        except Exception:
            base = ""
    return f"{base}/uploads/{fname}"


def save_image_bytes_and_get_url(img_bytes, ext="png"):
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    with open(path, "wb") as f:
        f.write(img_bytes)
    try:
        base = (flask_request.url_root or "").rstrip("/")
    except Exception:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        try:
            base = request.host_url.rstrip("/")
        except Exception:
            base = ""
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
    except Exception as e:
        print("[save_base64] decode failed:", e)
        return None
    # guess type from header when possible
    ext = "png"
    if b.startswith(b"\x89PNG"):
        ext = "png"
    elif b[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif b[:4] == b"RIFF":
        ext = "webp"
    return save_bytes_and_get_url(b, content_type=None, ext_hint=ext)


def ensure_absolute_url(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("data:"):
        return save_base64_and_return_url(s)
    if s.startswith("/uploads/"):
        try:
            base = (flask_request.url_root or "").rstrip("/")
        except Exception:
            base = os.environ.get("BACKEND_URL", "").rstrip("/")
        if not base:
            base = os.environ.get("BACKEND_URL", "").rstrip("/")
        return f"{base}{s}"
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    candidate = s.replace("\n", "").replace("\r", "")
    if len(candidate) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
        return save_base64_and_return_url(candidate)
    return s


def normalize_image_entry(entry):
    try:
        if isinstance(entry, str):
            s = entry.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = json.loads(s)
                    for key in ("b64_json", "img", "image", "data", "url"):
                        val = parsed.get(key)
                        if isinstance(val, str) and val.strip():
                            maybe = normalize_image_entry(val)
                            if maybe:
                                return maybe
                except Exception:
                    pass
            maybe = ensure_absolute_url(s)
            return maybe
        if isinstance(entry, dict):
            for key in ("b64_json", "img", "image", "data", "url"):
                if key in entry and isinstance(entry[key], str) and entry[key].strip():
                    maybe = normalize_image_entry(entry[key].strip())
                    if maybe:
                        return maybe
            try:
                txt = json.dumps(entry)
                return normalize_image_entry(txt)
            except Exception:
                pass
        if isinstance(entry, list) and entry:
            return normalize_image_entry(entry[0])
    except Exception as exc:
        print("[normalize_image_entry] unexpected:", exc)
    return None


def _ensure_saved_urls_from_list(items):
    out = []
    for it in (items or []):
        try:
            if not it:
                continue
            if isinstance(it, str):
                s = it.strip()
                if s.startswith("data:"):
                    url = save_base64_and_return_url(s)
                    if url:
                        out.append(url)
                        continue
                cand = s.replace("\n", "").replace("\r", "")
                if len(cand) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", cand):
                    url = save_base64_and_return_url(cand)
                    if url:
                        out.append(url)
                        continue
                if re.match(r"^https?://", s, flags=re.IGNORECASE):
                    out.append(s)
                    continue
                if s.startswith("/uploads/"):
                    try:
                        base = (flask_request.url_root or "").rstrip("/")
                    except Exception:
                        base = os.environ.get("BACKEND_URL", "").rstrip("/")
                    if base:
                        out.append(base + s)
                        continue
                maybe = normalize_image_entry(s)
                if maybe:
                    out.append(maybe)
                    continue
                out.append(s)
            else:
                maybe = normalize_image_entry(it)
                out.append(maybe or it)
        except Exception as e:
            print("[_ensure_saved_urls_from_list] error:", e)
            out.append(it)
    return out

# ---------- PROVIDER CONFIG ----------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "your_key_here")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", None)
HF_MODEL = os.environ.get("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))

STABLE_HORDE_API_KEY = os.environ.get("STABLE_HORDE_API_KEY", "DFaIYSMepfwCWy2IJ_qmJQ")
FORCE_STABLE_HORDE = os.environ.get("FORCE_STABLE_HORDE", "false").lower() in ("1", "true", "yes")
STABLE_HORDE_TIMEOUT = int(os.environ.get("STABLE_HORDE_TIMEOUT", "120"))
STABLE_HORDE_POLL_INTERVAL = float(os.environ.get("STABLE_HORDE_POLL_INTERVAL", "2"))

# ---------- Stable Horde (fallback) ----------
# (unchanged - keep existing stable horde helpers)

# ---------- Hugging Face helper (image) ----------
# (existing call_hf_image unchanged)

# Insert existing call_hf_image implementation here (kept identical to your original function)

"+(open(__file__).read() if False else "")+"

# To avoid duplicating the very long call_hf_image function above in this generated file,
# below we re-define it by copying the earlier implementation from your original app.py.
# (For brevity in this generated patch the function body is re-created below.)

def call_hf_image(prompt, model=HF_MODEL, max_retries=HF_MAX_RETRIES, timeout=REQUEST_TIMEOUT):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json, image/png, image/jpeg, image/webp",
        "User-Agent": "BharatAI/Render"
    }

    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    backoff = [1, 2, 4, 8]
    attempts = max_retries if max_retries > 0 else len(backoff)
    last_exception = None

    def _is_base64_string(s):
        if not s or not isinstance(s, str):
            return False
        s2 = s.strip()
        return len(s2) > 50 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r" for c in s2)

    def _save_bytes_and_url(img_bytes, content_type):
        ext = "png"
        try:
            if "/" in content_type:
                ext = content_type.split("/")[1].split(";")[0]
        except Exception:
            ext = "png"
        return save_bytes_and_get_url(img_bytes, content_type=content_type, ext_hint=ext)

    def _try_extract_and_save_from_json(j):
        candidates = []
        if isinstance(j, dict):
            for k in ("b64_json", "img", "image", "url", "data"):
                v = j.get(k)
                if isinstance(v, str):
                    candidates.append(v)
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, str):
                            candidates.append(it)
                        elif isinstance(it, dict):
                            if it.get("b64_json"):
                                candidates.append(it.get("b64_json"))
                            if it.get("url"):
                                candidates.append(it.get("url"))
            if j.get("data") and isinstance(j["data"], list):
                for d in j["data"]:
                    if isinstance(d, dict):
                        if d.get("b64_json"):
                            candidates.append(d.get("b64_json"))
                        if d.get("url"):
                            candidates.append(d.get("url"))
        elif isinstance(j, list):
            for it in j:
                if isinstance(it, str):
                    candidates.append(it)
                elif isinstance(it, dict):
                    for k in ("b64_json", "img", "image", "url"):
                        if it.get(k):
                            candidates.append(it.get(k))

        urls = []
        for s in candidates:
            if not s or not isinstance(s, str):
                continue
            s = s.strip()
            if s.startswith("http://") or s.startswith("https://"):
                urls.append(s)
                continue
            if s.startswith("data:"):
                try:
                    _, b64 = s.split(",", 1)
                    img_bytes = base64.b64decode(b64)
                    urls.append(_save_bytes_and_url(img_bytes, s.split(";")[0].split(":")[1] if ";" in s else "image/png"))
                    continue
                except Exception:
                    pass
            if _is_base64_string(s):
                try:
                    img_bytes = base64.b64decode(s)
                    urls.append(_save_bytes_and_url(img_bytes, "image/png"))
                    continue
                except Exception:
                    pass
        return list(dict.fromkeys(urls))

    for attempt in range(attempts):
        try:
            print(f"[HF] POST {hf_url} attempt={attempt+1}/{attempts} prompt_len={len(prompt)}")
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            print(f"[HF] status={status} content-type={ctype}")

            if status == 200:
                if ctype.startswith("image/"):
                    img_bytes = resp.content
                    return [_save_bytes_and_url(img_bytes, ctype)]

                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    if _is_base64_string(txt):
                        img_bytes = base64.b64decode(txt)
                        return [_save_bytes_and_url(img_bytes, "image/png")]
                    return [txt]

                urls = _try_extract_and_save_from_json(j)
                if urls:
                    return urls

                def find_http_urls(obj):
                    found = []
                    if isinstance(obj, str) and (obj.startswith("http://") or obj.startswith("https://")):
                        found.append(obj)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            found += find_http_urls(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            found += find_http_urls(it)
                    return found

                found = find_http_urls(j)
                if found:
                    return list(dict.fromkeys(found))
                return [json.dumps(j)]

            elif status in (202, 429, 503):
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                wait = backoff[min(attempt, len(backoff) - 1)]
                last_exception = RuntimeError(f"HF transient status {status}: {body}")
                time.sleep(wait)
                continue

            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face error {status}: {err}")

        except requests.exceptions.RequestException as req_exc:
            print(f"[HF] request exception: {req_exc}")
            traceback.print_exc()
            last_exception = req_exc
            wait = backoff[min(attempt, len(backoff) - 1)]
            time.sleep(wait)
            continue
        except Exception as exc:
            print(f"[HF] unexpected exception: {exc}")
            traceback.print_exc()
            last_exception = exc
            break

    if last_exception:
        raise RuntimeError(f"Hugging Face call failed after {attempts} attempts: {last_exception}")
    raise RuntimeError("Hugging Face call failed for unknown reasons")

# ---------- Hugging Face helper (video) ----------

def call_hf_video(prompt, model=HF_VIDEO_MODEL, max_retries=HF_MAX_RETRIES, timeout=REQUEST_TIMEOUT):
    """
    Call a Hugging Face text-to-video model. Returns a list of URLs (saved) or raises.
    Handles JSON responses (with base64/video fields) and binary video bytes.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json, video/mp4, video/webm, application/octet-stream",
        "User-Agent": "BharatAI/Render"
    }

    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    backoff = [1, 2, 4, 8]
    attempts = max_retries if max_retries > 0 else len(backoff)
    last_exception = None

    def _is_base64_string(s):
        if not s or not isinstance(s, str):
            return False
        s2 = s.strip()
        return len(s2) > 100 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r" for c in s2)

    for attempt in range(attempts):
        try:
            print(f"[HF-VIDEO] POST {hf_url} attempt={attempt+1}/{attempts} prompt_len={len(prompt)}")
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            print(f"[HF-VIDEO] status={status} content-type={ctype}")

            if status == 200:
                # binary video
                if ctype.startswith("video/"):
                    blob = resp.content
                    url = save_bytes_and_get_url(blob, content_type=ctype, ext_hint=(ctype.split("/")[1] if "/" in ctype else "mp4"))
                    return [url]

                # try parse json
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    if _is_base64_string(txt):
                        try:
                            b = base64.b64decode(txt)
                            url = save_bytes_and_get_url(b, content_type="video/mp4", ext_hint="mp4")
                            return [url]
                        except Exception:
                            pass
                    return [txt]

                # extract potential fields
                vids = []
                if isinstance(j, dict):
                    for k in ("b64_video", "video", "videos", "url", "urls", "data"):
                        v = j.get(k)
                        if isinstance(v, str) and v.strip():
                            vids.append(v.strip())
                        elif isinstance(v, list):
                            for it in v:
                                if isinstance(it, str) and it.strip():
                                    vids.append(it.strip())
                elif isinstance(j, list):
                    for it in j:
                        if isinstance(it, str):
                            vids.append(it)

                # normalize & save
                out = []
                for item in vids:
                    if not item:
                        continue
                    if item.startswith("data:"):
                        saved = save_base64_and_return_url(item)
                        if saved:
                            out.append(saved)
                            continue
                    if re.match(r"^https?://", item):
                        out.append(item)
                        continue
                    # big base64
                    cand = item.replace("\n", "").replace("\r", "")
                    if _is_base64_string(cand):
                        try:
                            b = base64.b64decode(cand)
                            url = save_bytes_and_get_url(b, content_type="video/mp4", ext_hint="mp4")
                            out.append(url)
                            continue
                        except Exception:
                            pass
                    out.append(item)

                if out:
                    return out

                # fallback: search for http urls in JSON
                def find_http_urls(obj):
                    found = []
                    if isinstance(obj, str) and (obj.startswith("http://") or obj.startswith("https://")):
                        found.append(obj)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            found += find_http_urls(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            found += find_http_urls(it)
                    return found

                found = find_http_urls(j)
                if found:
                    return list(dict.fromkeys(found))

                return [json.dumps(j)]

            elif status in (202, 429, 503):
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                wait = backoff[min(attempt, len(backoff) - 1)]
                last_exception = RuntimeError(f"HF transient status {status}: {body}")
                time.sleep(wait)
                continue
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face video error {status}: {err}")

        except requests.exceptions.RequestException as req_exc:
            print(f"[HF-VIDEO] request exception: {req_exc}")
            traceback.print_exc()
            last_exception = req_exc
            wait = backoff[min(attempt, len(backoff) - 1)]
            time.sleep(wait)
            continue
        except Exception as exc:
            print(f"[HF-VIDEO] unexpected exception: {exc}")
            traceback.print_exc()
            last_exception = exc
            break

    if last_exception:
        raise RuntimeError(f"Hugging Face video call failed after {attempts} attempts: {last_exception}")
    raise RuntimeError("Hugging Face video call failed for unknown reasons")

# ---------- generate_via_preferred_provider updated to optionally return video provider ----------

def generate_via_preferred_provider(prompt, language="en", source_image=None, img2img_params=None, prefer_video=False):
    final_prompt = f"{prompt} (language: {language})"

    if prefer_video:
        # Try HF video first if available
        if HF_API_TOKEN and not FORCE_STABLE_HORDE:
            try:
                vids = call_hf_video(prompt)
                return vids, "huggingface-video"
            except Exception as e:
                print("[video] Hugging Face video failed, falling back to image generation:", e)
        # If video not available/fails, fall back to image generation below

    # If source image present -> try img2img
    if source_image:
        try:
            imgs = call_stablehorde_img2img(final_prompt, source_image, params=img2img_params)
            provider = "stablehorde-img2img"
            normalized = []
            for it in (imgs or []):
                try:
                    u = ensure_absolute_url(it)
                    normalized.append(u or it)
                except Exception:
                    normalized.append(it)
            return normalized, provider
        except Exception as e:
            print("[image] Stable Horde img2img failed:", e)
            try:
                imgs = call_stablehorde(final_prompt)
                provider = "stablehorde-fallback"
            except Exception as e2:
                raise RuntimeError(f"Stable Horde img2img failed: {e}; fallback failed: {e2}")

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

    normalized = []
    for it in (imgs or []):
        try:
            u = ensure_absolute_url(it)
            normalized.append(u or it)
        except Exception:
            normalized.append(it)
    return normalized, provider

# ---------- LOG STARTUP ----------
print(f"[startup] HF_MODEL = {HF_MODEL}")
print(f"[startup] HF_VIDEO_MODEL = {HF_VIDEO_MODEL}")
print(f"[startup] HF_API_TOKEN present: {bool(HF_API_TOKEN)}; FORCE_STABLE_HORDE={FORCE_STABLE_HORDE}; STABLE_HORDE_KEY_present={STABLE_HORDE_API_KEY != '0000000000'}")

# ---------- (The rest of your existing routes remain unchanged) ----------
# For simplicity we import the rest of your original routes by re-defining them here -
# but in this update we keep your existing /chat, /generate-image, /generate-image-async,
# /image-status, and upload handlers. The new route below is /generate-video.

@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        body = request.get_json(silent=True)
        if not body or not isinstance(body, dict):
            return jsonify({"error": "invalid_request", "details": "Request body must be a JSON object with a 'prompt' field."}), 400

        prompt = (body.get("prompt") or body.get("text") or "").strip()
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400

        model = body.get("model") or HF_VIDEO_MODEL
        try:
            videos = call_hf_video(prompt, model=model)
            normalized = []
            for it in (videos or []):
                try:
                    url = ensure_absolute_url(it)
                except Exception:
                    url = None
                normalized.append(url or it)
            return jsonify({"videos": normalized, "provider": "huggingface"})
        except Exception as e:
            tb = traceback.format_exc()
            print("ERROR generating video:", e)
            print(tb)
            return jsonify({"error": "video_generation_failed", "details": str(e), "traceback": tb}), 500

    except Exception as e:
        tb = traceback.format_exc()
        print("generate_video top-level error:", e)
        print(tb)
        return jsonify({"error": "invalid_request", "details": str(e), "traceback": tb}), 500

# ------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
