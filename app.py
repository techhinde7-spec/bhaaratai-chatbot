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

from flask_cors import CORS

# Replace with the exact origin(s) your frontend uses
FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in"
]

# Apply CORS to the whole app but restrict to allowed origins
CORS(app,
     origins=FRONTEND_ORIGINS,
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "apikey", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])


# alias to use url_root inside helpers
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
    try:
        base = (flask_request.url_root or "").rstrip("/")
    except Exception:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
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
        try:
            base = (flask_request.url_root or "").rstrip("/")
        except Exception:
            base = os.environ.get("BACKEND_URL", "").rstrip("/")
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

def normalize_image_entry(entry):
    """
    Accepts: string or dict or list. Returns an absolute URL the frontend can fetch,
    or None if it can't be normalized.
    """
    try:
        if isinstance(entry, str):
            s = entry.strip()
            # Try JSON parse if looks like JSON string
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = json.loads(s)
                    # prefer commonly used fields
                    for key in ("b64_json", "img", "image", "data", "url"):
                        val = parsed.get(key)
                        if isinstance(val, str) and val.strip():
                            maybe = normalize_image_entry(val)
                            if maybe:
                                return maybe
                    # scan values for long base64-like strings
                    for v in parsed.values():
                        if isinstance(v, str) and len(v) > 60 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", v.replace("\n","").replace("\r","")):
                            return save_base64_and_return_url(v)
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
            # nested dict -> stringify and try
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
    """
    Convert any data: or base64 entries in a provider response into saved file URLs.
    Returns list of absolute URLs (or original entries if not convertible).
    """
    out = []
    for it in (items or []):
        try:
            if not it:
                continue
            if isinstance(it, str):
                s = it.strip()
                # data URL
                if s.startswith("data:"):
                    url = save_base64_and_return_url(s)
                    if url:
                        out.append(url)
                        continue
                # raw base64 heuristic
                cand = s.replace("\n","").replace("\r","")
                if len(cand) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", cand):
                    url = save_base64_and_return_url(cand)
                    if url:
                        out.append(url)
                        continue
                # absolute http(s)
                if re.match(r"^https?://", s, flags=re.IGNORECASE):
                    out.append(s)
                    continue
                # relative uploads path
                if s.startswith("/uploads/"):
                    try:
                        base = (flask_request.url_root or "").rstrip("/")
                    except Exception:
                        base = os.environ.get("BACKEND_URL", "").rstrip("/")
                    if base:
                        out.append(base + s)
                        continue
                # try normalize (covers JSON-encoded strings, etc.)
                maybe = normalize_image_entry(s)
                if maybe:
                    out.append(maybe)
                    continue
                # fallback keep original
                out.append(s)
            else:
                # dict/list
                maybe = normalize_image_entry(it)
                out.append(maybe or it)
        except Exception as e:
            print("[_ensure_saved_urls_from_list] error:", e)
            out.append(it)
    return out

# ---------- CORS ----------
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
except Exception:
    pass

# ---------- PROVIDER CONFIG ----------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "your_key_here")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", None)
HF_MODEL = os.environ.get("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))

STABLE_HORDE_API_KEY = os.environ.get("STABLE_HORDE_API_KEY", "DFaIYSMepfwCWy2IJ_qmJQ")
FORCE_STABLE_HORDE = os.environ.get("FORCE_STABLE_HORDE", "false").lower() in ("1", "true", "yes")
STABLE_HORDE_TIMEOUT = int(os.environ.get("STABLE_HORDE_TIMEOUT", "120"))
STABLE_HORDE_POLL_INTERVAL = float(os.environ.get("STABLE_HORDE_POLL_INTERVAL", "2"))

# ---------- Stable Horde (fallback) ----------
def call_stablehorde(prompt, api_key=STABLE_HORDE_API_KEY, timeout=STABLE_HORDE_TIMEOUT, poll_interval=STABLE_HORDE_POLL_INTERVAL):
    """
    Submit async generation to Stable Horde (aihorde.net) and poll until done.
    Returns list of data URLs or http URLs.
    This is a blocking call (it submits then polls until completion or timeout).
    """
    if not prompt:
        raise RuntimeError("Empty prompt for Stable Horde")

    submit_url = "https://aihorde.net/api/v2/generate/async"
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "models": ["stable_diffusion"],
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

    job_id = job.get("id") or job.get("request_id") or job.get("job_id") or job.get("requestUUID")
    # Sometimes API returns images immediately (rare) in "images" field
    if not job_id:
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

    # Poll status endpoint used in earlier tests
    check_url = f"https://aihorde.net/api/v2/generate/status/{job_id}"
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

        # Look for the common indicators that it's done
        if status.get("done") or status.get("finished") or status.get("status") == "done" or status.get("is_possible") is True and status.get("done"):
            images = []
            # common shapes: 'generations' with dicts containing 'img' or 'img' as URL
            gens = status.get("generations") or status.get("images") or []
            if isinstance(gens, list):
                for g in gens:
                    if isinstance(g, str) and g.startswith("http"):
                        images.append(g)
                    elif isinstance(g, dict):
                        # { img: "url" } or { img: "base64..." }
                        img_field = g.get("img") or g.get("image") or g.get("url")
                        if isinstance(img_field, str):
                            if img_field.startswith("http"):
                                images.append(img_field)
                            elif img_field.startswith("data:"):
                                images.append(img_field)
                            else:
                                # sometimes it's raw base64
                                if len(img_field) > 60 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", img_field.replace("\n","").replace("\r","")):
                                    images.append("data:image/png;base64," + img_field.replace("\n","").replace("\r",""))
            # also some responses include "result" or "outputs"
            if isinstance(status, dict):
                for key in ("result", "outputs"):
                    if key in status and isinstance(status[key], list):
                        for it in status[key]:
                            if isinstance(it, dict) and it.get("img"):
                                images.append("data:image/png;base64," + it["img"])
                            elif isinstance(it, str) and it.startswith("http"):
                                images.append(it)
            images = [i for i in images if i]
            if images:
                return images
            # If finished but no images found, still raise
            if status.get("done") or status.get("finished"):
                raise RuntimeError(f"Stable Horde finished but returned no images: {status}")

        time.sleep(poll_interval)

    raise RuntimeError(f"Stable Horde generation timed out after {timeout}s. Last error: {last_err}")

# ---------- New: Async submit & status helpers for Stable Horde ----------
def stablehorde_submit_async(prompt, api_key=STABLE_HORDE_API_KEY):
    """
    Submit to aihorde async endpoint and return the job id (or raise).
    """
    submit_url = "https://aihorde.net/api/v2/generate/async"
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "models": ["stable_diffusion"],
        "params": {"steps": 20, "cfg_scale": 7.0, "width": 512, "height": 512}
    }
    r = requests.post(submit_url, json=payload, headers=headers, timeout=30)
    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise RuntimeError(f"Stable Horde submit error {r.status_code}: {body}")
    j = r.json()
    job_id = j.get("id") or j.get("request_id") or j.get("job_id") or j.get("requestUUID")
    if not job_id:
        # sometimes immediate images are returned
        if isinstance(j, dict) and j.get("images"):
            imgs = []
            for it in j["images"]:
                if isinstance(it, dict) and it.get("img"):
                    imgs.append("data:image/png;base64," + it["img"])
                elif isinstance(it, str) and it.startswith("http"):
                    imgs.append(it)
            if imgs:
                return {"job_id": None, "images": imgs}
        raise RuntimeError(f"No job id returned from Stable Horde submit: {j}")
    return {"job_id": job_id, "raw": j}

def stablehorde_get_status(job_id, api_key=STABLE_HORDE_API_KEY):
    """
    Query aihorde status endpoint and return the parsed JSON.
    """
    check_url = f"https://aihorde.net/api/v2/generate/status/{job_id}"
    headers = {"apikey": api_key}
    r = requests.get(check_url, headers=headers, timeout=30)
    if r.status_code != 200:
        # return the raw body for debugging
        try:
            return {"error": r.text, "status_code": r.status_code}
        except Exception:
            return {"error": "unknown error", "status_code": r.status_code}
    return r.json()
def call_stablehorde_img2img(prompt, source_image_b64, api_key=STABLE_HORDE_API_KEY, timeout=STABLE_HORDE_TIMEOUT, poll_interval=STABLE_HORDE_POLL_INTERVAL, params=None):
    """
    Submit an img2img (inpainting / img2img) async generation request to Stable Horde and poll until done.
    - source_image_b64: data:image/*;base64,... OR raw base64 string
    - params: dict with keys like steps, cfg_scale, denoising_strength, width, height, sampler, model, etc.
    Returns list of data URLs or http URLs.
    """
    if not prompt:
        raise RuntimeError("Empty prompt for Stable Horde img2img")
    if not source_image_b64:
        raise RuntimeError("source_image missing for img2img")

    submit_url = "https://stablehorde.net/api/v2/generate/async"
    headers = {"apikey": api_key, "Content-Type": "application/json"}

    # normalize base64 - ensure it's just the base64 part or data:... form
    src = source_image_b64.strip()
    # If it is a data URL keep it, else if it looks like raw base64 prefix it
    if not src.startswith("data:"):
        # Try to be generous: if it has newlines, remove them
        cand = src.replace("\n", "").replace("\r", "")
        if cand.startswith("iVBOR") or cand.startswith("/9j/"):  # png/jpg base64 starts
            src = "data:image/png;base64," + cand
        else:
            # otherwise try to use as-is
            src = cand

    payload = {
        "prompt": prompt,
        "params": {
            "steps": 30,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512
        },
        # include the source image base64 as 'source_image' (Stable Horde accepts similar key in many clients)
        "source_image": src,
        # indicate we want img2img mode if supported
        "type": "img2img"
    }

    # merge user params if provided
    if params and isinstance(params, dict):
        payload_params = payload.get("params", {})
        payload_params.update({k: v for k, v in params.items() if k is not None})
        payload["params"] = payload_params

    try:
        r = requests.post(submit_url, json=payload, headers=headers, timeout=30)
    except Exception as e:
        raise RuntimeError(f"Stable Horde img2img submit failed: {e}")

    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise RuntimeError(f"Stable Horde img2img submit error {r.status_code}: {body}")

    try:
        job = r.json()
    except Exception:
        raise RuntimeError("Stable Horde img2img submit: failed to parse JSON response")

    job_id = job.get("id") or job.get("request_id") or job.get("job_id") or job.get("requestUUID")
    if not job_id:
        # fallback: maybe immediate images returned in response
        out = []
        if isinstance(job, dict):
            if job.get("images"):
                for it in job["images"]:
                    if isinstance(it, dict) and it.get("img"):
                        out.append("data:image/png;base64," + it["img"])
                    elif isinstance(it, str) and it.startswith("http"):
                        out.append(it)
        if out:
            return out
        raise RuntimeError(f"Stable Horde img2img submit: no job id in response: {job}")

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
            raise RuntimeError(f"Stable Horde img2img finished but returned no images: {status}")

        time.sleep(poll_interval)

    raise RuntimeError(f"Stable Horde img2img timed out after {timeout}s. Last error: {last_err}")





# ---------- Hugging Face image helper (IMPROVED & LOGGING) ----------
def call_hf_image(prompt, model=HF_MODEL, max_retries=HF_MAX_RETRIES, timeout=REQUEST_TIMEOUT):
    """
    Calls Hugging Face Inference API for image generation.
    - If the model returns image bytes directly (Content-Type image/*) we save them and return a URL.
    - If the model returns JSON containing base64 or urls, we extract those and save/normalize.
    Returns: list of absolute image URLs (or remote urls) or raises RuntimeError.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    # NOTE: we do not force Accept to image/png only, because some models return JSON.
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
        # determine extension
        ext = "png"
        try:
            if "/" in content_type:
                ext = content_type.split("/")[1].split(";")[0]
        except Exception:
            ext = "png"
        return save_image_bytes_and_get_url(img_bytes, ext=ext)

    def _try_extract_and_save_from_json(j):
        # j may be dict, list, etc.
        candidates = []

        if isinstance(j, dict):
            # common fields
            for k in ("b64_json", "img", "image", "url", "data"):
                v = j.get(k)
                if isinstance(v, str):
                    candidates.append(v)
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, str):
                            candidates.append(it)
                        elif isinstance(it, dict):
                            # nested
                            if it.get("b64_json"):
                                candidates.append(it.get("b64_json"))
                            if it.get("url"):
                                candidates.append(it.get("url"))
            # possibly j["data"] is [{b64_json:...}, ...]
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

        # normalize & save
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
        # dedupe preserve order
        return list(dict.fromkeys(urls))

    for attempt in range(attempts):
        try:
            print(f"[HF] POST {hf_url} attempt={attempt+1}/{attempts} prompt_len={len(prompt)}")
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            print(f"[HF] status={status} content-type={ctype}")

            if status == 200:
                # direct image bytes (some models return image bytes)
                if ctype.startswith("image/"):
                    img_bytes = resp.content
                    return [_save_bytes_and_url(img_bytes, ctype)]

                # else try parse JSON
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    # maybe raw base64 in text
                    if _is_base64_string(txt):
                        img_bytes = base64.b64decode(txt)
                        return [_save_bytes_and_url(img_bytes, "image/png")]
                    # fallback: return textual body as a single item
                    return [txt]

                # attempt to extract images from JSON
                urls = _try_extract_and_save_from_json(j)
                if urls:
                    return urls

                # find http urls nested anywhere in the JSON
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
                    # dedupe
                    return list(dict.fromkeys(found))

                # nothing extractable - return JSON string (caller will inspect)
                return [json.dumps(j)]

            # transient statuses - retry with backoff
            elif status in (202, 429, 503):
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                print(f"[HF] transient status {status}: {str(body)[:300]}")
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

    # after attempts
    if last_exception:
        raise RuntimeError(f"Hugging Face call failed after {attempts} attempts: {last_exception}")
    raise RuntimeError("Hugging Face call failed for unknown reasons")

# wrapper that chooses provider (HF preferred unless forced off)
def generate_via_preferred_provider(prompt, language="en", source_image=None, img2img_params=None):
    """
    Returns (images_list, provider_name)
    - If source_image is provided (base64 or data:), attempts img2img using Stable Horde.
    - Otherwise behave like before (HF preferred, fallback to Stable Horde).
    """
    final_prompt = f"{prompt} (language: {language})"

    # If user provided a source image -> use Stable Horde img2img (skip HF)
    if source_image:
        try:
            imgs = call_stablehorde_img2img(final_prompt, source_image, params=img2img_params)
            provider = "stablehorde-img2img"
            # normalize and return
            normalized = []
            for it in (imgs or []):
                try:
                    u = ensure_absolute_url(it)
                    normalized.append(u or it)
                except Exception:
                    normalized.append(it)
            return normalized, provider
        except Exception as e:
            # attempt fallback to regular stablehorde generate (text->image)
            print("[image] Stable Horde img2img failed:", e)
            try:
                imgs = call_stablehorde(final_prompt)
                provider = "stablehorde-fallback"
            except Exception as e2:
                raise RuntimeError(f"Stable Horde img2img failed: {e}; fallback failed: {e2}")

    # No source image: previous logic (HF preferred unless forced)
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
    for it in (imgs or []):
        try:
            u = ensure_absolute_url(it)
            normalized.append(u or it)
        except Exception:
            normalized.append(it)
    return normalized, provider


# ---------- LOG STARTUP ----------
print(f"[startup] HF_MODEL = {HF_MODEL}")
print(f"[startup] HF_API_TOKEN present: {bool(HF_API_TOKEN)}; FORCE_STABLE_HORDE={FORCE_STABLE_HORDE}; STABLE_HORDE_KEY_present={STABLE_HORDE_API_KEY != '0000000000'}")

# Optional Web Search (Tavily). Set env: TAVILY_API_KEY
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_URL = os.environ.get("TAVILY_URL", "https://api.tavily.com/search")

# ---------- AGENT CONFIG (unchanged) ----------
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

# ---------- Helpers for files, parsing, chat endpoints (kept concise) ----------
def extract_text_from_file(path, mimetype):
    try:
        if mimetype == "text/plain":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif mimetype == "application/pdf":
            text = ""
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif mimetype in (
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Failed to parse {path}: {e}")
    return ""

def save_files(files):
    saved = []
    for f in files:
        if not getattr(f, "filename", None):
            continue
        fname = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(path)
        url = f"/uploads/{fname}"
        text = extract_text_from_file(path, f.mimetype)
        saved.append({
            "name": f.filename,
            "path": path,
            "url": url,
            "mimetype": f.mimetype,
            "content": (text or "")[:8000]
        })
    return saved

def auto_router(message, saved_files):
    msg = (message or "").lower()
    if saved_files:
        if any(sf["content"] for sf in saved_files):
            return "docs"
    if any(k in msg for k in ["search", "latest", "news", "today", "google", "cite"]):
        return "web"
    if any(k in msg for k in ["bug", "error", "stack", "function", "class", "code", "refactor"]):
        return "code"
    return "general"

def build_messages(message, agent_key, saved_files, web_results=None):
    cfg = AGENTS.get(agent_key, AGENTS[DEFAULT_AGENT_KEY])
    system = cfg["system"]
    user_content = message or ""

    if agent_key == "docs" and saved_files:
        parts = []
        for f in saved_files:
            if f["content"]:
                snippet = f["content"][:4000]
                parts.append(f"--- FILE: {f['name']} ---\n{snippet}")
        if parts:
            user_content = (
                f"Use ONLY these excerpts to answer. "
                f"If insufficient, say you couldn't find it.\n\n" +
                "\n\n".join(parts) +
                "\n\nUSER QUESTION:\n" + (message or "")
            )

    if agent_key == "web" and web_results:
        bullets = []
        for i, r in enumerate(web_results, start=1):
            bullets.append(f"[{i}] {r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}")
        ctx = "SOURCES:\n" + "\n\n".join(bullets)
        user_content = (
            f"{ctx}\n\nTASK: Answer the question. "
            f"Summarize ONLY in bullet points with inline citations like [1], [2].\n\nUSER QUESTION:\n{message or ''}"
        )

    user_content += "\n\nRemember: Use ONLY bullet points or numbered lists. Never write paragraphs."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]

def call_together(messages, model, temperature=0.7, max_tokens=600):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Together API error:", e)
        traceback.print_exc()
        return "⚠️ Sorry, I couldn’t fetch a response."

def web_search_tavily(query, max_results=4):
    if not TAVILY_API_KEY:
        return []
    try:
        res = requests.post(
            TAVILY_URL,
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results},
            timeout=20,
        )
        res.raise_for_status()
        data = res.json() or {}
        items = data.get("results") or []
        results = []
        for it in items:
            results.append({
                "title": it.get("title") or "",
                "url": it.get("url") or "",
                "snippet": it.get("content") or it.get("snippet") or ""
            })
        return results
    except Exception as e:
        print("Tavily error:", e)
        return []

# ---------- ROUTES ----------
@app.route("/health")
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()})

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"response": "⚠️ File too large. Try smaller files or compress first."}), 413

@app.route("/chat", methods=["POST"])
def chat():
    try:
        is_form = bool(request.form) or bool(request.files)
        if is_form:
            message = request.form.get("message", "")
            agent_in = request.form.get("agent", "auto")
            files = request.files.getlist("files")
        else:
            data = request.get_json(silent=True) or {}
            message = data.get("message", "")
            agent_in = data.get("agent", "auto")
            files = []

        if not (message or files):
            return jsonify({"response": "⚠️ You sent an empty request."}), 400

        saved_files = save_files(files) if files else []
        agent_key = agent_in if agent_in in AGENTS else auto_router(message, saved_files)

        web_results = []
        if agent_key == "web":
            web_results = web_search_tavily(message)

        cfg = AGENTS.get(agent_key, AGENTS[DEFAULT_AGENT_KEY])
        msgs = build_messages(message, agent_key, saved_files, web_results)
        reply = call_together(
            msgs, model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"]
        )

        images = []
        provider_used = None
        # 1) detect explicit user intent `/image prompt`
        m_user_img = re.match(r"^/image\s+(.+)$", (message or "").strip(), flags=re.IGNORECASE)
        if m_user_img:
            img_prompt = m_user_img.group(1).strip()
            try:
                images, provider_used = generate_via_preferred_provider(img_prompt)
            except Exception as e_img:
                reply += f"\n\n[Image generation failed: {str(e_img)}]"

        # 2) detect model instruction in the reply like: [generate_image: prompt]
        if not images:
            m = re.search(r"\[generate_image\s*:\s*(.+?)\]", reply, flags=re.IGNORECASE)
            if m:
                img_prompt = m.group(1).strip()
                try:
                    images, provider_used = generate_via_preferred_provider(img_prompt)
                except Exception as e_img:
                    reply += f"\n\n[Image generation failed: {str(e_img)}]"

        # Append clickable links for web results
        if agent_key == "web" and web_results:
            src_lines = []
            for i, r in enumerate(web_results, start=1):
                url = r.get("url", "")
                title = r.get("title", f"Source {i}")
                if url:
                    src_lines.append(f'<a href="{url}" target="_blank" rel="noopener">🔗 {title}</a>')
            if src_lines:
                reply += "<br><br>" + "<br>".join(src_lines)

        # File chips
        links_html = ""
        if saved_files:
            chips = " ".join(
                f'<a class="file-chip" href="{f["url"]}" target="_blank" rel="noopener">{f["name"]}</a>'
                for f in saved_files
            )
            links_html = f"<div class='file-chip-wrap'>{chips}</div>"

        print(f">>> Agent: {agent_key}  Msg: {message[:80]!r}  Files: {len(saved_files)}  Images: {len(images)}  Provider: {provider_used}")
        return jsonify({
            "response": f"{reply}{links_html}",
            "agent": agent_key,
            "files": saved_files,
            "images": images,
            "provider": provider_used
        })

    except Exception as ex:
        print("ERROR /chat:", ex)
        traceback.print_exc()
        return jsonify({"response": "⚠️ Server error while handling your request.", "error": str(ex)}), 500

@app.route("/generate-image", methods=["POST"])
def generate_image():
    """
    Called by frontend image button.
    Expects JSON: { prompt: "...", language: "en" }
    Returns: { images: [absolute_url1, ...], provider: "huggingface"|"stablehorde" }
    This endpoint is synchronous: it will block until provider returns images (HF immediate or Stable Horde polled).
    """
    try:
        try:
            body = request.get_json(silent=True)
        except Exception:
            return jsonify({"error": "invalid_json", "details": "Failed to parse JSON request body."}), 400

        if not body or not isinstance(body, dict):
            return jsonify({"error": "invalid_request", "details": "Request body must be a JSON object with a 'prompt' field."}), 400

        prompt = (body.get("prompt") or body.get("text") or "").strip()
        language = body.get("language", "en")
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400

               # read optional source_image or params from request
        source_image = body.get("source_image") or body.get("image") or body.get("img")
        img2img_params = body.get("params") or body.get("img2img_params") or None

        try:
            images, provider = generate_via_preferred_provider(prompt, language=language, source_image=source_image, img2img_params=img2img_params)
        except Exception as hf_err:

            tb = traceback.format_exc()
            print("ERROR generating image:", hf_err)
            print(tb)
            return jsonify({
                "error": "image_generation_failed",
                "details": str(hf_err),
                "traceback": tb
            }), 500

        # If anything looks like a relative / data / base64, ensure absolute fetchable URLs
        normalized = []
        for it in (images or []):
            url = None
            try:
                url = normalize_image_entry(it)
            except Exception:
                url = None
            if url:
                normalized.append(url)
            else:
                # fallback try ensure_absolute_url for strings
                if isinstance(it, str):
                    fallback = ensure_absolute_url(it)
                    normalized.append(fallback or it)
                else:
                    normalized.append(it)

        return jsonify({"images": normalized, "provider": provider})

    except Exception as e:
        tb = traceback.format_exc()
        print("generate_image top-level error:", e)
        print(tb)
        return jsonify({"error": "invalid_request", "details": str(e), "traceback": tb}), 500

# ---------------- New async endpoints for Stable Horde ----------------
@app.route("/generate-image-async", methods=["POST"])
def generate_image_async():
    """
    Submit an image generation job to Stable Horde and return job_id immediately.
    JSON body: { prompt: "..." }
    Returns: { job_id: "...", provider: "stablehorde" } or { images: [...] } if images returned immediately.
    """
    try:
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or data.get("text") or "").strip()
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400

        try:
            res = stablehorde_submit_async(prompt)
        except Exception as e:
            tb = traceback.format_exc()
            print("stablehorde submit error:", e)
            print(tb)
            return jsonify({"error": "stablehorde_submit_failed", "details": str(e)}), 500

        # If submission returned images immediately (rare), return them
        if res.get("images"):
            imgs = _ensure_saved_urls_from_list(res["images"])
            return jsonify({"images": imgs, "provider": "stablehorde"})

        job_id = res.get("job_id")
        if not job_id:
            return jsonify({"error": "no_job_id_returned", "raw": res}), 500

        return jsonify({"job_id": job_id, "provider": "stablehorde"})
    except Exception as e:
        tb = traceback.format_exc()
        print("generate_image_async error:", e)
        print(tb)
        return jsonify({"error": "internal_error", "details": str(e)}), 500

@app.route("/image-status/<job_id>", methods=["GET"])
def image_status(job_id):
    """
    Poll Stable Horde status for job_id and return a normalized response.
    Returns { status: "processing"|"done"|"failed", img_url: "...", generations: [...] }
    """
    try:
        info = stablehorde_get_status(job_id)
        # If error dictionary returned
        if info is None:
            return jsonify({"status": "processing"}), 202
        if isinstance(info, dict) and info.get("error"):
            return jsonify({"status": "error", "details": info.get("error"), "status_code": info.get("status_code")}), 500

        # look for finished
        done = False
        if info.get("done") or info.get("finished") or info.get("status") == "done" or info.get("is_possible") is True and info.get("done"):
            done = True

        generations = info.get("generations") or info.get("images") or []
        images = []
        if isinstance(generations, list):
            for g in generations:
                if isinstance(g, str) and g.startswith("http"):
                    images.append(g)
                elif isinstance(g, dict):
                    img_field = g.get("img") or g.get("image") or g.get("url")
                    if isinstance(img_field, str):
                        if img_field.startswith("http"):
                            images.append(img_field)
                        elif img_field.startswith("data:"):
                            images.append(img_field)
                        else:
                            # base64 heuristics
                            cand = img_field.replace("\n","").replace("\r","")
                            if len(cand) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", cand):
                                images.append("data:image/png;base64," + cand)
        # If done and images found - normalize and return first one + generations
        if done and images:
            saved = _ensure_saved_urls_from_list(images)
            return jsonify({"status": "done", "img_url": saved[0] if saved else images[0], "generations": images})
        # If done but no images present, still return raw info
        if done:
            return jsonify({"status": "done", "generations": generations, "raw": info})

        # still processing
        return jsonify({"status": "processing", "queue_position": info.get("queue_position"), "wait_time": info.get("wait_time")}), 202

    except Exception as e:
        tb = traceback.format_exc()
        print("image_status error:", e)
        print(tb)
        return jsonify({"status": "error", "details": str(e)}), 500

# ------------------------------------------------------------------------

if __name__ == "__main__":
    # Keep debug True locally, Render will run via gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
