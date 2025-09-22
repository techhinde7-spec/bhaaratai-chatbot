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

# ---------- New helper: normalize / persist returned image entries ----------
def normalize_image_entry(entry):
    """
    Accepts: string or dict or list. Returns an absolute URL the frontend can fetch,
    or None if it can't be normalized.
    Uses ensure_absolute_url and save_base64_and_return_url helpers defined above.
    """
    try:
        # If it's already a URL or data:... or base64 string, ensure_absolute_url will handle.
        if isinstance(entry, str):
            s = entry.strip()

            # If string looks like a JSON-encoded object, parse and try to extract b64/url
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = json.loads(s)
                    # prefer b64_json, img, image, url fields
                    for key in ("b64_json", "img", "image", "data", "url"):
                        if key in parsed and isinstance(parsed[key], str) and parsed[key].strip():
                            candidate = parsed[key].strip()
                            maybe = normalize_image_entry(candidate)
                            if maybe:
                                return maybe
                    # fallback: scan values for a long base64-like string
                    for v in parsed.values():
                        if isinstance(v, str) and len(v) > 60 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", v.replace("\n","").replace("\r","")):
                            return save_base64_and_return_url(v)
                except Exception:
                    # not valid JSON, fall through to base64 checks
                    pass

            # If it's data:...base64, or raw base64, pass to ensure_absolute_url (which saves)
            try:
                maybe = ensure_absolute_url(s)
                if maybe:
                    return maybe
            except Exception as e:
                print("[normalize_image_entry] ensure_absolute_url failed:", e)

            return None

        # If it's a dict with expected fields
        if isinstance(entry, dict):
            for key in ("b64_json", "img", "image", "data", "url"):
                if key in entry and isinstance(entry[key], str) and entry[key].strip():
                    maybe = normalize_image_entry(entry[key].strip())
                    if maybe:
                        return maybe
            # if nested: try converting dict -> json -> string handling
            try:
                text = json.dumps(entry)
                return normalize_image_entry(text)
            except Exception:
                pass

        # arrays: try first element
        if isinstance(entry, (list, tuple)) and entry:
            return normalize_image_entry(entry[0])

    except Exception as exc:
        print("[normalize_image_entry] unexpected:", exc)
    return None

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


# ---------- Hugging Face image helper (IMPROVED & LOGGING) ----------
def call_hf_image(prompt, model=HF_MODEL, max_retries=HF_MAX_RETRIES, timeout=REQUEST_TIMEOUT):
    """
    Calls Hugging Face inference API robustly.
    Returns: list of image URLs (absolute URLs under /uploads) or remote URLs.
    Raises RuntimeError on unrecoverable error.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "BharatAI/Render"
    }

    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True}
    }

    backoff = [1, 2, 4, 8]
    attempts = max_retries if max_retries > 0 else len(backoff)
    last_exception = None

    def _is_base64_string(s):
        if not s or not isinstance(s, str):
            return False
        s2 = s.strip()
        return len(s2) > 50 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r" for c in s2)

    def _maybe_save_and_return_urls(strings):
        out = []
        for s in strings:
            if not s or not isinstance(s, str):
                continue
            s = s.strip()
            # already a URL
            if s.startswith("http://") or s.startswith("https://"):
                out.append(s)
                continue
            # data URL
            if s.startswith("data:"):
                try:
                    _, b64 = s.split(",", 1)
                    img_bytes = base64.b64decode(b64)
                    ext = "png"
                    m = s.split(";")[0]
                    if "/" in m:
                        ext = m.split("/")[1].split("+")[0]
                    out.append(save_image_bytes_and_get_url(img_bytes, ext=ext))
                except Exception:
                    continue
                continue
            # raw base64
            if _is_base64_string(s):
                try:
                    img_bytes = base64.b64decode(s)
                    out.append(save_image_bytes_and_get_url(img_bytes, ext="png"))
                except Exception:
                    continue
                continue
        return out

    for attempt in range(attempts):
        try:
            print(f"[HF] POST {hf_url} attempt={attempt+1}/{attempts} payload_len={len(prompt)}")
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            print(f"[HF] status={resp.status_code} content-type={resp.headers.get('Content-Type')}")
            if resp.status_code == 200:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if content_type.startswith("image/"):
                    img_bytes = resp.content
                    ext = content_type.split("/")[1].split(";")[0] or "png"
                    return [save_image_bytes_and_get_url(img_bytes, ext=ext)]

                # otherwise parse JSON or fallback text
                try:
                    j = resp.json()
                except Exception:
                    txt = resp.text or ""
                    maybe = _maybe_save_and_return_urls([txt])
                    if maybe:
                        return maybe
                    return [txt]

                # collect candidate strings from common HF shapes
                candidates = []
                if isinstance(j, list):
                    for it in j:
                        if isinstance(it, str):
                            candidates.append(it)
                        elif isinstance(it, dict):
                            if it.get("b64_json"):
                                candidates.append(it["b64_json"])
                            if it.get("url"):
                                candidates.append(it["url"])
                if isinstance(j, dict):
                    if j.get("b64_json"):
                        candidates.append(j.get("b64_json"))
                    if j.get("url"):
                        candidates.append(j.get("url"))
                    if j.get("data") and isinstance(j["data"], list):
                        for d in j["data"]:
                            if isinstance(d, dict):
                                if d.get("b64_json"):
                                    candidates.append(d.get("b64_json"))
                                if d.get("url"):
                                    candidates.append(d.get("url"))
                            elif isinstance(d, str):
                                candidates.append(d)
                    if j.get("images") and isinstance(j["images"], list):
                        for it in j["images"]:
                            if isinstance(it, str):
                                candidates.append(it)
                            elif isinstance(it, dict) and it.get("b64_json"):
                                candidates.append(it.get("b64_json"))
                    if j.get("result") and isinstance(j["result"], list):
                        for it in j["result"]:
                            if isinstance(it, dict) and it.get("img"):
                                candidates.append(it.get("img"))
                            elif isinstance(it, str):
                                candidates.append(it)

                urls = _maybe_save_and_return_urls(candidates)
                if urls:
                    return urls

                # find nested http urls
                def find_urls(obj):
                    found = []
                    if isinstance(obj, str) and (obj.startswith("http://") or obj.startswith("https://")):
                        found.append(obj)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            found += find_urls(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            found += find_urls(it)
                    return found

                found_urls = find_urls(j)
                if found_urls:
                    return list(dict.fromkeys(found_urls))

                return [json.dumps(j)]

            elif resp.status_code in (202, 503, 429):
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                print(f"[HF] transient status {resp.status_code}, body: {str(body)[:300]}")
                wait = backoff[min(attempt, len(backoff) - 1)]
                time.sleep(wait)
                last_exception = RuntimeError(f"HF transient status {resp.status_code}: {body}")
                continue

            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"Hugging Face error {resp.status_code}: {err}")

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

# ---------- LOG STARTUP ----------
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

# ---------- HELPER: file parsing and saving ----------
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

    # 🔑 Reinforce bullet rules in every prompt
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
        # 1) detect explicit user intent `/image prompt` -> generate image directly from user prompt
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

# ---------- Single generate-image route (only one) ----------
@app.route("/generate-image", methods=["POST"])
def generate_image():
    """
    Called by frontend image button.
    Expects JSON: { prompt: "...", language: "en" }
    Returns: { images: [absolute_url1, ...], provider: "huggingface"|"stablehorde" }
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

        # call provider(s)
        try:
            images, provider = generate_via_preferred_provider(prompt, language=language)
        except Exception as hf_err:
            tb = traceback.format_exc()
            print("ERROR generating image:", hf_err)
            print(tb)
            return jsonify({
                "error": "image_generation_failed",
                "details": str(hf_err),
                "traceback": tb
            }), 500

        # Normalize/convert each returned image entry into an absolute URL where possible
        normalized = []
        for it in (images or []):
            try:
                url = normalize_image_entry(it)
                if url:
                    normalized.append(url)
                else:
                    # last-resort: attempt ensure_absolute_url (may return same or None)
                    fallback = ensure_absolute_url(it) if isinstance(it, str) else None
                    if fallback:
                        normalized.append(fallback)
                    else:
                        # include raw item so frontend can inspect error shapes
                        normalized.append(it)
            except Exception as e:
                print("[generate-image] normalization failed for entry:", e)
                normalized.append(it)

        return jsonify({"images": normalized, "provider": provider})

    except Exception as e:
        tb = traceback.format_exc()
        print("generate_image top-level error:", e)
        print(tb)
        return jsonify({"error": "invalid_request", "details": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    # Keep debug True locally, Render will run via gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)

