# app.py - Final ready-to-deploy backend for BharatAI (updated with invite-code API)
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
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150 MB

# DB file (kept in project root; on Render this is ephemeral — consider Postgres for production)
DATABASE = os.path.join(os.getcwd(), "codes.db")

# Frontend origins - adjust if needed (added webtoolslive)
FRONTEND_ORIGINS = [
    "https://bhaaratai.in",
    "https://www.bhaaratai.in",
    "http://localhost:3000",
    "http://localhost:8000",
    "https://webtoolslive.com",
    "https://www.webtoolslive.com",
]

# CORS - allow those origins; we'll also add headers in after_request for extra compatibility
CORS(app, origins=FRONTEND_ORIGINS, supports_credentials=False)

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # fallback permissive header so static sites can test — remove/change for stricter production
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, apikey, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

# ---------- ENV / PROVIDER ----------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "ali-vilab/text-to-video-ms-1.7b")
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "120"))
HF_RETRIES = int(os.environ.get("HF_RETRIES", "3"))

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_URL = os.environ.get("TOGETHER_URL", "https://api.together.xyz/v1/chat/completions")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")  # publish/management bearer token

# ---------- DB helpers ----------
def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        # sqlite3 default isolation and row factory
        db = g._db = sqlite3.connect(DATABASE, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception=None):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()

def init_db():
    """Initialize codes table if missing and add usage columns if needed."""
    db = get_db()
    # Create table if not exists
    db.execute("""
    CREATE TABLE IF NOT EXISTS codes (
        id TEXT PRIMARY KEY,
        code TEXT UNIQUE NOT NULL,
        note TEXT,
        published_by TEXT,
        published_at TEXT,
        usage_count INTEGER DEFAULT 0,
        max_uses INTEGER DEFAULT 4,
        valid INTEGER DEFAULT 1
    );
    """)
    db.commit()
    # Ensure columns exist (safe to run multiple times)
    # SQLite ALTER ADD column will fail if exists; wrap in try/except
    try:
        db.execute("ALTER TABLE codes ADD COLUMN usage_count INTEGER DEFAULT 0")
        db.execute("ALTER TABLE codes ADD COLUMN max_uses INTEGER DEFAULT 4")
        db.execute("ALTER TABLE codes ADD COLUMN valid INTEGER DEFAULT 1")
        db.commit()
    except Exception:
        # columns already exist or other benign error
        pass

# initialize DB eagerly (works across Flask versions)
with app.app_context():
    try:
        init_db()
    except Exception as e:
        import sys
        print("DB init error:", e, file=sys.stderr)

# ---------- UTILITIES (file + HF helpers kept as before) ----------
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
    base = ""
    try:
        base = (request.host_url or "").rstrip("/")
    except Exception:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
    if not base:
        base = os.environ.get("BACKEND_URL", "").rstrip("/")
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
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    candidate = s.replace("\n", "").replace("\r", "")
    if len(candidate) > 60 and re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
        return save_base64_and_return_url(candidate)
    return s

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# HF helpers (unchanged from your code)
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
        if ctype.startswith("image/"):
            ext = ctype.split("/")[1].split(";")[0] or "png"
            return [save_bytes_and_get_url(resp.content, content_type=ctype, ext_hint=ext)]
        try:
            j = resp.json()
        except Exception:
            txt = resp.text or ""
            if isinstance(txt, str) and len(txt) > 100:
                u = save_base64_and_return_url(txt)
                if u:
                    return [u]
            raise RuntimeError("HF returned non-image and non-json response")
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
        raise RuntimeError(f"Hugging Face returned JSON but no image fields. snippet: {json.dumps(j)[:1000]}")
    elif status in (202, 429, 503):
        raise RuntimeError(f"Hugging Face transient status {status}: {resp.text}")
    else:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"Hugging Face image error {status}: {err}")

def call_hf_video(prompt, model=HF_VIDEO_MODEL, timeout=HF_TIMEOUT, retries=HF_RETRIES):
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

# ---------- Invite-code API ----------
@app.route("/api/codes", methods=["GET"])
def api_codes():
    try:
        db = get_db()
        cur = db.execute("SELECT code, note, usage_count, max_uses, valid FROM codes WHERE valid = 1")
        rows = cur.fetchall()
        codes = []
        for r in rows:
            codes.append({
                "code": r["code"],
                "note": r["note"],
                "usage_count": r["usage_count"] or 0,
                "max_uses": r["max_uses"] or 4,
                "valid": r["valid"] if r["valid"] is not None else 1
            })
        return jsonify({"ok": True, "codes": codes})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": "server_error", "details": str(e)}), 500

@app.route("/api/use", methods=["POST"])
def api_use():
    try:
        data = request.get_json(silent=True) or {}
        code_text = (data.get("code") or "").strip()
        if not code_text:
            return jsonify({"ok": False, "error": "empty_code"}), 400
        db = get_db()
        # select row for update pattern (sqlite single writer): SELECT -> check -> UPDATE
        cur = db.execute("SELECT id, usage_count, max_uses, valid FROM codes WHERE code = ? LIMIT 1", (code_text,))
        row = cur.fetchone()
        if not row:
            return jsonify({"ok": False, "error": "not_found"}), 404
        if row["valid"] == 0:
            return jsonify({"ok": False, "error": "expired"}), 400
        usage = (row["usage_count"] or 0) + 1
        max_ = row["max_uses"] or 4
        new_valid = 1 if usage < max_ else 0
        db.execute("UPDATE codes SET usage_count = ?, valid = ? WHERE id = ?", (usage, new_valid, row["id"]))
        db.commit()
        return jsonify({"ok": True, "code": code_text, "usage_count": usage, "max_uses": max_, "expired": new_valid == 0})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": "server_error", "details": str(e)}), 500

@app.route("/api/publish", methods=["POST"])
def api_publish():
    try:
        auth = request.headers.get("Authorization", "")
        if not ADMIN_TOKEN:
            return jsonify({"ok": False, "error": "admin_token_not_configured"}), 500
        if not auth.lower().startswith("bearer ") or auth.split(None, 1)[1].strip() != ADMIN_TOKEN:
            return jsonify({"ok": False, "error": "unauthorized"}), 403
        data = request.get_json(silent=True) or {}
        code_text = (data.get("code") or "").strip()
        note = data.get("note", "")
        max_uses = int(data.get("max_uses", 4) or 4)
        if not code_text:
            return jsonify({"ok": False, "error": "empty_code"}), 400
        db = get_db()
        new_id = str(uuid.uuid4())
        published_by = data.get("published_by", "api")
        published_at = datetime.datetime.utcnow().isoformat() + "Z"
        try:
            db.execute("INSERT INTO codes (id, code, note, published_by, published_at, usage_count, max_uses, valid) VALUES (?, ?, ?, ?, ?, 0, ?, 1)",
                       (new_id, code_text, note, published_by, published_at, max_uses))
            db.commit()
        except sqlite3.IntegrityError:
            return jsonify({"ok": False, "error": "duplicate"}), 409
        return jsonify({"ok": True, "code": code_text, "id": new_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": "server_error", "details": str(e)}), 500

# ---------- ROUTES (existing features kept) ----------
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
                try:
                    reply = jr["choices"][0]["message"]["content"]
                except Exception:
                    reply = jr.get("output") or jr.get("result") or str(jr)
                return jsonify({"response": reply, "timestamp": datetime.datetime.utcnow().isoformat()})
            except Exception as e:
                print("Together API error:", e)
                traceback.print_exc()
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
