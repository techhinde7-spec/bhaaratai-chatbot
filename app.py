# app.py
import os, uuid, datetime, traceback, requests, time, re, base64
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
HF_MODEL = os.environ.get("HF_MODEL", "stabilityai/stable-diffusion-2")
HF_MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", "3"))

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
        "model":  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
        "model":  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
        "model":  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
        "model":  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "temperature": 0.25,
        "max_tokens": 700,
    }
}
DEFAULT_AGENT_KEY = "general"

# ---------- HELPERS ----------
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

# ---------- Hugging Face image helper ----------
def call_hf_image(prompt, model=HF_MODEL, max_retries=HF_MAX_RETRIES, timeout=120):
    """
    Calls Hugging Face inference API.
    Returns: list of image data-URLs or remote URLs.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    hf_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "image/png, application/json",
    }
    payload = {"inputs": prompt}
    for attempt in range(max_retries):
        resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            # direct image bytes
            if content_type.startswith("image/"):
                img_bytes = resp.content
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                return [f"data:{content_type};base64,{b64}"]
            # JSON response (could include b64_json or urls)
            try:
                j = resp.json()
                images = []
                if isinstance(j, list):
                    for item in j:
                        if isinstance(item, dict):
                            if item.get("b64_json"):
                                images.append(f"data:image/png;base64,{item['b64_json']}")
                            elif item.get("url"):
                                images.append(item["url"])
                            elif item.get("generated_image"):
                                images.append(item["generated_image"])
                        elif isinstance(item, str) and item.startswith("http"):
                            images.append(item)
                elif isinstance(j, dict):
                    if j.get("images") and isinstance(j["images"], list):
                        for it in j["images"]:
                            if isinstance(it, str):
                                images.append(it)
                            elif isinstance(it, dict) and it.get("b64_json"):
                                images.append(f"data:image/png;base64,{it['b64_json']}")
                    if j.get("data") and isinstance(j["data"], list):
                        for d in j["data"]:
                            if isinstance(d, dict) and d.get("b64_json"):
                                images.append(f"data:image/png;base64,{d['b64_json']}")
                if images:
                    return images
            except Exception:
                pass
            # fallback to raw text body
            return [resp.text]
        elif resp.status_code in (503, 202):
            time.sleep(2 + attempt * 2)
            continue
        else:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"Hugging Face error {resp.status_code}: {err}")
    raise RuntimeError("Hugging Face model not ready after retries")

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
        # 1) detect explicit user intent `/image prompt` -> generate image directly from user prompt
        m_user_img = re.match(r"^/image\s+(.+)$", (message or "").strip(), flags=re.IGNORECASE)
        if m_user_img:
            img_prompt = m_user_img.group(1).strip()
            try:
                images = call_hf_image(img_prompt)
            except Exception as e_img:
                reply += f"\n\n[Image generation failed: {str(e_img)}]"

        # 2) detect model instruction in the reply like: [generate_image: prompt]
        if not images:
            m = re.search(r"\[generate_image\s*:\s*(.+?)\]", reply, flags=re.IGNORECASE)
            if m:
                img_prompt = m.group(1).strip()
                try:
                    images = call_hf_image(img_prompt)
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

        print(f">>> Agent: {agent_key}  Msg: {message[:80]!r}  Files: {len(saved_files)}  Images: {len(images)}")
        return jsonify({
            "response": f"{reply}{links_html}",
            "agent": agent_key,
            "files": saved_files,
            "images": images
        })

    except Exception as ex:
        print("ERROR /chat:", ex); traceback.print_exc()
        return jsonify({"response": "⚠️ Server error while handling your request.", "error": str(ex)}), 500

@app.route("/generate-image", methods=["POST"])
def generate_image():
    """
    Called by frontend image button.
    Expects JSON: { prompt: "...", language: "en" }
    Returns: { images: [dataUrl1, ...], model: HF_MODEL }
    """
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or body.get("text") or "").strip()
        language = body.get("language", "en")
        if not prompt:
            return jsonify({"error": "missing_prompt"}), 400

        final_prompt = f"{prompt} (language: {language})"
        images = call_hf_image(final_prompt)
        return jsonify({"images": images, "model": HF_MODEL})
    except Exception as e:
        return jsonify({"error": "image_generation_failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
