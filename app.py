# app.py
import os, uuid, datetime, traceback, requests
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
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

# Optional Web Search (Tavily). Set env: TAVILY_API_KEY
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"

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

        print(f">>> Agent: {agent_key}  Msg: {message[:80]!r}  Files: {len(saved_files)}")
        return jsonify({
            "response": f"{reply}{links_html}",
            "agent": agent_key,
            "files": saved_files
        })

    except Exception as ex:
        print("ERROR /chat:", ex); traceback.print_exc()
        return jsonify({"response": "⚠️ Server error while handling your request.", "error": str(ex)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
