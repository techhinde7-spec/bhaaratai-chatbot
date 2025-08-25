# app.py (additions/edits)

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid, datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- Utilities ----------
def save_files(files):
    saved = []
    for f in files:
        if not f.filename:
            continue
        fname = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(path)
        # Make a public-ish URL if you serve /uploads statically; else return path
        url = f"/uploads/{fname}"  # ensure you have a static route for this
        saved.append({"name": f.filename, "path": path, "url": url, "mimetype": f.mimetype})
    return saved

# ---------- Agent implementations (examples) ----------
def agent_general(message, files, ctx):
    """Default LLM reply (replace with your real model call)."""
    # TODO: call your model here with message + files summary from ctx
    file_note = ""
    if files:
        file_note = "\n\nI can see these files:\n" + "\n".join([f"- {f['name']} ({f['mimetype']})" for f in files])
    return f"🤖 (General) You said: {message or '(no text)'}{file_note}"

def agent_docs(message, files, ctx):
    """Answer from uploaded docs only (RAG-lite)."""
    # TODO: extract text from PDFs/DOCX, index per session, retrieve top chunks, then answer.
    if not files and not ctx.get("session_docs"):
        return "📄 (Docs) Please upload PDFs/DOCX to answer from them."
    return "📄 (Docs) I would search your uploaded documents and answer based on relevant passages."

def agent_web(message, files, ctx):
    """Use web search before answering (requires your search key)."""
    # TODO: integrate your search API, fetch top results, cite links.
    return "🌐 (Web) I'd search the web for this query, synthesize results, and answer with citations."

def agent_code(message, files, ctx):
    """Coding assistant with snippets/tests."""
    return "💻 (Code) I can generate and explain code, write tests, and suggest improvements."

# Registry: short keys → callables
AGENTS = {
    "general": agent_general,
    "docs":    agent_docs,
    "web":     agent_web,
    "code":    agent_code,
}

def auto_router(message, files):
    """Very simple auto routing; expand with better heuristics later."""
    text = (message or "").lower()
    if files:
        # If mostly docs, go to docs agent
        if any(f['mimetype'] in ("application/pdf",
                                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                 "application/msword") for f in files):
            return "docs"
    if any(k in text for k in ["search the web", "latest", "news", "google this"]):
        return "web"
    if any(k in text for k in ["code", "bug", "error", "function", "class", "algorithm"]):
        return "code"
    return "general"

# ---------- Chat endpoint ----------
@app.route("/chat", methods=["POST"])
def chat():
    # Works for both fetch(FormData) and fetch(JSON body) gracefully
    message = request.form.get("message") if request.form else (request.json.get("message") if request.is_json else "")
    agent   = request.form.get("agent") if request.form else (request.json.get("agent") if request.is_json else "general")
    files   = request.files.getlist("files") if request.files else []

    # Guard: empty everything
    if not (message or files):
        return jsonify({"response": "⚠️ You sent an empty request."}), 400

    # Save uploads (if any)
    saved_files = save_files(files) if files else []

    # Build a lightweight context you can extend (session_id, user_id, memory, vector_index, etc.)
    ctx = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "session_id": request.headers.get("X-Session-Id", "default"),
        "saved_files": saved_files,
        # "session_docs": ... load per-session doc index here
    }

    # Route agent
    if not agent or agent == "auto":
        agent_key = auto_router(message, saved_files)
    else:
        agent_key = agent if agent in AGENTS else "general"

    handler = AGENTS[agent_key]
    reply = handler(message, saved_files, ctx)

    # If you want to include uploaded file links back to client, add a little HTML
    links_html = ""
    if saved_files:
        items = []
        for f in saved_files:
            # Render as buttons/cards on the frontend; sending <a> tags here is fine—your UI styles them
            items.append(f'<a class="file-chip" href="{f["url"]}" target="_blank" rel="noopener">{f["name"]}</a>')
        links_html = "<div class='file-chip-wrap'>" + " ".join(items) + "</div>"

    return jsonify({
        "response": f"{reply}{links_html}",
        "agent": agent_key,
        "files": saved_files
    })

# Serve /uploads if you're not already doing it through a reverse proxy
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return app.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
