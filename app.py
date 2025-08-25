# app.py
import os, uuid, datetime, traceback
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ---------- CONFIG ----------
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
# 50 MB max; adjust for your needs (also bump proxy limits if using Nginx/Render/Cloudflare)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# CORS (pip install flask-cors)
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
except Exception:
    pass  # If you can't add dependency right now, skip— but CORS will be required in browsers

# ---------- UTILS ----------
def save_files(files):
    saved = []
    for f in files:
        if not getattr(f, "filename", None):
            continue
        fname = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(path)
        url = f"/uploads/{fname}"
        saved.append({"name": f.filename, "path": path, "url": url, "mimetype": f.mimetype})
    return saved

# Example agents (replace with your real logic)
def agent_general(message, files, ctx):
    file_note = ""
    if files:
        file_note = "\n\nI can see these files:\n" + "\n".join([f"- {f['name']} ({f['mimetype']})" for f in files])
    return f"🤖 (General) You said: {message or '(no text)'}{file_note}"

def agent_docs(message, files, ctx):
    if not files:
        return "📄 (Docs) Please upload PDFs/DOCX to answer from them."
    return "📄 (Docs) I would search your uploaded documents and answer based on relevant passages."

def agent_web(message, files, ctx):
    return "🌐 (Web) I'd search the web for this query, synthesize results, and answer with citations."

def agent_code(message, files, ctx):
    return "💻 (Code) I can generate and explain code, write tests, and suggest improvements."

AGENTS = {
    "general": agent_general,
    "docs":    agent_docs,
    "web":     agent_web,
    "code":    agent_code,
}

def auto_router(message, files):
    text = (message or "").lower()
    if files:
        if any(f["mimetype"] in (
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ) for f in files):
            return "docs"
    if any(k in text for k in ["search the web", "latest", "news", "google this"]):
        return "web"
    if any(k in text for k in ["code", "bug", "error", "function", "class", "algorithm"]):
        return "code"
    return "general"

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
        # Support FormData (preferred) and JSON
        is_form = bool(request.form) or bool(request.files)
        if is_form:
            message = request.form.get("message", "")
            agent = request.form.get("agent", "auto")
            files = request.files.getlist("files")
        else:
            data = request.get_json(silent=True) or {}
            message = data.get("message", "")
            agent = data.get("agent", "auto")
            files = []

        if not (message or files):
            return jsonify({"response": "⚠️ You sent an empty request."}), 400

        saved_files = save_files(files) if files else []

        ctx = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "session_id": request.headers.get("X-Session-Id", "default"),
            "saved_files": saved_files,
        }

        agent_key = agent if agent in AGENTS else auto_router(message, saved_files)
        handler = AGENTS.get(agent_key, agent_general)
        reply = handler(message, saved_files, ctx)

        links_html = ""
        if saved_files:
            chips = " ".join(
                f'<a class="file-chip" href="{f["url"]}" target="_blank" rel="noopener">{f["name"]}</a>'
                for f in saved_files
            )
            links_html = f"<div class='file-chip-wrap'>{chips}</div>"

        return jsonify({"response": f"{reply}{links_html}", "agent": agent_key, "files": saved_files})

    except Exception as ex:
        # Log full traceback on server
        print("ERROR /chat:", ex)
        traceback.print_exc()
        # Send a helpful JSON error to client
        return jsonify({
            "response": "⚠️ Server error while handling your request.",
            "error": str(ex),
        }), 500

if __name__ == "__main__":
    # For production behind a proxy, use gunicorn/uvicorn; debug=False recommended
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
