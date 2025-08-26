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

# CORS (pip install flask-cors)
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
except Exception:
    pass

# Together API
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "your_key_here")
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"


# ---------- HELPERS ----------
def extract_text_from_file(path, mimetype):
    """Extract text content from supported file types."""
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


def call_together_ai(message, agent="general"):
    """
    Send a message to Together AI and return its response.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # choose model you want
        "messages": [
            {"role": "system", "content": f"You are BharatAI, an assistant that helps with {agent} tasks."},
            {"role": "user", "content": message}
        ],
        "max_tokens": 600,
        "temperature": 0.7
    }

    try:
        r = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Together API error:", e)
        traceback.print_exc()
        return "⚠️ Sorry, I couldn’t fetch a response from the AI."


def save_files(files):
    """Save uploaded files and return metadata with extracted text."""
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
            "content": text[:5000]  # limit to avoid huge prompts
        })
    return saved


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
            agent = request.form.get("agent", "general")
            files = request.files.getlist("files")
        else:
            data = request.get_json(silent=True) or {}
            message = data.get("message", "")
            agent = data.get("agent", "general")
            files = []

        if not (message or files):
            return jsonify({"response": "⚠️ You sent an empty request."}), 400

        saved_files = save_files(files) if files else []

        # Build context from files
        file_context = ""
        if saved_files:
            for f in saved_files:
                if f["content"]:
                    file_context += f"\n\n--- File: {f['name']} ---\n{f['content']}"
            message += "\n\nPlease consider the above file contents when answering."

        # Call Together AI with combined input
        reply = call_together_ai(message, agent)

        # Build file chips
        links_html = ""
        if saved_files:
            chips = " ".join(
                f'<a class="file-chip" href="{f["url"]}" target="_blank" rel="noopener">{f["name"]}</a>'
                for f in saved_files
            )
            links_html = f"<div class='file-chip-wrap'>{chips}</div>"

        return jsonify({
            "response": f"{reply}{links_html}",
            "agent": agent,
            "files": saved_files
        })

    except Exception as ex:
        print("ERROR /chat:", ex)
        traceback.print_exc()
        return jsonify({
            "response": "⚠️ Server error while handling your request.",
            "error": str(ex),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
