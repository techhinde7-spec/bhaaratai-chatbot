import os
import uuid
from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Folder for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Your Together AI API Key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "your_api_key_here")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "docx", "doc", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve uploaded files
@app.route("/files/<filename>")
def serve_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "").strip()
    files = request.files.getlist("files")

    responses = []

    # 🧠 AI response from Together AI
    if message:
        try:
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "meta-llama/Llama-3-8b-chat-hf",  # you can change model here
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 300,
            }
            r = requests.post("https://api.together.xyz/v1/chat/completions",
                              headers=headers, json=data)
            reply = r.json()["choices"][0]["message"]["content"]
            responses.append(reply)
        except Exception as e:
            responses.append(f"⚠️ AI error: {str(e)}")

    # 📂 Handle file uploads
    if files:
        file_links = []
        for f in files:
            if f and allowed_file(f.filename):
                filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                f.save(filepath)

                file_url = url_for("serve_file", filename=filename, _external=True)
                file_links.append(f'<a href="{file_url}" target="_blank">{f.filename}</a>')
            else:
                file_links.append(f"⚠️ {f.filename} (not allowed)")
        if file_links:
            responses.append("📎 Uploaded files:<br>" + "<br>".join(file_links))

    # Default fallback
    if not responses:
        responses.append("⚠️ You sent an empty request.")

    return jsonify({"response": "<br><br>".join(responses)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
