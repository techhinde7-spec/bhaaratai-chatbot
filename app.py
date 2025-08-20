from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # allow frontend access

# Folder to temporarily save uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "doc", "docx", "xls", "xlsx", "zip"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/chat", methods=["POST"])
def chat():
    # Get text message
    message = request.form.get("message", "").strip()

    # Get uploaded files
    uploaded_files = request.files.getlist("files")

    responses = []

    # Handle message
    if message:
        responses.append(f"📝 You said: {message}")

    # Handle files
    saved_files = []
    for f in uploaded_files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(filepath)
            saved_files.append(filename)

            # Add type-based response
            ext = filename.rsplit(".", 1)[1].lower()
            if ext in ["png", "jpg", "jpeg", "gif"]:
                responses.append(f"📷 Received image: {filename}")
            elif ext == "pdf":
                responses.append(f"📄 Received PDF: {filename}")
            elif ext in ["doc", "docx"]:
                responses.append(f"📝 Received Word document: {filename}")
            elif ext in ["xls", "xlsx"]:
                responses.append(f"📊 Received Excel file: {filename}")
            elif ext == "zip":
                responses.append(f"📦 Received ZIP archive: {filename}")
            else:
                responses.append(f"📎 Received file: {filename}")
        else:
            responses.append(f"⚠️ File type not allowed: {f.filename}")

    # If nothing sent
    if not message and not uploaded_files:
        responses.append("⚠️ You sent an empty request.")

    return jsonify({"response": "\n".join(responses)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
