from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "your_actual_key_here")
TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"

@app.route("/", methods=["GET"])
def home():
    return "✅ BharatAI backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": user_input}],
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"response": reply})
    else:
        return jsonify({"error": "Together API call failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)




