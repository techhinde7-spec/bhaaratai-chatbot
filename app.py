from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Add this line
import requests
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS right after initializing app

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "your_together_api_key_here")
TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"

@app.route("/")
def home():
    return "Bhaaratai Chatbot backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": TOGETHER_MODEL,
            "messages": [{"role": "user", "content": user_input}],
            "temperature": 0.7,
            "max_tokens": 512,
        },
    )

    if response.status_code == 200:
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"response": reply})
    else:
        return jsonify({"error": "Failed to get response from Together AI"}), 500

if __name__ == "__main__":
    app.run(debug=True)

