from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os

app = Flask(__name__)

# Set your Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyDNh6ag0_LtAnqqARTSyGv2HVoI1A1he_g"
genai.configure(api_key=GOOGLE_API_KEY)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
