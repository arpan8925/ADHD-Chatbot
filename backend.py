from flask import Flask, request, jsonify, render_template
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load the Sentence Transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Renamed variable

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Renamed variable

# SQLite Database Setup
def get_db_connection():
    conn = sqlite3.connect("adhd_assistant.db", check_same_thread=False)
    return conn, conn.cursor()

def detect_routine_issues(user_id, user_message):
    cursor.execute("SELECT * FROM user_routine WHERE user_id = ?", (user_id,))
    record = cursor.fetchone()

    if not record:
        return "I don’t have your routine data yet. Want to tell me how your day has been?"

    sleep, meals, exercise, hygiene, hobbies, outdoors, relaxation = record[1:]
    issues = []

    emergency_keywords = ["emergency", "accident", "hospital", "stressful", "family issue"]
    if any(keyword in user_message.lower() for keyword in emergency_keywords):
        return "It sounds like something serious happened. Do you want to talk about it? ❤️"

    # Routine checks (unchanged)
    if not sleep or ("AM" in sleep and int(sleep.split()[0]) > 1):
        issues.append("You might be feeling tired because of late sleep.")
    if meals == "skipped":
        issues.append("Skipping meals can drain your energy.")
    if not exercise:
        issues.append("No movement today! A short stretch could help.")
    if not hygiene:
        issues.append("Self-care is important! A quick shower can refresh your mind.")
    if not hobbies:
        issues.append("You haven't done anything fun today. Want to take a break?")
    if not outdoors:
        issues.append("Getting some sunlight can boost your mood!")
    if not relaxation:
        issues.append("You haven't taken a mental break today. Maybe some deep breathing?")

    return "Hey, I noticed some things that might be affecting your mood:\n\n" + "\n".join(issues) if issues else "Your routine looks okay! 😊"

conn, cursor = get_db_connection()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_details (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        last_issue TEXT,
        issue_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_routine (
        user_id TEXT,
        sleep TEXT,
        meals TEXT,
        exercise TEXT,
        hygiene TEXT,
        hobbies TEXT,
        outdoors TEXT,
        relaxation TEXT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES user_details(user_id)
    )
''')

conn.commit()

# FAISS Memory Storage
index = faiss.IndexFlatL2(384)
stored_moods = []

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "123")
    user_message = data.get("message", "")

    # Store embedding using correct model
    user_embedding = embedding_model.encode([user_message])
    index.add(np.array(user_embedding))
    stored_moods.append(user_message)

    # Check for previous issues
    cursor.execute("SELECT last_issue FROM user_context WHERE user_id = ?", (user_id,))
    last_issue = cursor.fetchone()

    if last_issue:
        return jsonify({"response": f"Last time, you mentioned {last_issue[0]}. How are you feeling now? ❤️"})

    # Detect routine issues
    routine_issues = detect_routine_issues(user_id, user_message)

    # Store serious issues
    if "emergency" in routine_issues.lower() or "serious" in routine_issues.lower():
        cursor.execute('''
            INSERT INTO user_context (user_id, last_issue) 
            VALUES (?, ?) 
            ON CONFLICT(user_id) DO UPDATE SET last_issue=?
        ''', (user_id, user_message, user_message))
        conn.commit()

    # Generate response
    prompt = f"User feels: {user_message}. Routine issues: {routine_issues}. How can I help in a supportive ADHD-friendly way?"
    ai_response = "I'm having trouble generating a response right now. Try again later."  # Default response

    try:
        response = gemini_model.generate_content(prompt)  # Use renamed model
        ai_response = response.text
    except Exception as e:
        print(f"Gemini API error: {str(e)}")

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)