from flask import Flask, request, jsonify, render_template
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import os
import random

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

greeting_phrases = [
    "hi", "hello", "hey", "howdy", "greetings", 
    "good morning", "good afternoon", "good evening",
    "hi there", "hey there", "hello friend", "salutations",
    "how's it going", "what's up", "yo", "hola", "bonjour",
    "namaste", "g'day", "ahoy", "aloha", "how do you do",
    "nice to see you", "look who it is", "how are you",
    "pleasure to meet you", "top of the morning", "hiya"
]

greeting_embeddings = embedding_model.encode(greeting_phrases)

# Convert to numpy array and normalize
greeting_embeddings_np = greeting_embeddings.astype('float32')


# Normalize and create FAISS index
faiss.normalize_L2(greeting_embeddings_np)
dimension = greeting_embeddings_np.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(greeting_embeddings_np)


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
    user_message = data.get("message", "").strip()
    user_message_lower = user_message.lower()
    user_embedding = embedding_model.encode([user_message])[0]

    # Convert to numpy array directly
    user_embedding_np = user_embedding.astype('float32')  # Remove any .cpu() calls
    user_embedding_np = np.expand_dims(user_embedding_np, axis=0)
    faiss.normalize_L2(user_embedding_np)

    print(f"Greeting embeddings type: {type(greeting_embeddings)}")
    print(f"User embedding type: {type(user_embedding)}")

    # FAISS similarity search
    D, I = faiss_index.search(user_embedding_np, 1)
    max_similarity = D[0][0].item()
    stored_moods.append(user_message)

    if max_similarity > 0.65:  # Same threshold as before
        is_greeting = True
    else:
        is_greeting = False

    if is_greeting:
        try:
            prompt = f"""Generate a warm, friendly response to this greeting: "{user_message}".
            Use informal language, vary sentence structure, and include 1-2 relevant emojis. 
            Examples of good responses: 
            - "Hey there! 👋 What brings you here today?"
            - "Hello! 🌟 How can I make your day better?"
            - "Hi! 🎉 Ready for some productive fun?" 
            Keep response under 2 sentences. Also ask How he is feeling today or How is he ?"""
            
            response = gemini_model.generate_content(prompt)
            if response.text:
                return jsonify({"response": response.text})
            
        except Exception as e:
            print(f"Greeting generation failed: {str(e)}")
            # Fallback to dynamic template-based response
            fallbacks = [
                "Hi there! 👋 How can I help you today?",
                "Hello! 🌟 What's on your mind?",
                "Hey! 🚀 Ready to tackle something awesome?"
            ]
            return jsonify({"response": random.choice(fallbacks)})
    
    # Handle depression-related keywords
    depression_keywords = { 
        "depressed", "sad", "hopeless", "worthless", "suicidal",
        "empty", "alone", "miserable", "end it all"
    }
    if any(keyword in user_message_lower for keyword in depression_keywords):
        # Store as serious issue
        cursor.execute('''
            INSERT INTO user_details (user_id, last_issue) 
            VALUES (?, ?) 
            ON CONFLICT(user_id) DO UPDATE SET last_issue=?
        ''', (user_id, user_message, user_message))
        conn.commit()
        return jsonify({
            "response": "I'm really sorry you're feeling this way. 💛 You're not alone. "
                        "Would you like me to help you connect with professional support?"
                        "\n\nYou can also text/call 988 for the Suicide & Crisis Lifeline."
        })

    # Check for previous issues
    cursor.execute("SELECT last_issue FROM user_details WHERE user_id = ?", (user_id,))
    last_issue = cursor.fetchone()

    if last_issue:
        return jsonify({
            "response": f"Last time we talked & you mentioned {last_issue[0]}. "
                        f"How are you feeling about that now? 💭"
        })

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
    prompt = f"""ADHD-friendly response to: "{user_message}". 
    Detected issues: {routine_issues}. 
    Respond with empathy, bullet points if needed, and offer practical help."""
    
    ai_response = "I'm here to help! Let's break this down together. 💪"  # Default response

    try:
        response = gemini_model.generate_content(prompt)
        if response.text:
            ai_response = response.text
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        ai_response = "Hmm, my thoughts are a bit scattered right now. Could you try asking again? 🌪️"

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)