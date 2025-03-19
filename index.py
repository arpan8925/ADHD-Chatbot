from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import os
from database import Database
from faiss_memory import FAISSMemory
from llms import LLMRoutine, LLMMemory, LLMChat

# Initialize Flask App
app = Flask(__name__, template_folder="templates")

# Load API Key
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("ERROR: Gemini API key is missing. Please set it in the .env file.")

genai.configure(api_key=api_key)

# Initialize the database
db_instance = Database()  # No need to pass the path since the default is set

# Initialize the vector database (FAISS)
memory_instance = FAISSMemory()

# Initialize LLMs
llm_routine = LLMRoutine()
llm_memory = LLMMemory(memory_instance=memory_instance)
llm_chat = LLMChat()

# ✅ **Embedding Model**
class EmbeddingModel:
    """Handles text embeddings using SentenceTransformer."""
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.history = {}  # Dictionary to store user conversations

    def get_embedding(self, user_message):
        """Encodes a user message into an embedding vector."""
        return self.model.encode(user_message, convert_to_numpy=True).astype(np.float32)

    def add_message(self, user_id, user_message):
        """Adds message to history and stores embeddings in FAISS & SQLite."""
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].append(user_message)

        if not user_message:
            return jsonify({"response": "I'm here to listen! What’s on your mind? 😊"})

        # Store text conversation in SQLite
        db_instance.store_conversation_history(user_id, user_message)

        # Generate and store embedding in FAISS
        embedding = self.get_embedding(user_message)
        memory_instance.store_embedding(user_id, user_message, embedding)

        return jsonify({"response": self.generate_ai_response(user_id, user_message)})

    def retrieve_past_messages(self, user_id):
        """Fetches the last 5 messages of a user from conversation history."""
        db_instance.cursor.execute(
            "SELECT message FROM conversation_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
            (user_id,)
        )
        return [row[0] for row in db_instance.cursor.fetchall()]

    def generate_ai_response(self, user_id, user_message):
        """Provides conversational ADHD support, referencing past conversations and user routines."""
        # Fetch user routine from SQLite
        user_routine = db_instance.get_user_routine(user_id)

        # Retrieve past messages from FAISS
        embedding = self.get_embedding(user_message)
        similar_messages = memory_instance.retrieve_similar_messages(embedding, user_id)

        # Format past messages as context
        past_messages = "\n".join([f"{msg['timestamp']}: {msg['message']}" for msg in similar_messages])

        prompt = f"""
        User Message: "{user_message}"
        User Routine: {user_routine if user_routine else "No routine found."}
        Past Conversations: {past_messages if past_messages else "No past data yet."}


        AI Response:
        """

        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text.strip()


# ✅ **Create Global Embedding Instance**
embedding_instance = EmbeddingModel()


# 🏡 **Home Route**
@app.route("/")
def home():
    return render_template("index.html")


# ✨ **Chat Route**
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "123")
    user_message = data.get("message", "").strip().lower()

    if not user_message:
        return jsonify({"response": "I'm here to listen! What’s on your mind? 😊"})

    # Determine which LLM to use
    if "routine" in user_message :
        response = llm_routine.generate_routine(user_id, user_message)
    elif "why do I feel" in user_message or "I feel" in user_message:
        response = llm_memory.analyze_emotions(user_id, user_message, embedding_instance, db_instance)
    else:
        response = llm_chat.chat_with_user(user_id, user_message, db_instance)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
