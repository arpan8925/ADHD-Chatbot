import google.generativeai as genai
from faiss_memory import FAISSMemory
from database import Database
import numpy as np
import re
from cachetools import TTLCache

# 📌 Time Extraction Function
def extract_time(user_message):
    """Extracts time (e.g., '3pm', '14:30') from user messages."""
    match = re.search(r'(\d{1,2}):?(\d{0,2}) ?(am|pm)?', user_message)
    if match:
        hour, minute, period = match.groups()
        if not minute:
            minute = "00"
        if period:
            period = period.lower()
            if period == "pm" and int(hour) < 12:
                hour = str(int(hour) + 12)
            elif period == "am" and hour == "12":
                hour = "00"
        return f"{hour}:{minute}"
    return None

# Load AI Model
class LLMBase:
    """Base class for all LLMs"""
    def __init__(self, model_name="gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model_name)
    
    def call_llm(self, prompt, system_prompt=None, response_format=None, max_tokens=None):
        response = self.model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else "I'm here for you. Want to talk about it? 💙"



# 🏡 **LLM1 - Routine Generator**

class LLMRoutine(LLMBase):
    """Handles routine scheduling using LLM's NLP capabilities"""

    def __init__(self):
        super().__init__()
        # Using a TTLCache to store data temporarily for the session duration
        self.session_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache expires in 1 hour

    def generate_routine(self, user_id, user_message):
        # Construct the system prompt
        system_prompt = """You are an expert schedule planner. Your task is to:
    1. Parse all activities with explicit times from the user's message
    2. Generate a complete 24-hour schedule filling gaps with healthy/productive activities
    3. Format as an HTML table with time slots and activities

    Follow these steps:
    1. Extract ALL activities with times mentioned (even implicit ones)
    2. Convert all times to 12-hour format with AM/PM
    3. Start the schedule **at the user's wake-up time** and continue in 1-hour increments until sleep time, if wake-up time and sleep time not given, ask for it to the user..
    4. Fill empty slots with appropriate activities from these categories:
    - Physical exercise
    - Mental health
    - Skill development
    - Nutrition
    - Personal growth
    5. Add emojis to make it engaging
    6. Return ONLY valid HTML table with 2 columns (Time, Activity)

    Example format:
    <!DOCTYPE html>
    <table>
    <tr><th>Time</th><th>Activity</th></tr>
    ...
    </table>"""

        # Combine prompt and user message
        prompt = f"{system_prompt}\n\nUser Input: {user_message}"

        # Get LLM response using the base class method
        response = self.call_llm(prompt)

        # Store explicit activities in session cache
        self._store_explicit_activities(user_id, user_message)

        return response

    def _store_explicit_activities(self, user_id, user_message):
        # Use LLM to extract explicit time-activity pairs
        extraction_prompt = """Extract ALL (time, activity) pairs from this message. 
Return ONLY as JSON format: [{"time": "<time> AM/PM", "activity": "<activity>"}]"""

        result = self.call_llm(
            user_message,
            system_prompt=extraction_prompt,
            response_format="json",
            max_tokens=500
        )

        if isinstance(result, list):
            self.session_cache[user_id] = result  # Store data temporarily

    def get_cached_routine(self, user_id):
        """Retrieve the cached routine for the given user ID."""
        return self.session_cache.get(user_id, [])


# 🧠 **LLM2 - Memory & Emotional Analysis**
class LLMMemory:
    def __init__(self, model_name="gemini-1.5-pro", memory_instance=None):
        self.model = genai.GenerativeModel(model_name)
        self.memory_instance = memory_instance if memory_instance else FAISSMemory()

    def analyze_emotions(self, user_id, user_message, embedding_model, db_instance):
        """Finds past conversations to explain emotional states"""
        # Generate embedding for user message
        query_embedding = embedding_model.get_embedding(user_message)

        # Retrieve past messages from FAISS
        similar_conversations = self.memory_instance.retrieve_similar_messages(query_embedding, user_id)

        # Get user routine from SQLite
        user_routine = db_instance.get_user_routine(user_id)

        # ✅ Ensure there's a fallback response if no past data exists
        if not similar_conversations:
            return "I'm here for you. Do you want to talk about why you feel this way? 💙"

        # Format past messages as context
        past_messages = "\n".join([f"{msg['timestamp']}: {msg['message']}" for msg in similar_conversations])

        prompt = f"""
        User Message: "{user_message}"
        Past Context: 
        {past_messages if past_messages else "No past data yet"}
        User Routine: {user_routine if user_routine else "No routine found"}

        Based on these past interactions, why might the user be feeling this way? 
        Suggest possible reasons and motivational advice.
        """

        return self.model.generate_content(prompt).text.strip()

# 🤝 **LLM3 - ADHD Caregiver AI**
class LLMChat(LLMBase):
    """Handles general ADHD-friendly conversations and encouragement"""
    def chat_with_user(self, user_id, user_message, db_instance):
        """Provides ADHD-friendly guidance, referring to past conversations and user routines."""
        # Fetch user routine from SQLite
        user_routine = db_instance.get_user_routine(user_id)
        
        # Retrieve past messages for context
        past_messages = db_instance.get_conversation_history(user_id)

        prompt = f"""
        User Message: "{user_message}"
        User Routine: {user_routine if user_routine else "No routine found"}
        Past Conversations: {', '.join(past_messages) if past_messages else "No past data yet"}

        ADHD Considerations:
        - Encourage small steps (task chunking)
        - Offer motivational reinforcement
        - Refer to the user's routine when appropriate

        AI Response:
        """

        return self.call_llm(prompt)
