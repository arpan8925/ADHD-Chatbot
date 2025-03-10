ADHD Chatbot with NLP (Gemini, Langchain, FAISS, SQLite)
========================================================

Overview
--------

An AI chatbot that supports users with ADHD by:

1.  Handling basic conversations
2.  Analyzing emotional states (e.g., depression mentions)
3.  Checking daily routine compliance from SQLite DB
4.  Providing personalized guidance using Gemini NLP + FAISS embeddings

Workflow Architecture
---------------------

The chatbot follows this workflow:

*   User Input → NLP Processing (Gemini + Langchain)
*   NLP routes to: Basic Greeting or Depression Flag
*   Depression Flag → SQLite DB Check (Sleep/Food/Routine Data)
*   Data Analysis → Guidance Generation or General Advice
*   Response delivered to user
*   Interaction logged (SQLite + FAISS)

Note: A full mermaid diagram is available in the original documentation.

Step-by-Step Execution Plan
---------------------------

### Depression Detection Flow

Trigger Detection:

`depression_triggers = ["depressed", "sad", "hopeless", "overwhelmed"] def check_depression(input_text):return any(trigger in input_text.lower() for trigger in depression_triggers)`

Routine Check Logic:

`def evaluate_routine(user_id): cursor.execute('''SELECT last_sleep_hours, last_meal_time FROM users WHERE id=?''', (user_id,)) data = cursor.fetchone()  # ADHD-Specific Thresholds sleep_ok = data[0] >= 7 if data[0] else False  meal_ok = (datetime.now() - parse(data[1])).hours < 4 if data[1] else False return {"needs_sleep_help": not sleep_ok,   "needs_nutrition_help": not meal_ok}`

### Response Generation

Guidance Logic:

`guidance_prompt = """ You have ADHD. Based on these issues: {issues_list} Provide 3 concise suggestions. Format: 1. [Sleep Tip] 2. [Nutrition Tip] 3. [Movement Tip]   """ def generate_guidance(issues): return Gemini().generate(guidance_prompt.format(issues_list=issues))`

### Interaction Logging

`def log_interaction(user_id, message, response): 
cursor.execute('''INSERT INTO interactions (user_id, message, response) VALUES (?,?,?)''',(user_id, message, response)) # FAISS vector_db.add_texts([f"User: {message}", f"Bot: {response}"])`

Execution Flow
--------------

1.  **Input Received:** User sends message
2.  **NLP Routing:**
    *   Detect greeting → Basic response
    *   Detect depression trigger → Routine check
3.  **Database Query:** Fetch sleep/meal data
4.  **Analysis:** Compare against ADHD-healthy thresholds
5.  **Response Generation:**
    *   Specific guidance if routine issues found
    *   General coping strategies if routine OK
6.  **Logging:** Store interaction in both DBs

ADHD-Specific Thresholds
------------------------

Metric 

Healthy Range

Sleep

7-9 hours nightly

Meal Frequency

Every 3-4 hours

Task Breakdown

25-min focused blocks

Future Enhancements
-------------------

*   Mood tracking visualization
*   Medication reminder system
*   ADHD-friendly task breakdown engine