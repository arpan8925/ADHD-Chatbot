import sqlite3
import faiss
import numpy as np

# SQLite Database Setup
class Database:
    def __init__(self, db_path="adhd_assistant.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        # Removed unnecessary execute call during initialization

    # Create or Modify Tables
    def initialize_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_routine (
                user_id TEXT,
                date TEXT DEFAULT (DATE('now')),
                activity TEXT,
                time TEXT,
                PRIMARY KEY (user_id, date, activity)
            )
        ''')
        self.conn.commit()

    def create_conversation_history_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                user_id TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def store_user_routine(self, user_id, activity, parsed_time):
       """Stores user chat history in conversation_history table."""
       self.cursor.execute('''
            INSERT INTO user_routine (user_id, activity, time)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, activity) DO UPDATE SET time=EXCLUDED.time
        ''', (user_id, activity, parsed_time))
       self.conn.commit()

    def store_conversation_history(self, user_id, message):
        """Stores user chat history in conversation_history table."""
        self.cursor.execute(
            "INSERT INTO conversation_history (user_id, message) VALUES (?, ?)", 
            (user_id, message)
        )
        self.conn.commit()

    def get_user_routine(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT activity FROM user_routine WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_conversation_history(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT message FROM conversation_history WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()
