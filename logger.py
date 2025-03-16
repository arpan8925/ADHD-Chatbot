import sqlite3
import datetime

LOG_DB_PATH = "logs.db"

class Logger:
    def __init__(self, db_path=LOG_DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_tables()

    def _initialize_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event TEXT,
            details TEXT
        )
        """)
        self.conn.commit()

    def log_event(self, event, details=""):
        timestamp = datetime.datetime.now().isoformat()
        self.cursor.execute("""
        INSERT INTO logs (timestamp, event, details) VALUES (?, ?, ?)
        """, (timestamp, event, details))
        self.conn.commit()

    def get_logs(self, limit=10):
        """Retrieves recent logs for debugging."""
        self.cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return self.cursor.fetchall()
