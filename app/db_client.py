import sqlite3
import json

class DatabaseClient:
    def __init__(self, db_path="idp_results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                document_type TEXT,
                trust_score REAL,
                extracted_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def save_result(self, filename, doc_type, trust_score, result_dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO extractions (filename, document_type, trust_score, extracted_json)
            VALUES (?, ?, ?, ?)
        ''', (filename, doc_type, trust_score, json.dumps(result_dict)))
        conn.commit()
        conn.close()

    def get_history(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM extractions ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
