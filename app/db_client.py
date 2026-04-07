import sqlite3
import json

class DatabaseClient:
    def __init__(self, db_path="idp_results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Users/Departments
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 2. API Keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT UNIQUE,
                user_id INTEGER,
                label TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        
        # 3. Extractions & Metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                document_type TEXT,
                trust_score REAL,
                extracted_json TEXT,
                is_verified INTEGER DEFAULT 0,
                corrected_json TEXT,
                processing_time REAL,
                error_type TEXT,
                api_key_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(api_key_id) REFERENCES api_keys(id)
            )
        ''')
        conn.commit()
        conn.close()

    def save_correction(self, job_id, corrected_data):
        """Saves user correction to build 'Ground Truth' for learning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Note: job_id here is the celery job id, but extractions table has integer id.
        # We need to map job_id to extraction record. Let's assume we store job_id in extraction or use integer id.
        # Fixed: save_result should store job_id. Let's add job_id column.
        cursor.execute('''
            UPDATE extractions 
            SET corrected_json = ?, is_verified = 1 
            WHERE id = (SELECT id FROM extractions WHERE filename LIKE ? ORDER BY created_at DESC LIMIT 1)
        ''', (json.dumps(corrected_data), f"%{job_id}%"))
        conn.commit()
        conn.close()

    def get_verified_examples(self, doc_type, limit=3):
        """Retrieves best corrected examples for Dynamic Few-Shot."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT document_type, corrected_json FROM extractions 
            WHERE document_type = ? AND is_verified = 1 
            ORDER BY created_at DESC LIMIT ?
        ''', (doc_type, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def create_api_key(self, user_name, label="Default"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (user_name,))
        cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
        uid = cursor.fetchone()[0]
        
        # Simplified for demo: return plain key and store plain (real version uses hashes)
        raw_key = f"sk_{uuid.uuid4().hex[:12]}"
        cursor.execute("INSERT INTO api_keys (key_hash, user_id, label) VALUES (?, ?, ?)", (raw_key, uid, label))
        conn.commit()
        conn.close()
        return raw_key

    def verify_key(self, key):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM api_keys WHERE key_hash = ? AND is_active = 1", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def save_result(self, filename, doc_type, trust_score, result_dict, processing_time=0.0, error_type=None, key_id=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO extractions (filename, document_type, trust_score, extracted_json, processing_time, error_type, api_key_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, doc_type, trust_score, json.dumps(result_dict), processing_time, error_type, key_id))
        conn.commit()
        conn.close()

    def get_stats(self):
        """Returns statistics for Dashboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Doc distribution
        cursor.execute("SELECT document_type, COUNT(*) FROM extractions GROUP BY document_type")
        docs = dict(cursor.fetchall())
        
        # Success Rate (trust_score > 0.90 as proxy)
        cursor.execute("SELECT COUNT(*) FROM extractions WHERE trust_score >= 0.90")
        success = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM extractions")
        total = cursor.fetchone()[0] or 1
        
        # Average processing time
        cursor.execute("SELECT AVG(processing_time) FROM extractions")
        avg_time = cursor.fetchone()[0] or 0.0
        
        conn.close()
        return {
            "doc_types": docs,
            "success_rate": (success / total) * 100,
            "avg_processing_time": round(avg_time, 2),
            "total_documents": total
        }

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
