import sqlite3
import json
import uuid
from loguru import logger

class DatabaseClient:
    def __init__(self, db_path="data/idp_results.db"):
        self.db_path = db_path
        # Ensure data directory exists
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
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
                task_id TEXT UNIQUE,
                filename TEXT,
                document_type TEXT,
                trust_score REAL,
                extracted_json TEXT,
                status TEXT DEFAULT 'SUCCESS',
                is_verified INTEGER DEFAULT 0,
                corrected_json TEXT,
                validation_report TEXT,
                processing_time REAL,
                error_type TEXT,
                api_key_id INTEGER,
                raw_text TEXT,
                approved_by TEXT,
                approved_at TIMESTAMP,
                corrected_fields TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(api_key_id) REFERENCES api_keys(id)
            )
        ''')
        
        # Robust Migration: Check and add missing columns
        cursor.execute("PRAGMA table_info(extractions)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        
        migration_targets = [
            ("task_id", "TEXT"),
            ("status", "TEXT DEFAULT 'SUCCESS'"),
            ("is_verified", "INTEGER DEFAULT 0"),
            ("corrected_json", "TEXT"),
            ("validation_report", "TEXT"),
            ("processing_time", "REAL"),
            ("error_type", "TEXT"),
            ("api_key_id", "INTEGER"),
            ("raw_text", "TEXT"),
            ("approved_by", "TEXT"),
            ("approved_at", "TIMESTAMP"),
            ("corrected_fields", "TEXT")
        ]
        
        for col, col_def in migration_targets:
            if col not in existing_cols:
                logger.info(f"Database Migration: Adding column {col} to extractions table")
                cursor.execute(f"ALTER TABLE extractions ADD COLUMN {col} {col_def}")
                
        conn.commit()
        conn.close()

    def save_correction(self, task_id, corrected_data, approved_by="User", corrected_fields=None):
        """Saves user correction and audit trail."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        import datetime
        now = datetime.datetime.now().isoformat()
        
        cursor.execute('''
            UPDATE extractions 
            SET corrected_json = ?, 
                is_verified = 1, 
                approved_by = ?, 
                approved_at = ?, 
                corrected_fields = ?,
                status = 'APPROVED'
            WHERE task_id = ?
        ''', (json.dumps(corrected_data), approved_by, now, json.dumps(corrected_fields or []), task_id))
        conn.commit()
        conn.close()

    def save_result(self, task_id, filename, doc_type, trust_score, result_dict, processing_time=0.0, status="SUCCESS", validation_report=None, key_id=None, raw_text=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO extractions (task_id, filename, document_type, trust_score, extracted_json, status, processing_time, validation_report, api_key_id, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (task_id, filename, doc_type, trust_score, json.dumps(result_dict), status, processing_time, json.dumps(validation_report or {}), key_id, raw_text))
        conn.commit()
        conn.close()

    def get_pending_reviews(self, limit=20):
        """Retrieves documents flagged for human review."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM extractions 
            WHERE status = 'REVIEW_REQUIRED' 
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_verified_examples(self, doc_type, limit=3):
        """Retrieves best corrected examples for Dynamic Few-Shot."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT raw_text, corrected_json FROM extractions 
            WHERE document_type = ? AND is_verified = 1 
            ORDER BY created_at DESC LIMIT ?
        ''', (doc_type, limit))
        rows = cursor.fetchall()
        conn.close()
        return [{"raw_text": row["raw_text"], "corrected_json": row["corrected_json"]} for row in rows]

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
