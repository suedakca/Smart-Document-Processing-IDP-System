import sqlite3
import json
from loguru import logger
from datetime import datetime, timedelta

class IntelligenceEngine:
    def __init__(self, db_path="idp_results.db"):
        self.db_path = db_path

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_platform_insights(self):
        """Calculates high-level intelligence metrics for the platform."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            # 1. Total Efficiency Gain (Hypothetical: 5 mins saved per verified doc)
            cursor.execute("SELECT COUNT(*) FROM extractions WHERE trust_score > 0.85")
            efficient_docs = cursor.fetchone()[0]
            hours_saved = (efficient_docs * 5) / 60
            
            # 2. Accuracy Trend (Last 7 days)
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            cursor.execute("SELECT AVG(trust_score) FROM extractions WHERE created_at >= ?", (seven_days_ago,))
            recent_accuracy = cursor.fetchone()[0] or 0.0
            
            # 3. Learning Progress
            cursor.execute("SELECT COUNT(*) FROM extractions WHERE is_verified = 1")
            learning_points = cursor.fetchone()[0]
            
            # 4. Anomaly Count (Low trust score or high processing time)
            cursor.execute("SELECT COUNT(*) FROM extractions WHERE trust_score < 0.5 OR processing_time > 30")
            anomalies = cursor.fetchone()[0]

            return {
                "efficiency": {
                    "hours_saved": round(hours_saved, 1),
                    "automation_rate": round((efficient_docs / (self._get_total_count() or 1)) * 100, 1)
                },
                "accuracy_score": round(recent_accuracy * 100, 1),
                "knowledge_base_size": learning_points,
                "active_anomalies": anomalies
            }
        except Exception as e:
            logger.error(f"Intelligence Engine Error: {str(e)}")
            return {}
        finally:
            conn.close()

    def detect_anomalies(self, limit=5):
        """Identifies recent extractions that require immediate data intelligence review."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename, document_type, trust_score, processing_time, created_at 
            FROM extractions 
            WHERE trust_score < 0.6 OR processing_time > 40
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def _get_total_count(self):
        conn = self._get_conn()
        res = conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
        conn.close()
        return res
