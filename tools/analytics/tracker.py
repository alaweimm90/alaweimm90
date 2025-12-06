"""Prompt usage tracking and analytics."""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict

class PromptTracker:
    def __init__(self, db_path: str = "prompt_analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_usage (
                id INTEGER PRIMARY KEY,
                prompt_name TEXT,
                timestamp TEXT,
                success BOOLEAN,
                duration REAL,
                quality REAL
            )
        """)
        conn.commit()
        conn.close()
    
    def log_usage(self, prompt_name: str, success: bool, duration: float, quality: float):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO prompt_usage (prompt_name, timestamp, success, duration, quality) VALUES (?, ?, ?, ?, ?)",
            (prompt_name, datetime.now().isoformat(), success, duration, quality)
        )
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def get_stats(self, days: int = 30) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                AVG(duration) as avg_duration,
                AVG(quality) as avg_quality
            FROM prompt_usage
        """)
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        return {
            'total_executions': total,
            'success_rate': (row[1] / total) if total else 0,
            'avg_duration': row[2] or 0,
            'avg_quality': row[3] or 0
        }

if __name__ == '__main__':
    tracker = PromptTracker()
    tracker.initialize()
    print("[OK] Analytics tracker initialized")
