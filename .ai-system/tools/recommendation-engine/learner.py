"""Learn from user patterns"""
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).parent.parent / "analytics"))
from tracker import PromptTracker

class PatternLearner:
    def __init__(self):
        self.tracker = PromptTracker()
    
    def analyze_patterns(self) -> Dict:
        """Analyze usage patterns"""
        conn = self.tracker._get_connection()
        
        # Most successful prompts
        cursor = conn.execute("""
            SELECT prompt_name, 
                   COUNT(*) as uses,
                   AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                   AVG(quality) as avg_quality
            FROM prompt_usage
            GROUP BY prompt_name
            HAVING uses > 1
            ORDER BY success_rate DESC, avg_quality DESC
            LIMIT 10
        """)
        successful = cursor.fetchall()
        
        # Time patterns
        cursor = conn.execute("""
            SELECT strftime('%H', timestamp) as hour,
                   COUNT(*) as uses
            FROM prompt_usage
            GROUP BY hour
            ORDER BY uses DESC
            LIMIT 5
        """)
        time_patterns = cursor.fetchall()
        
        conn.close()
        
        return {
            'successful_prompts': successful,
            'peak_hours': time_patterns
        }
    
    def get_similar_prompts(self, prompt_name: str) -> List[str]:
        """Find prompts often used together"""
        conn = self.tracker._get_connection()
        
        # Get timestamps for this prompt
        cursor = conn.execute("""
            SELECT timestamp FROM prompt_usage
            WHERE prompt_name = ?
            ORDER BY timestamp
        """, (prompt_name,))
        
        timestamps = [row[0] for row in cursor.fetchall()]
        
        # Find prompts used within 1 hour
        similar = defaultdict(int)
        for ts in timestamps:
            cursor = conn.execute("""
                SELECT prompt_name, COUNT(*) as count
                FROM prompt_usage
                WHERE prompt_name != ?
                AND ABS(julianday(timestamp) - julianday(?)) < 0.042
                GROUP BY prompt_name
            """, (prompt_name, ts))
            
            for name, count in cursor.fetchall():
                similar[name] += count
        
        conn.close()
        
        return sorted(similar.items(), key=lambda x: x[1], reverse=True)[:5]

if __name__ == "__main__":
    learner = PatternLearner()
    patterns = learner.analyze_patterns()
    
    print("\n[LEARN] Usage Patterns")
    print("\n  Most Successful Prompts:")
    for name, uses, success, quality in patterns['successful_prompts'][:5]:
        print(f"    {name}: {success:.0%} success, {quality:.2f} quality ({uses} uses)")
