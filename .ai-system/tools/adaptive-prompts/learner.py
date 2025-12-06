"""Learn from user feedback and adapt prompts"""
from pathlib import Path
from typing import Dict, List
import json
import sys
sys.path.append(str(Path(__file__).parent.parent / "analytics"))
from tracker import PromptTracker

class AdaptiveLearner:
    def __init__(self):
        self.tracker = PromptTracker()
        self.feedback_path = Path("feedback.json")
        self.feedback = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        if self.feedback_path.exists():
            return json.loads(self.feedback_path.read_text())
        return {}
    
    def save_feedback(self):
        self.feedback_path.write_text(json.dumps(self.feedback, indent=2))
    
    def record_feedback(self, prompt_name: str, feedback_type: str, data: Dict):
        """Record user feedback"""
        if prompt_name not in self.feedback:
            self.feedback[prompt_name] = []
        
        self.feedback[prompt_name].append({
            'type': feedback_type,
            'data': data
        })
        
        self.save_feedback()
    
    def get_improvements(self, prompt_name: str) -> List[str]:
        """Suggest improvements based on feedback"""
        improvements = []
        
        # Check success rate
        try:
            stats = self.tracker.get_stats(30)
            if stats['success_rate'] < 0.8:
                improvements.append("Add more examples to improve clarity")
        except:
            pass
        
        # Check feedback
        if prompt_name in self.feedback:
            feedback_items = self.feedback[prompt_name]
            
            # Count negative feedback
            negative = sum(1 for f in feedback_items if f['type'] == 'negative')
            if negative > 2:
                improvements.append("Review prompt structure based on user feedback")
        
        return improvements
    
    def adapt_prompt(self, prompt_name: str, prompt_content: str) -> str:
        """Adapt prompt based on learning"""
        improvements = self.get_improvements(prompt_name)
        
        if not improvements:
            return prompt_content
        
        # Add improvement suggestions as comments
        adapted = f"<!-- Suggested improvements:\n"
        for imp in improvements:
            adapted += f"  - {imp}\n"
        adapted += "-->\n\n" + prompt_content
        
        return adapted

if __name__ == "__main__":
    learner = AdaptiveLearner()
    
    # Record feedback
    learner.record_feedback("code-review", "positive", {"comment": "Very helpful"})
    learner.record_feedback("code-review", "negative", {"comment": "Too verbose"})
    
    # Get improvements
    improvements = learner.get_improvements("code-review")
    print(f"\n[LEARN] Improvements for code-review:")
    for imp in improvements:
        print(f"  - {imp}")
