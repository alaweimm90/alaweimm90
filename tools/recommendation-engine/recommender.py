"""AI-powered prompt recommendation engine"""
from pathlib import Path
from typing import List, Dict
from collections import Counter
import sys
sys.path.append(str(Path(__file__).parent.parent / "analytics"))
from tracker import PromptTracker

class PromptRecommender:
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self.tracker = PromptTracker()
    
    def recommend(self, context: str, limit: int = 5) -> List[Dict]:
        """Recommend prompts based on context"""
        keywords = self._extract_keywords(context)
        prompts = self._find_matching_prompts(keywords)
        scored = self._score_prompts(prompts, keywords)
        return sorted(scored, key=lambda x: x['score'], reverse=True)[:limit]
    
    def _extract_keywords(self, context: str) -> List[str]:
        """Extract keywords from context"""
        keywords = ['test', 'review', 'optimize', 'refactor', 'debug', 'architecture', 
                   'security', 'performance', 'api', 'database', 'frontend', 'backend']
        return [k for k in keywords if k in context.lower()]
    
    def _find_matching_prompts(self, keywords: List[str]) -> List[Path]:
        """Find prompts matching keywords"""
        matches = []
        for prompt_file in self.prompts_dir.rglob("*.md"):
            content = prompt_file.read_text(encoding='utf-8').lower()
            if any(kw in content for kw in keywords):
                matches.append(prompt_file)
        return matches
    
    def _score_prompts(self, prompts: List[Path], keywords: List[str]) -> List[Dict]:
        """Score prompts by relevance and usage"""
        results = []
        for prompt in prompts:
            content = prompt.read_text(encoding='utf-8').lower()
            
            # Keyword match score
            keyword_score = sum(1 for kw in keywords if kw in content)
            
            # Usage score from analytics
            prompt_name = prompt.stem
            stats = self.tracker.get_stats(30)
            usage_score = stats.get('total_executions', 0) * 0.1
            
            total_score = keyword_score + usage_score
            
            results.append({
                'path': str(prompt.relative_to(self.prompts_dir)),
                'name': prompt.stem,
                'score': total_score,
                'keywords_matched': keyword_score
            })
        
        return results
    
    def suggest_workflow(self, task: str) -> List[str]:
        """Suggest workflow steps for a task"""
        workflows = {
            'fullstack': ['architecture', 'backend', 'frontend', 'testing', 'deployment'],
            'optimization': ['profiling', 'analysis', 'implementation', 'benchmarking'],
            'refactoring': ['review', 'planning', 'implementation', 'testing'],
            'api': ['design', 'implementation', 'documentation', 'testing']
        }
        
        for key, steps in workflows.items():
            if key in task.lower():
                return steps
        
        return ['planning', 'implementation', 'testing']

if __name__ == "__main__":
    recommender = PromptRecommender()
    
    context = "I need to optimize the performance of my API"
    recommendations = recommender.recommend(context)
    
    print(f"\n[RECOMMEND] Top recommendations for: '{context}'")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['name']} (score: {rec['score']:.1f})")
