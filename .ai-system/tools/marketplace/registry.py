"""Prompt marketplace registry"""
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class PromptRegistry:
    def __init__(self, registry_path: str = "registry.json"):
        self.registry_path = Path(registry_path)
        self.prompts = self._load()
    
    def _load(self) -> Dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}
    
    def save(self):
        self.registry_path.write_text(json.dumps(self.prompts, indent=2))
    
    def publish(self, prompt_data: Dict) -> str:
        """Publish a prompt to marketplace"""
        prompt_id = f"{prompt_data['author']}/{prompt_data['name']}"
        
        self.prompts[prompt_id] = {
            **prompt_data,
            'id': prompt_id,
            'published_at': datetime.now().isoformat(),
            'downloads': 0,
            'rating': 0.0,
            'reviews': []
        }
        
        self.save()
        return prompt_id
    
    def search(self, query: str) -> List[Dict]:
        """Search prompts"""
        results = []
        query_lower = query.lower()
        
        for prompt_id, data in self.prompts.items():
            if (query_lower in data['name'].lower() or 
                query_lower in data.get('description', '').lower() or
                any(query_lower in tag.lower() for tag in data.get('tags', []))):
                results.append(data)
        
        return sorted(results, key=lambda x: x.get('downloads', 0), reverse=True)
    
    def get(self, prompt_id: str) -> Dict:
        """Get prompt details"""
        return self.prompts.get(prompt_id)
    
    def rate(self, prompt_id: str, rating: float, review: str = None):
        """Rate a prompt"""
        if prompt_id not in self.prompts:
            return
        
        prompt = self.prompts[prompt_id]
        reviews = prompt.get('reviews', [])
        reviews.append({'rating': rating, 'review': review, 'date': datetime.now().isoformat()})
        
        # Update average rating
        avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
        prompt['rating'] = avg_rating
        prompt['reviews'] = reviews
        
        self.save()
    
    def download(self, prompt_id: str):
        """Track download"""
        if prompt_id in self.prompts:
            self.prompts[prompt_id]['downloads'] += 1
            self.save()

if __name__ == "__main__":
    registry = PromptRegistry()
    
    # Publish test prompt
    prompt_id = registry.publish({
        'name': 'code-review-expert',
        'author': 'alawein',
        'description': 'Expert code review prompt',
        'tags': ['code-review', 'quality'],
        'version': '1.0.0'
    })
    
    print(f"[PUBLISH] {prompt_id}")
