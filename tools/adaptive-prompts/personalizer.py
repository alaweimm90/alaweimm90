"""Personalize prompts based on user preferences"""
from typing import Dict
import json
from pathlib import Path

class PromptPersonalizer:
    def __init__(self):
        self.prefs_path = Path("preferences.json")
        self.preferences = self._load_prefs()
    
    def _load_prefs(self) -> Dict:
        if self.prefs_path.exists():
            return json.loads(self.prefs_path.read_text())
        return {
            'style': 'concise',  # concise, detailed, technical
            'language': 'python',
            'framework': 'fastapi',
            'tone': 'professional'
        }
    
    def save_prefs(self):
        self.prefs_path.write_text(json.dumps(self.preferences, indent=2))
    
    def set_preference(self, key: str, value: str):
        """Set user preference"""
        self.preferences[key] = value
        self.save_prefs()
    
    def personalize(self, prompt_content: str) -> str:
        """Personalize prompt based on preferences"""
        personalized = prompt_content
        
        # Add style note
        style = self.preferences.get('style', 'concise')
        if style == 'concise':
            personalized = f"<!-- Style: Concise responses preferred -->\n\n{personalized}"
        elif style == 'detailed':
            personalized = f"<!-- Style: Detailed explanations preferred -->\n\n{personalized}"
        
        # Replace language placeholders
        lang = self.preferences.get('language', 'python')
        personalized = personalized.replace('{{language}}', lang)
        
        # Replace framework placeholders
        framework = self.preferences.get('framework', 'fastapi')
        personalized = personalized.replace('{{framework}}', framework)
        
        return personalized

if __name__ == "__main__":
    personalizer = PromptPersonalizer()
    
    # Set preferences
    personalizer.set_preference('style', 'detailed')
    personalizer.set_preference('language', 'typescript')
    
    # Personalize prompt
    prompt = "Write {{language}} code using {{framework}}"
    result = personalizer.personalize(prompt)
    print(f"\n[PERSONALIZE]\n{result}")
