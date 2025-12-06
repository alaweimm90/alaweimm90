"""Test adaptive prompts"""
from learner import AdaptiveLearner
from personalizer import PromptPersonalizer

def test_adaptive():
    print("\n[TEST] Adaptive Prompts")
    
    # Test 1: Learning from feedback
    print("\n  Test 1: Feedback Learning")
    learner = AdaptiveLearner()
    
    learner.record_feedback("optimization", "positive", {"comment": "Clear and helpful"})
    learner.record_feedback("optimization", "positive", {"comment": "Great examples"})
    learner.record_feedback("testing", "negative", {"comment": "Too complex"})
    learner.record_feedback("testing", "negative", {"comment": "Needs more examples"})
    learner.record_feedback("testing", "negative", {"comment": "Unclear instructions"})
    
    improvements = learner.get_improvements("testing")
    print(f"    Improvements for 'testing': {len(improvements)}")
    for imp in improvements:
        print(f"      - {imp}")
    
    # Test 2: Prompt adaptation
    print("\n  Test 2: Prompt Adaptation")
    original = "# Testing Prompt\n\nWrite tests for your code."
    adapted = learner.adapt_prompt("testing", original)
    has_suggestions = "Suggested improvements" in adapted
    print(f"    Adapted prompt has suggestions: {has_suggestions}")
    
    # Test 3: Personalization
    print("\n  Test 3: Personalization")
    personalizer = PromptPersonalizer()
    
    personalizer.set_preference('style', 'concise')
    personalizer.set_preference('language', 'rust')
    personalizer.set_preference('framework', 'actix')
    
    prompt = "Write {{language}} code using {{framework}}"
    personalized = personalizer.personalize(prompt)
    
    has_rust = 'rust' in personalized.lower()
    has_actix = 'actix' in personalized.lower()
    print(f"    Language substituted: {has_rust}")
    print(f"    Framework substituted: {has_actix}")
    
    # Test 4: Preference persistence
    print("\n  Test 4: Preference Persistence")
    personalizer2 = PromptPersonalizer()
    lang = personalizer2.preferences.get('language')
    print(f"    Loaded language preference: {lang}")
    
    print("\n[OK] Adaptive prompts test complete")

if __name__ == "__main__":
    test_adaptive()
