"""Test recommendation engine"""
from recommender import PromptRecommender
from learner import PatternLearner
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "analytics"))
from tracker import PromptTracker

def test_recommendations():
    print("\n[TEST] Recommendation Engine")
    
    # Setup test data
    tracker = PromptTracker()
    tracker.initialize()
    
    # Add test usage data
    test_prompts = [
        ("optimization-framework", True, 2.1, 0.90),
        ("optimization-framework", True, 2.3, 0.88),
        ("code-review", True, 1.5, 0.85),
        ("testing-qa-strategy", True, 1.8, 0.87),
        ("api-design-development", True, 2.0, 0.89),
    ]
    
    for name, success, duration, quality in test_prompts:
        tracker.log_usage(name, success, duration, quality)
    
    # Test recommendations
    recommender = PromptRecommender()
    
    print("\n  Test 1: API optimization context")
    context = "I need to optimize the performance of my API"
    recs = recommender.recommend(context, limit=3)
    for i, rec in enumerate(recs, 1):
        print(f"    {i}. {rec['name']} (score: {rec['score']:.1f}, keywords: {rec['keywords_matched']})")
    
    print("\n  Test 2: Testing context")
    context = "Need to write tests for my application"
    recs = recommender.recommend(context, limit=3)
    for i, rec in enumerate(recs, 1):
        print(f"    {i}. {rec['name']} (score: {rec['score']:.1f})")
    
    print("\n  Test 3: Workflow suggestion")
    task = "Build a fullstack application"
    workflow = recommender.suggest_workflow(task)
    print(f"    Suggested steps: {' -> '.join(workflow)}")
    
    # Test pattern learning
    print("\n  Test 4: Pattern Learning")
    learner = PatternLearner()
    patterns = learner.analyze_patterns()
    
    if patterns['successful_prompts']:
        print(f"    Found {len(patterns['successful_prompts'])} successful prompts")
        top = patterns['successful_prompts'][0]
        print(f"    Top: {top[0]} ({top[2]:.0%} success)")
    
    print("\n[OK] Recommendation engine test complete")

if __name__ == "__main__":
    test_recommendations()
