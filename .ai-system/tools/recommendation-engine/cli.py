"""CLI for recommendation engine"""
from recommender import PromptRecommender
from learner import PatternLearner
import sys as system

def main():
    if len(system.argv) < 2:
        print("Usage: python cli.py [recommend|workflow|patterns] <query>")
        print("\nExamples:")
        print("  python cli.py recommend 'optimize API performance'")
        print("  python cli.py workflow 'fullstack app'")
        print("  python cli.py patterns")
        return
    
    command = system.argv[1]
    
    if command == "recommend":
        if len(system.argv) < 3:
            print("Error: Provide context for recommendations")
            return
        
        context = " ".join(system.argv[2:])
        recommender = PromptRecommender()
        recs = recommender.recommend(context)
        
        print(f"\n[RECOMMEND] Top prompts for: '{context}'")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['name']}")
            print(f"     Path: {rec['path']}")
            print(f"     Score: {rec['score']:.1f}")
    
    elif command == "workflow":
        if len(system.argv) < 3:
            print("Error: Provide task description")
            return
        
        task = " ".join(system.argv[2:])
        recommender = PromptRecommender()
        steps = recommender.suggest_workflow(task)
        
        print(f"\n[WORKFLOW] Suggested steps for: '{task}'")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    
    elif command == "patterns":
        learner = PatternLearner()
        patterns = learner.analyze_patterns()
        
        print("\n[PATTERNS] Usage Analysis")
        print("\n  Most Successful Prompts:")
        for name, uses, success, quality in patterns['successful_prompts'][:5]:
            print(f"    {name}")
            print(f"      Success: {success:.0%}, Quality: {quality:.2f}, Uses: {uses}")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
