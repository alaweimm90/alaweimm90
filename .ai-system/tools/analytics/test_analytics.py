"""Test Analytics System"""
from tracker import PromptTracker
from insights import InsightsGenerator
from dashboard import Dashboard
from datetime import datetime
import time

def test_full_analytics():
    """Test complete analytics pipeline"""
    print("\n[TEST] Full Analytics Pipeline")
    
    # Initialize
    tracker = PromptTracker("test_analytics.db")
    tracker.initialize()
    
    # Simulate usage data
    print("  Generating test data...")
    test_data = [
        ("code-review", True, 2.5, 0.85),
        ("code-review", True, 2.3, 0.88),
        ("code-review", False, 5.2, 0.65),
        ("refactoring", True, 1.8, 0.92),
        ("refactoring", True, 1.9, 0.90),
        ("optimization", True, 3.1, 0.78),
        ("optimization", False, 6.5, 0.55),
        ("testing", True, 2.0, 0.85),
    ]
    
    for name, success, duration, quality in test_data:
        tracker.log_usage(name, success, duration, quality)
        time.sleep(0.01)  # Small delay for timestamp variation
    
    # Get stats
    stats = tracker.get_stats(30)
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Avg duration: {stats['avg_duration']:.2f}s")
    print(f"  Avg quality: {stats['avg_quality']:.2f}")
    
    # Generate insights
    insights_gen = InsightsGenerator(tracker)
    insights = insights_gen.generate_insights(30)
    print(f"  Generated {len(insights)} insights")
    
    # Show dashboard
    dashboard = Dashboard("test_analytics.db")
    dashboard.show(30)
    
    print("[OK] Analytics pipeline test complete")

if __name__ == "__main__":
    test_full_analytics()
