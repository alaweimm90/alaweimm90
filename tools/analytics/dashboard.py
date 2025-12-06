"""Simple CLI Dashboard for Prompt Analytics"""
from tracker import PromptTracker
from insights import InsightsGenerator
from datetime import datetime, timedelta

class Dashboard:
    def __init__(self, db_path: str = "prompt_analytics.db"):
        self.tracker = PromptTracker(db_path)
        self.insights = InsightsGenerator(self.tracker)
    
    def show(self, days: int = 30):
        """Display analytics dashboard"""
        print(f"\n{'='*60}")
        print(f"PROMPT ANALYTICS DASHBOARD (Last {days} Days)")
        print(f"{'='*60}\n")
        
        # Overall stats
        stats = self.tracker.get_stats(days)
        print("[STATS] OVERALL STATISTICS")
        print(f"  Total Executions: {stats['total_executions']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Avg Duration: {stats['avg_duration']:.2f}s")
        print(f"  Avg Quality: {stats['avg_quality']:.2f}")
        
        # Top prompts
        print(f"\n[TOP] MOST USED PROMPTS")
        top = self.insights._get_top_prompts(days)
        for i, (name, count) in enumerate(top[:5], 1):
            print(f"  {i}. {name}: {count} uses")
        
        # Insights
        print(f"\n[INSIGHTS] ANALYSIS")
        insights = self.insights.generate_insights(days)
        if not insights:
            print("  [OK] No issues detected")
        else:
            for insight in insights:
                icon = {'info': '[INFO]', 'warning': '[WARN]', 'critical': '[CRIT]'}[insight.severity]
                print(f"  {icon} {insight.title}")
                print(f"        {insight.description}")
        
        # Recommendations
        recs = self.insights.get_recommendations()
        if recs:
            print(f"\n[RECS] RECOMMENDATIONS")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.show()
