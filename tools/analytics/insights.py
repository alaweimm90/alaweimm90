"""Prompt Analytics Insights Generator"""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta
import statistics

@dataclass
class Insight:
    type: str  # performance, usage, quality, recommendation
    title: str
    description: str
    severity: str  # info, warning, critical
    data: Dict

class InsightsGenerator:
    def __init__(self, tracker):
        self.tracker = tracker
    
    def generate_insights(self, days: int = 30) -> List[Insight]:
        """Generate insights from usage data"""
        insights = []
        stats = self.tracker.get_stats(days)
        
        # Performance insights
        if stats['avg_duration'] > 5.0:
            insights.append(Insight(
                type='performance',
                title='Slow Prompt Execution',
                description=f'Average duration {stats["avg_duration"]:.1f}s exceeds 5s threshold',
                severity='warning',
                data={'avg_duration': stats['avg_duration']}
            ))
        
        # Success rate insights
        if stats['success_rate'] < 0.8:
            insights.append(Insight(
                type='quality',
                title='Low Success Rate',
                description=f'Success rate {stats["success_rate"]:.1%} below 80% target',
                severity='critical',
                data={'success_rate': stats['success_rate']}
            ))
        
        # Usage patterns
        top_prompts = self._get_top_prompts(days)
        if top_prompts:
            insights.append(Insight(
                type='usage',
                title='Most Used Prompts',
                description=f'Top prompt: {top_prompts[0][0]} ({top_prompts[0][1]} uses)',
                severity='info',
                data={'top_prompts': top_prompts[:5]}
            ))
        
        # Quality insights
        if stats['avg_quality'] < 0.7:
            insights.append(Insight(
                type='quality',
                title='Low Quality Scores',
                description=f'Average quality {stats["avg_quality"]:.2f} below 0.7 threshold',
                severity='warning',
                data={'avg_quality': stats['avg_quality']}
            ))
        
        return insights
    
    def _get_top_prompts(self, days: int) -> List[tuple]:
        """Get most frequently used prompts"""
        cutoff = datetime.now() - timedelta(days=days)
        conn = self.tracker._get_connection()
        cursor = conn.execute("""
            SELECT prompt_name, COUNT(*) as count
            FROM prompt_usage
            WHERE timestamp > ?
            GROUP BY prompt_name
            ORDER BY count DESC
            LIMIT 10
        """, (cutoff.isoformat(),))
        return cursor.fetchall()
    
    def get_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        insights = self.generate_insights()
        recommendations = []
        
        for insight in insights:
            if insight.type == 'performance' and insight.severity == 'warning':
                recommendations.append(f"Optimize slow prompts or break into smaller steps")
            elif insight.type == 'quality' and insight.severity == 'critical':
                recommendations.append(f"Review and improve prompts with low success rates")
            elif insight.type == 'quality' and 'avg_quality' in insight.data:
                recommendations.append(f"Add more examples and structure to improve quality scores")
        
        return recommendations
