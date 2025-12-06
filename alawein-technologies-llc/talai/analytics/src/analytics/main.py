"""
Main Analytics Interface for TalAI

Unified interface for all analytics, ML insights, recommendations, and visualizations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .engine.analytics_engine import AnalyticsEngine
from .engine.time_series import TimeSeriesAnalyzer
from .engine.scoring import QualityScorer
from .ml_insights.embeddings import EmbeddingProcessor
from .ml_insights.classification import DomainClassifier
from .ml_insights.prediction import OutcomePredictor
from .recommendations.recommendation_engine import RecommendationEngine
from .visualization.dashboard import DashboardGenerator

logger = logging.getLogger(__name__)

class TalAIAnalytics:
    """
    Main analytics interface for TalAI system.

    Provides unified access to all analytics capabilities:
    - Performance analytics
    - ML-powered insights
    - Intelligent recommendations
    - Interactive visualizations
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TalAI Analytics.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize all components
        self.analytics_engine = AnalyticsEngine(
            data_path=self.config.get('data_path')
        )
        self.time_series = TimeSeriesAnalyzer()
        self.quality_scorer = QualityScorer()
        self.embedding_processor = EmbeddingProcessor(
            model_name=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.domain_classifier = DomainClassifier()
        self.outcome_predictor = OutcomePredictor()
        self.recommendation_engine = RecommendationEngine()
        self.dashboard_generator = DashboardGenerator()

        logger.info("TalAI Analytics initialized")

    async def analyze_hypothesis(self, hypothesis_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive hypothesis analysis.

        Args:
            hypothesis_data: Hypothesis information

        Returns:
            Complete analysis results
        """
        try:
            results = {
                'hypothesis_id': hypothesis_data.get('id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'analyses': {}
            }

            # 1. Quality scoring
            quality_score = await self.quality_scorer.score_hypothesis(hypothesis_data)
            results['analyses']['quality'] = {
                'overall_score': quality_score.overall_score,
                'components': quality_score.components,
                'confidence': quality_score.confidence,
                'feedback': quality_score.feedback
            }

            # 2. Domain classification
            domain_result = await self.domain_classifier.classify_domain(
                hypothesis_data.get('statement', '')
            )
            results['analyses']['domain'] = domain_result

            # 3. Outcome prediction
            prediction = await self.outcome_predictor.predict_hypothesis_success(
                hypothesis_data
            )
            results['analyses']['prediction'] = prediction

            # 4. Risk assessment
            risk = await self.outcome_predictor.assess_validation_risk(hypothesis_data)
            results['analyses']['risk'] = risk

            # 5. Validation mode recommendation
            mode_rec = await self.recommendation_engine.recommend_validation_mode(
                hypothesis_data
            )
            results['analyses']['recommended_mode'] = mode_rec

            # 6. Evidence source suggestions
            evidence_suggestions = await self.recommendation_engine.suggest_evidence_sources(
                hypothesis_data
            )
            results['analyses']['evidence_suggestions'] = evidence_suggestions

            # 7. Duration estimation
            duration = await self.outcome_predictor.predict_validation_duration(
                hypothesis_data
            )
            results['analyses']['duration_estimate'] = duration

            return results

        except Exception as e:
            logger.error(f"Error analyzing hypothesis: {e}")
            return {'error': str(e)}

    async def track_validation_event(self, validation_data: Dict) -> Dict[str, Any]:
        """
        Track and analyze a validation event.

        Args:
            validation_data: Validation event data

        Returns:
            Tracking results and insights
        """
        try:
            # Track in analytics engine
            metrics = await self.analytics_engine.track_validation(validation_data)

            # Check for anomalies
            history = list(self.analytics_engine.metrics_cache.values())
            anomaly_result = await self.outcome_predictor.detect_anomalies(
                [m.to_dict() for m in history[-100:]],
                validation_data
            )

            # Update user preferences
            if 'user_id' in validation_data:
                await self.recommendation_engine.update_user_preferences(
                    validation_data['user_id'],
                    validation_data
                )

            return {
                'metrics': metrics.to_dict(),
                'anomaly_detection': anomaly_result,
                'insights_generated': len(self.analytics_engine.insights_cache)
            }

        except Exception as e:
            logger.error(f"Error tracking validation: {e}")
            return {'error': str(e)}

    async def find_similar_research(self,
                                   hypothesis: str,
                                   database: List[Dict],
                                   top_k: int = 5) -> List[Dict]:
        """
        Find similar research/hypotheses.

        Args:
            hypothesis: Query hypothesis
            database: Database of hypotheses
            top_k: Number of results

        Returns:
            Similar research results
        """
        try:
            # Use embedding similarity
            similar = await self.embedding_processor.find_similar_hypotheses(
                hypothesis, database, top_k
            )

            # Get recommendations for each
            for item in similar:
                item['recommendations'] = await self.recommendation_engine.recommend_similar_validations(
                    {'statement': hypothesis},
                    [item],
                    top_k=1
                )

            return similar

        except Exception as e:
            logger.error(f"Error finding similar research: {e}")
            return []

    async def generate_insights_report(self,
                                      time_range_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive insights report.

        Args:
            time_range_days: Time range for analysis

        Returns:
            Insights report
        """
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'time_range_days': time_range_days,
                'sections': {}
            }

            # Get validation data
            all_metrics = [m.to_dict() for m in self.analytics_engine.metrics_cache.values()]

            if all_metrics:
                # 1. Success trends
                trends = await self.analytics_engine.compute_success_trends(
                    time_window_days=time_range_days
                )
                report['sections']['success_trends'] = trends

                # 2. Time series analysis
                ts_analysis = await self.time_series.analyze_validation_trends(
                    all_metrics, 'success_rate'
                )
                report['sections']['time_series'] = ts_analysis

                # 3. Domain insights
                domains = list(set(m['domain'] for m in all_metrics if 'domain' in m))
                domain_insights = {}
                for domain in domains[:5]:  # Limit to top 5 domains
                    insight = await self.analytics_engine.get_domain_insights(domain)
                    domain_insights[domain] = insight
                report['sections']['domains'] = domain_insights

                # 4. Forecast
                forecast = await self.outcome_predictor.forecast_research_trends(
                    all_metrics, horizon_days=7
                )
                report['sections']['forecast'] = forecast

                # 5. Recent insights
                report['sections']['recent_insights'] = [
                    i.to_dict() for i in self.analytics_engine.insights_cache[-10:]
                ]

            # 6. System performance
            report['sections']['performance'] = {
                'total_validations': len(all_metrics),
                'unique_domains': len(set(m.get('domain') for m in all_metrics if 'domain' in m)),
                'avg_processing_time': sum(m.get('validation_duration_seconds', 0)
                                          for m in all_metrics) / len(all_metrics)
                                      if all_metrics else 0
            }

            return report

        except Exception as e:
            logger.error(f"Error generating insights report: {e}")
            return {'error': str(e)}

    async def create_interactive_dashboard(self,
                                          validation_data: Optional[List[Dict]] = None
                                          ) -> Dict[str, Any]:
        """
        Create interactive analytics dashboard.

        Args:
            validation_data: Optional validation data (uses cached if not provided)

        Returns:
            Dashboard with visualizations
        """
        try:
            # Use cached data if not provided
            if validation_data is None:
                validation_data = [m.to_dict() for m in self.analytics_engine.metrics_cache.values()]

            if not validation_data:
                return {'error': 'No validation data available'}

            # Generate dashboard
            dashboard = await self.dashboard_generator.generate_validation_dashboard(
                validation_data
            )

            # Add network visualization if enough data
            if len(validation_data) > 10:
                # Create hypothesis network
                nodes = [{'label': f"H{i}", 'value': v.get('success_rate', 0.5)}
                        for i, v in enumerate(validation_data[:20])]
                edges = []  # Would be populated based on similarity

                network = await self.dashboard_generator.create_network_graph(
                    nodes, edges, 'Hypothesis Similarity Network'
                )
                dashboard['visualizations']['network'] = network

            # Add research landscape
            landscape = await self.dashboard_generator.create_research_landscape_map(
                validation_data[:50]  # Limit for performance
            )
            dashboard['visualizations']['landscape'] = landscape

            return dashboard

        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return {'error': str(e)}

    async def run_ab_test_analysis(self,
                                  test_name: str,
                                  variant_a: List[Dict],
                                  variant_b: List[Dict]) -> Dict[str, Any]:
        """
        Run A/B test analysis.

        Args:
            test_name: Name of the test
            variant_a: Variant A data
            variant_b: Variant B data

        Returns:
            A/B test results
        """
        try:
            results = await self.analytics_engine.run_ab_test(
                test_name, variant_a, variant_b
            )

            # Add visualization
            if results and 'error' not in results:
                # Create comparison visualization
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Variant A',
                    x=['Success Rate'],
                    y=[results['variant_a']['success_rate']]
                ))
                fig.add_trace(go.Bar(
                    name='Variant B',
                    x=['Success Rate'],
                    y=[results['variant_b']['success_rate']]
                ))

                fig.update_layout(
                    title=f'A/B Test: {test_name}',
                    yaxis_title='Success Rate',
                    barmode='group'
                )

                results['visualization'] = {
                    'figure': fig.to_dict(),
                    'html': fig.to_html(include_plotlyjs='cdn')
                }

            return results

        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            return {'error': str(e)}

    async def cluster_research_topics(self,
                                     hypotheses: List[Dict],
                                     method: str = 'kmeans') -> Dict[str, Any]:
        """
        Cluster research topics/hypotheses.

        Args:
            hypotheses: List of hypotheses
            method: Clustering method

        Returns:
            Clustering results with visualization
        """
        try:
            # Perform clustering
            clusters = await self.embedding_processor.cluster_hypotheses(
                hypotheses, method
            )

            # Generate visualization if successful
            if clusters and 'error' not in clusters:
                # Create cluster visualization
                viz_data = await self.embedding_processor.generate_embedding_visualization(
                    hypotheses, method='tsne'
                )

                if 'coordinates' in viz_data:
                    # Add cluster assignments to coordinates
                    for coord in viz_data['coordinates']:
                        h_id = coord['hypothesis_id']
                        # Find cluster for this hypothesis
                        for cluster_id, members in clusters['clusters'].items():
                            if any(m['hypothesis_id'] == h_id for m in members):
                                coord['cluster'] = cluster_id
                                break

                clusters['visualization'] = viz_data

            return clusters

        except Exception as e:
            logger.error(f"Error clustering research topics: {e}")
            return {'error': str(e)}

    async def export_analytics(self,
                              format: str = 'json',
                              include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Export all analytics data.

        Args:
            format: Export format ('json', 'html')
            include_visualizations: Whether to include visualizations

        Returns:
            Exported analytics
        """
        try:
            # Gather all analytics
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'analytics_report': await self.analytics_engine.export_analytics_report(),
                'insights': await self.generate_insights_report()
            }

            if include_visualizations:
                dashboard = await self.create_interactive_dashboard()
                export_data['dashboard'] = dashboard

            # Export in requested format
            if format == 'html' and include_visualizations:
                return await self.dashboard_generator.export_dashboard_report(
                    export_data['dashboard'], 'html'
                )
            else:
                return {
                    'format': format,
                    'content': json.dumps(export_data, indent=2, default=str),
                    'size_bytes': len(json.dumps(export_data, default=str).encode())
                }

        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            return {'error': str(e)}

    async def get_personalized_recommendations(self,
                                              user_id: str,
                                              hypothesis_data: Dict) -> Dict[str, Any]:
        """
        Get personalized recommendations for a user.

        Args:
            user_id: User identifier
            hypothesis_data: Current hypothesis

        Returns:
            Personalized recommendations
        """
        try:
            recommendations = {
                'user_id': user_id,
                'generated_at': datetime.now().isoformat(),
                'recommendations': {}
            }

            # 1. Validation mode recommendation
            mode_rec = await self.recommendation_engine.recommend_validation_mode(
                hypothesis_data,
                self.analytics_engine.user_sessions.get(user_id, [])
            )
            recommendations['recommendations']['validation_mode'] = mode_rec

            # 2. Similar validations
            history = [m.to_dict() for m in self.analytics_engine.metrics_cache.values()]
            similar = await self.recommendation_engine.recommend_similar_validations(
                hypothesis_data, history[:100], top_k=3
            )
            recommendations['recommendations']['similar_validations'] = similar

            # 3. Expert reviewers
            experts = await self.recommendation_engine.recommend_expert_reviewers(
                hypothesis_data
            )
            recommendations['recommendations']['expert_reviewers'] = experts

            # 4. Citations
            citations = await self.recommendation_engine.recommend_citations(
                hypothesis_data
            )
            recommendations['recommendations']['relevant_citations'] = citations

            # 5. Collaborative recommendations
            collaborative = await self.recommendation_engine.get_collaborative_recommendations(
                user_id, hypothesis_data
            )
            recommendations['recommendations']['collaborative'] = collaborative

            return recommendations

        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return {'error': str(e)}


async def main():
    """Example usage of TalAI Analytics."""
    # Initialize analytics
    analytics = TalAIAnalytics()

    # Example hypothesis
    hypothesis = {
        'id': 'hyp_001',
        'statement': 'Quantum entanglement can be used for instantaneous communication',
        'domain': 'physics',
        'evidence': [
            {'source': {'peer_reviewed': True}, 'quality_score': 0.8},
            {'source': {'peer_reviewed': False}, 'quality_score': 0.6}
        ],
        'assumptions': [
            {'risk_level': 'high', 'justification': 'Assumes no decoherence'}
        ],
        'methodology': {
            'research_design': 'experimental',
            'sample_size': 50
        }
    }

    # Analyze hypothesis
    print("Analyzing hypothesis...")
    analysis = await analytics.analyze_hypothesis(hypothesis)
    print(f"Quality Score: {analysis['analyses']['quality']['overall_score']:.2f}")
    print(f"Success Prediction: {analysis['analyses']['prediction']['success_probability']:.2%}")
    print(f"Risk Level: {analysis['analyses']['risk']['risk_level']}")

    # Track validation
    validation = {
        'hypothesis_id': hypothesis['id'],
        'start_time': datetime.now().isoformat(),
        'end_time': (datetime.now() + timedelta(minutes=5)).isoformat(),
        'evidence': hypothesis['evidence'],
        'success_rate': 0.7,
        'confidence': 0.85,
        'domain': hypothesis['domain'],
        'mode': 'standard'
    }

    tracking = await analytics.track_validation_event(validation)
    print(f"\nValidation tracked: {tracking['metrics']['hypothesis_id']}")

    # Generate insights report
    print("\nGenerating insights report...")
    report = await analytics.generate_insights_report(time_range_days=7)
    print(f"Report sections: {list(report['sections'].keys())}")

    # Create dashboard
    print("\nCreating interactive dashboard...")
    dashboard = await analytics.create_interactive_dashboard()
    if 'key_metrics' in dashboard:
        print(f"Total validations: {dashboard['key_metrics']['total_validations']}")

    print("\nAnalytics demo completed!")


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    asyncio.run(main())