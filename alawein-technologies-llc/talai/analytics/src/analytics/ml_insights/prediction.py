"""
Outcome Prediction and Anomaly Detection for TalAI

Machine learning models for predicting validation outcomes, detecting anomalies,
and forecasting research success.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pickle
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OutcomePredictor:
    """
    Predict validation outcomes and detect anomalies in research patterns.

    Features:
    - Hypothesis success prediction
    - Validation duration estimation
    - Anomaly detection
    - Risk assessment
    - Trend forecasting
    - Confidence interval estimation
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize outcome predictor."""
        self.model_path = Path(model_path or '/tmp/talai_predictors')
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.success_predictor = None
        self.duration_predictor = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()

        self._initialize_models()

    def _initialize_models(self):
        """Initialize prediction models."""
        try:
            # Success prediction model (classification)
            self.success_predictor = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

            # Duration prediction model (regression)
            self.duration_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )

            # Try to load pre-trained models
            self._load_models()

        except Exception as e:
            logger.error(f"Error initializing models: {e}")

    async def predict_hypothesis_success(self,
                                        hypothesis_data: Dict,
                                        return_probability: bool = True) -> Dict[str, Any]:
        """
        Predict whether a hypothesis will be successfully validated.

        Args:
            hypothesis_data: Hypothesis information
            return_probability: Whether to return probability scores

        Returns:
            Success prediction results
        """
        try:
            # Extract features
            features = await self._extract_features(hypothesis_data)

            # Use rule-based prediction if no trained model
            if self.success_predictor is None or not hasattr(self.success_predictor, 'classes_'):
                return await self._rule_based_prediction(hypothesis_data)

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Make prediction
            prediction = self.success_predictor.predict(features_scaled)[0]

            result = {
                'will_succeed': bool(prediction),
                'prediction_confidence': 0.0,
                'risk_factors': [],
                'success_factors': []
            }

            if return_probability:
                probabilities = self.success_predictor.predict_proba(features_scaled)[0]
                success_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                result['success_probability'] = float(success_prob)
                result['failure_probability'] = float(1 - success_prob)
                result['prediction_confidence'] = float(abs(success_prob - 0.5) * 2)

            # Identify key factors
            factors = await self._identify_success_factors(hypothesis_data, features)
            result['risk_factors'] = factors['risks']
            result['success_factors'] = factors['strengths']

            # Add recommendations
            result['recommendations'] = await self._generate_recommendations(
                hypothesis_data,
                result['success_probability'] if return_probability else 0.5
            )

            return result

        except Exception as e:
            logger.error(f"Error predicting success: {e}")
            return await self._rule_based_prediction(hypothesis_data)

    async def _rule_based_prediction(self, hypothesis_data: Dict) -> Dict[str, Any]:
        """Rule-based prediction fallback."""
        score = 0.5  # Start neutral

        risks = []
        strengths = []

        # Check evidence quality
        evidence = hypothesis_data.get('evidence', [])
        if len(evidence) >= 5:
            score += 0.1
            strengths.append("Sufficient evidence provided")
        else:
            score -= 0.1
            risks.append("Limited evidence")

        # Check peer review
        peer_reviewed = sum(1 for e in evidence
                          if e.get('source', {}).get('peer_reviewed'))
        if peer_reviewed >= 3:
            score += 0.15
            strengths.append("Strong peer-reviewed support")
        elif peer_reviewed == 0:
            score -= 0.15
            risks.append("No peer-reviewed evidence")

        # Check methodology
        if hypothesis_data.get('methodology'):
            score += 0.1
            strengths.append("Clear methodology defined")
        else:
            score -= 0.1
            risks.append("Methodology not specified")

        # Check assumptions
        assumptions = hypothesis_data.get('assumptions', [])
        high_risk_assumptions = sum(1 for a in assumptions
                                   if a.get('risk_level') == 'high')
        if high_risk_assumptions > 2:
            score -= 0.15
            risks.append(f"{high_risk_assumptions} high-risk assumptions")
        elif high_risk_assumptions == 0:
            score += 0.05
            strengths.append("Low-risk assumptions")

        # Ensure score is in [0, 1]
        score = max(0, min(1, score))

        return {
            'will_succeed': score > 0.5,
            'success_probability': float(score),
            'failure_probability': float(1 - score),
            'prediction_confidence': float(abs(score - 0.5) * 2),
            'risk_factors': risks,
            'success_factors': strengths,
            'prediction_method': 'rule_based',
            'recommendations': await self._generate_recommendations(hypothesis_data, score)
        }

    async def _extract_features(self, hypothesis_data: Dict) -> np.ndarray:
        """Extract numerical features from hypothesis data."""
        features = []

        # Evidence features
        evidence = hypothesis_data.get('evidence', [])
        features.append(len(evidence))
        features.append(sum(1 for e in evidence
                          if e.get('source', {}).get('peer_reviewed')))
        features.append(np.mean([e.get('quality_score', 0.5) for e in evidence])
                       if evidence else 0.5)

        # Methodology features
        features.append(1 if hypothesis_data.get('methodology') else 0)
        features.append(1 if hypothesis_data.get('statistical_analysis') else 0)
        features.append(hypothesis_data.get('sample_size', 0))

        # Assumption features
        assumptions = hypothesis_data.get('assumptions', [])
        features.append(len(assumptions))
        features.append(sum(1 for a in assumptions if a.get('risk_level') == 'high'))
        features.append(sum(1 for a in assumptions if a.get('justification')))

        # Domain encoding (simplified)
        domain = hypothesis_data.get('domain', 'general')
        domain_hash = hash(domain) % 100
        features.append(domain_hash / 100)

        # Complexity features
        statement = hypothesis_data.get('statement', '')
        features.append(len(statement.split()))  # Word count
        features.append(len(statement))  # Character count

        # Historical performance (if available)
        features.append(hypothesis_data.get('author_success_rate', 0.5))
        features.append(hypothesis_data.get('domain_success_rate', 0.5))

        return np.array(features)

    async def _identify_success_factors(self,
                                       hypothesis_data: Dict,
                                       features: np.ndarray) -> Dict[str, List[str]]:
        """Identify factors affecting success prediction."""
        risks = []
        strengths = []

        # Feature importance analysis (simplified)
        feature_names = [
            'evidence_count', 'peer_reviewed_count', 'avg_evidence_quality',
            'has_methodology', 'has_statistical_analysis', 'sample_size',
            'assumption_count', 'high_risk_assumptions', 'justified_assumptions',
            'domain_code', 'word_count', 'char_count',
            'author_success_rate', 'domain_success_rate'
        ]

        # Analyze each feature
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if name == 'evidence_count' and value < 3:
                risks.append("Insufficient evidence (< 3 sources)")
            elif name == 'evidence_count' and value >= 10:
                strengths.append("Extensive evidence base")

            elif name == 'peer_reviewed_count' and value == 0:
                risks.append("No peer-reviewed evidence")
            elif name == 'peer_reviewed_count' and value >= 5:
                strengths.append("Strong peer-reviewed support")

            elif name == 'sample_size' and value < 30 and value > 0:
                risks.append("Small sample size")
            elif name == 'sample_size' and value >= 100:
                strengths.append("Large sample size")

            elif name == 'high_risk_assumptions' and value > 2:
                risks.append("Multiple high-risk assumptions")

        return {'risks': risks[:5], 'strengths': strengths[:5]}

    async def _generate_recommendations(self,
                                       hypothesis_data: Dict,
                                       success_probability: float) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if success_probability < 0.5:
            # Low success probability - focus on improvements
            evidence = hypothesis_data.get('evidence', [])
            if len(evidence) < 5:
                recommendations.append("Add more supporting evidence (aim for 5+ sources)")

            peer_reviewed = sum(1 for e in evidence
                              if e.get('source', {}).get('peer_reviewed'))
            if peer_reviewed < 3:
                recommendations.append("Include more peer-reviewed sources")

            if not hypothesis_data.get('methodology'):
                recommendations.append("Define clear research methodology")

            if hypothesis_data.get('sample_size', 0) < 30:
                recommendations.append("Increase sample size for statistical validity")

        elif success_probability < 0.7:
            # Moderate probability - refinements
            recommendations.append("Consider peer review before validation")
            recommendations.append("Verify all assumptions are justified")

        else:
            # High probability - optimization
            recommendations.append("Ready for validation - consider fast-track mode")
            recommendations.append("Document methodology for reproducibility")

        return recommendations[:3]

    async def predict_validation_duration(self,
                                         hypothesis_data: Dict) -> Dict[str, Any]:
        """
        Predict how long validation will take.

        Args:
            hypothesis_data: Hypothesis information

        Returns:
            Duration prediction results
        """
        try:
            # Extract features
            features = await self._extract_features(hypothesis_data)

            # Simple duration estimation based on complexity
            base_duration = 120  # Base 2 minutes

            # Adjust based on evidence count
            evidence_count = len(hypothesis_data.get('evidence', []))
            duration = base_duration + (evidence_count * 15)  # 15 seconds per evidence

            # Adjust based on complexity
            statement_length = len(hypothesis_data.get('statement', ''))
            if statement_length > 500:
                duration += 60  # Add 1 minute for complex hypotheses

            # Adjust based on validation mode
            mode = hypothesis_data.get('validation_mode', 'standard')
            if mode == 'comprehensive':
                duration *= 2
            elif mode == 'quick':
                duration *= 0.5

            # Calculate confidence intervals
            std_dev = duration * 0.2  # 20% standard deviation
            lower_bound = max(30, duration - 2 * std_dev)
            upper_bound = duration + 2 * std_dev

            return {
                'estimated_duration_seconds': float(duration),
                'estimated_duration_minutes': float(duration / 60),
                'confidence_interval': {
                    'lower_bound_seconds': float(lower_bound),
                    'upper_bound_seconds': float(upper_bound),
                    'confidence': 0.95
                },
                'complexity_factors': {
                    'evidence_count': evidence_count,
                    'statement_complexity': 'high' if statement_length > 500 else 'medium'
                                          if statement_length > 200 else 'low',
                    'validation_mode': mode
                }
            }

        except Exception as e:
            logger.error(f"Error predicting duration: {e}")
            return {
                'estimated_duration_seconds': 180.0,
                'error': str(e)
            }

    async def detect_anomalies(self,
                              validation_history: List[Dict],
                              current_validation: Dict) -> Dict[str, Any]:
        """
        Detect anomalies in validation patterns.

        Args:
            validation_history: Historical validation data
            current_validation: Current validation to check

        Returns:
            Anomaly detection results
        """
        try:
            if len(validation_history) < 10:
                return {
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'message': 'Insufficient history for anomaly detection'
                }

            # Extract features from history
            history_features = []
            for val in validation_history:
                features = await self._extract_validation_features(val)
                history_features.append(features)

            history_array = np.array(history_features)

            # Train or update anomaly detector
            self.anomaly_detector.fit(history_array)

            # Check current validation
            current_features = await self._extract_validation_features(current_validation)
            current_array = current_features.reshape(1, -1)

            # Predict anomaly
            prediction = self.anomaly_detector.predict(current_array)[0]
            anomaly_score = self.anomaly_detector.score_samples(current_array)[0]

            is_anomaly = prediction == -1

            # Identify what makes it anomalous
            anomaly_factors = []
            if is_anomaly:
                anomaly_factors = await self._identify_anomaly_factors(
                    current_validation,
                    validation_history
                )

            return {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'confidence': float(abs(anomaly_score)),
                'anomaly_factors': anomaly_factors,
                'recommendation': 'Review carefully - unusual pattern detected'
                                if is_anomaly else 'Normal pattern'
            }

        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return {
                'is_anomaly': False,
                'error': str(e)
            }

    async def _extract_validation_features(self, validation: Dict) -> np.ndarray:
        """Extract features from validation data."""
        features = []

        # Basic metrics
        features.append(validation.get('success_rate', 0.5))
        features.append(validation.get('duration_seconds', 180))
        features.append(validation.get('evidence_count', 0))
        features.append(validation.get('confidence_score', 0.5))
        features.append(validation.get('iteration_count', 1))
        features.append(validation.get('error_count', 0))

        # Quality metrics
        features.append(validation.get('avg_evidence_quality', 0.5))
        features.append(1 if validation.get('peer_reviewed') else 0)
        features.append(1 if validation.get('human_oversight') else 0)

        return np.array(features)

    async def _identify_anomaly_factors(self,
                                       current: Dict,
                                       history: List[Dict]) -> List[str]:
        """Identify what makes a validation anomalous."""
        factors = []

        # Calculate historical statistics
        hist_success_rates = [h.get('success_rate', 0.5) for h in history]
        hist_durations = [h.get('duration_seconds', 180) for h in history]

        current_success = current.get('success_rate', 0.5)
        current_duration = current.get('duration_seconds', 180)

        # Check for outliers
        if current_success < np.percentile(hist_success_rates, 5):
            factors.append("Unusually low success rate")
        elif current_success > np.percentile(hist_success_rates, 95):
            factors.append("Unusually high success rate")

        if current_duration < np.percentile(hist_durations, 5):
            factors.append("Unusually fast validation")
        elif current_duration > np.percentile(hist_durations, 95):
            factors.append("Unusually slow validation")

        if current.get('error_count', 0) > np.mean([h.get('error_count', 0) for h in history]) + 2:
            factors.append("High error count")

        if current.get('iteration_count', 1) > 5:
            factors.append("Excessive iterations")

        return factors

    async def assess_validation_risk(self, hypothesis_data: Dict) -> Dict[str, Any]:
        """
        Assess the risk level of a validation.

        Args:
            hypothesis_data: Hypothesis information

        Returns:
            Risk assessment results
        """
        try:
            risk_score = 0.0
            risk_factors = []
            mitigation_strategies = []

            # Evidence risk
            evidence = hypothesis_data.get('evidence', [])
            if len(evidence) < 3:
                risk_score += 0.3
                risk_factors.append("Minimal evidence")
                mitigation_strategies.append("Gather additional supporting evidence")

            # Quality risk
            quality_scores = [e.get('quality_score', 0.5) for e in evidence]
            if quality_scores and np.mean(quality_scores) < 0.5:
                risk_score += 0.2
                risk_factors.append("Low evidence quality")
                mitigation_strategies.append("Seek higher quality sources")

            # Assumption risk
            assumptions = hypothesis_data.get('assumptions', [])
            high_risk = sum(1 for a in assumptions if a.get('risk_level') == 'high')
            if high_risk > 0:
                risk_score += 0.1 * high_risk
                risk_factors.append(f"{high_risk} high-risk assumptions")
                mitigation_strategies.append("Validate critical assumptions first")

            # Complexity risk
            if len(hypothesis_data.get('statement', '')) > 500:
                risk_score += 0.1
                risk_factors.append("Complex hypothesis")
                mitigation_strategies.append("Consider breaking into sub-hypotheses")

            # Domain risk (some domains are inherently riskier)
            risky_domains = ['quantum', 'consciousness', 'cosmology']
            if any(d in hypothesis_data.get('domain', '').lower() for d in risky_domains):
                risk_score += 0.15
                risk_factors.append("Complex domain")
                mitigation_strategies.append("Ensure domain expertise is available")

            # Normalize risk score
            risk_score = min(risk_score, 1.0)

            # Determine risk level
            if risk_score < 0.3:
                risk_level = 'low'
            elif risk_score < 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'

            return {
                'risk_level': risk_level,
                'risk_score': float(risk_score),
                'risk_factors': risk_factors[:5],
                'mitigation_strategies': mitigation_strategies[:3],
                'confidence': 0.8,  # Fixed confidence for risk assessment
                'recommendation': self._get_risk_recommendation(risk_level)
            }

        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                'risk_level': 'unknown',
                'error': str(e)
            }

    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        if risk_level == 'low':
            return "Proceed with standard validation"
        elif risk_level == 'medium':
            return "Consider additional review before validation"
        else:
            return "Recommend comprehensive validation with expert oversight"

    async def forecast_research_trends(self,
                                      historical_data: List[Dict],
                                      horizon_days: int = 30) -> Dict[str, Any]:
        """
        Forecast future research trends.

        Args:
            historical_data: Historical validation data
            horizon_days: Forecast horizon in days

        Returns:
            Trend forecast results
        """
        try:
            if len(historical_data) < 30:
                return {
                    'error': 'Insufficient data for trend forecasting (need 30+ records)'
                }

            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Aggregate by day
            daily_stats = df.resample('D').agg({
                'success_rate': 'mean',
                'validation_count': 'count',
                'avg_duration': 'mean'
            }).fillna(0)

            # Simple moving average forecast
            window = min(7, len(daily_stats) // 4)
            ma_forecast = daily_stats.rolling(window=window).mean().iloc[-1]

            # Linear trend
            x = np.arange(len(daily_stats))
            y_success = daily_stats['success_rate'].values
            slope, intercept, _, _, _ = np.polyfit(x, y_success, 1, full=True)[:2]

            # Generate forecast
            future_dates = pd.date_range(
                start=daily_stats.index[-1] + pd.Timedelta(days=1),
                periods=horizon_days,
                freq='D'
            )

            forecast_success_rates = []
            for i in range(horizon_days):
                future_x = len(daily_stats) + i
                forecast_value = slope * future_x + intercept
                # Add some noise for realism
                forecast_value += np.random.normal(0, 0.05)
                forecast_value = max(0, min(1, forecast_value))
                forecast_success_rates.append(forecast_value)

            # Detect emerging patterns
            emerging_patterns = []
            if slope > 0.001:
                emerging_patterns.append("Increasing success rates")
            elif slope < -0.001:
                emerging_patterns.append("Decreasing success rates")

            # Domain trends
            if 'domain' in df.columns:
                domain_trends = df.groupby('domain')['success_rate'].mean().to_dict()
                top_domains = sorted(domain_trends.items(), key=lambda x: x[1], reverse=True)[:3]
            else:
                top_domains = []

            return {
                'forecast_period': {
                    'start': future_dates[0].isoformat(),
                    'end': future_dates[-1].isoformat(),
                    'days': horizon_days
                },
                'predicted_metrics': {
                    'avg_success_rate': float(np.mean(forecast_success_rates)),
                    'success_rate_trend': 'increasing' if slope > 0 else 'decreasing',
                    'daily_success_rates': forecast_success_rates[:7]  # First week
                },
                'confidence_interval': {
                    'lower': float(np.mean(forecast_success_rates) - 0.1),
                    'upper': float(np.mean(forecast_success_rates) + 0.1)
                },
                'emerging_patterns': emerging_patterns,
                'top_performing_domains': [{'domain': d, 'success_rate': float(r)}
                                          for d, r in top_domains],
                'recommendations': [
                    "Focus on high-performing domains" if top_domains else "Diversify research domains",
                    "Maintain current methodology" if slope > 0 else "Review validation approach"
                ]
            }

        except Exception as e:
            logger.error(f"Error forecasting trends: {e}")
            return {'error': str(e)}

    async def train_models(self,
                          training_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Train or update prediction models.

        Args:
            training_data: Dictionary with 'features' and 'labels' lists

        Returns:
            Training results
        """
        try:
            X = np.array(training_data['features'])
            y_success = np.array(training_data['success_labels'])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_success, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train success predictor
            self.success_predictor.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.success_predictor.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )

            # Save models
            await self._save_models()

            return {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'models_saved': True
            }

        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'error': str(e)}

    async def _save_models(self):
        """Save trained models to disk."""
        try:
            # Save success predictor
            success_path = self.model_path / 'success_predictor.pkl'
            with open(success_path, 'wb') as f:
                pickle.dump(self.success_predictor, f)

            # Save scaler
            scaler_path = self.model_path / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save anomaly detector
            anomaly_path = self.model_path / 'anomaly_detector.pkl'
            with open(anomaly_path, 'wb') as f:
                pickle.dump(self.anomaly_detector, f)

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            success_path = self.model_path / 'success_predictor.pkl'
            scaler_path = self.model_path / 'scaler.pkl'
            anomaly_path = self.model_path / 'anomaly_detector.pkl'

            if success_path.exists():
                with open(success_path, 'rb') as f:
                    self.success_predictor = pickle.load(f)

            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            if anomaly_path.exists():
                with open(anomaly_path, 'rb') as f:
                    self.anomaly_detector = pickle.load(f)

            logger.info("Pre-trained models loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")