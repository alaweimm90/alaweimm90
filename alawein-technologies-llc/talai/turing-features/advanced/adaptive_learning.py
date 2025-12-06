#!/usr/bin/env python3
"""
Adaptive Learning Module for TalAI Turing Challenge

Implements continuous learning from validation history with
pattern recognition, meta-learning, and performance improvement.

Features:
- Learn from past validations
- Improve hypothesis classification over time
- Pattern recognition in successful/failed hypotheses
- Meta-learning from user feedback
- Continuous model improvement
"""

import asyncio
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict, deque
import hashlib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Learning strategies for adaptation"""
    ONLINE_LEARNING = "online_learning"
    BATCH_LEARNING = "batch_learning"
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class FeedbackType(Enum):
    """Types of user feedback"""
    EXPLICIT = "explicit"  # Direct rating/score
    IMPLICIT = "implicit"  # Based on user actions
    CORRECTIVE = "corrective"  # Corrections to predictions
    PREFERENTIAL = "preferential"  # Preference between options


@dataclass
class ValidationRecord:
    """Record of a hypothesis validation"""
    record_id: str
    hypothesis: str
    timestamp: datetime
    predicted_outcome: str
    actual_outcome: Optional[str] = None
    confidence_score: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    model_version: str = "1.0.0"


@dataclass
class LearningPattern:
    """Identified pattern in hypothesis validation"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    success_rate: float
    feature_importance: Dict[str, float]
    examples: List[str]
    discovered_at: datetime


@dataclass
class ModelPerformance:
    """Performance metrics for the learning model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaLearningInsight:
    """Insights from meta-learning process"""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    recommendations: List[str]
    timestamp: datetime


class AdaptiveLearner:
    """Core adaptive learning engine"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        memory_size: int = 10000,
        model_update_frequency: int = 100
    ):
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.update_frequency = model_update_frequency

        # Learning components
        self.validation_history: deque = deque(maxlen=memory_size)
        self.pattern_database: Dict[str, LearningPattern] = {}
        self.performance_history: List[ModelPerformance] = []

        # ML models
        self.hypothesis_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.confidence_predictor = GradientBoostingClassifier(
            n_estimators=50,
            random_state=42
        )
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3)
        )
        self.feature_scaler = StandardScaler()

        # Meta-learning components
        self.meta_insights: List[MetaLearningInsight] = []
        self.learning_curves: Dict[str, List[float]] = defaultdict(list)

        # State
        self.is_trained = False
        self.updates_since_training = 0
        self.model_version = "1.0.0"

    async def learn_from_validation(
        self,
        validation_record: ValidationRecord
    ) -> Dict[str, Any]:
        """Learn from a single validation result"""
        # Add to history
        self.validation_history.append(validation_record)
        self.updates_since_training += 1

        # Extract patterns
        patterns = await self._extract_patterns(validation_record)
        for pattern in patterns:
            self.pattern_database[pattern.pattern_id] = pattern

        # Update online if needed
        if self.updates_since_training >= self.update_frequency:
            await self.update_models()

        # Generate learning report
        return {
            "record_id": validation_record.record_id,
            "patterns_identified": len(patterns),
            "model_updated": self.updates_since_training >= self.update_frequency,
            "current_accuracy": self._get_current_accuracy(),
            "insights": await self._generate_insights(validation_record)
        }

    async def predict_hypothesis_outcome(
        self,
        hypothesis: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predict the outcome of a hypothesis based on learned patterns"""
        if not self.is_trained:
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "reason": "Model not yet trained"
            }

        # Extract features
        features = await self._extract_features(hypothesis, context)

        # Make prediction
        try:
            # Transform text features
            text_features = self.text_vectorizer.transform([hypothesis]).toarray()[0]

            # Combine all features
            feature_vector = np.concatenate([
                text_features,
                [features.get(k, 0) for k in sorted(features.keys())]
            ])

            # Scale features
            feature_vector = self.feature_scaler.transform([feature_vector])

            # Predict
            prediction = self.hypothesis_classifier.predict(feature_vector)[0]
            confidence_proba = self.hypothesis_classifier.predict_proba(feature_vector)[0]
            confidence = float(np.max(confidence_proba))

            # Identify similar patterns
            similar_patterns = await self._find_similar_patterns(hypothesis)

            return {
                "prediction": prediction,
                "confidence": confidence,
                "similar_patterns": similar_patterns,
                "feature_importance": self._get_feature_importance(),
                "model_version": self.model_version
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
            }

    async def update_models(self) -> Dict[str, Any]:
        """Update ML models based on accumulated data"""
        if len(self.validation_history) < 10:
            return {"status": "insufficient_data", "records": len(self.validation_history)}

        # Prepare training data
        X, y = await self._prepare_training_data()

        if X.shape[0] < 10:
            return {"status": "insufficient_processed_data"}

        # Split data for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train models
        self.hypothesis_classifier.fit(X_train, y_train)
        self.confidence_predictor.fit(X_train, y_train)

        # Evaluate performance
        y_pred = self.hypothesis_classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='weighted', zero_division=0
        )

        # Record performance
        performance = ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            feature_importance=self._get_feature_importance()
        )
        self.performance_history.append(performance)

        # Update state
        self.is_trained = True
        self.updates_since_training = 0
        self.model_version = f"1.{len(self.performance_history)}.0"

        # Generate meta-learning insights
        insights = await self._perform_meta_learning()

        return {
            "status": "success",
            "model_version": self.model_version,
            "performance": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "meta_insights": len(insights)
        }

    async def process_user_feedback(
        self,
        record_id: str,
        feedback: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.EXPLICIT
    ) -> Dict[str, Any]:
        """Process and learn from user feedback"""
        # Find the validation record
        record = None
        for r in self.validation_history:
            if r.record_id == record_id:
                record = r
                break

        if not record:
            return {"status": "record_not_found"}

        # Update record with feedback
        record.feedback = {
            "type": feedback_type.value,
            "content": feedback,
            "timestamp": datetime.now().isoformat()
        }

        # Adjust learning based on feedback type
        learning_adjustment = await self._adjust_learning_from_feedback(
            record, feedback, feedback_type
        )

        # Update patterns
        await self._update_patterns_from_feedback(record, feedback)

        return {
            "status": "feedback_processed",
            "record_id": record_id,
            "feedback_type": feedback_type.value,
            "learning_adjustment": learning_adjustment,
            "model_update_scheduled": self.updates_since_training >= self.update_frequency - 10
        }

    async def identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify areas where the model can improve"""
        opportunities = []

        # Analyze failure patterns
        failure_patterns = await self._analyze_failure_patterns()
        for pattern in failure_patterns:
            opportunities.append({
                "type": "failure_pattern",
                "description": pattern["description"],
                "frequency": pattern["frequency"],
                "suggested_action": pattern["suggestion"]
            })

        # Analyze performance trends
        if len(self.performance_history) > 5:
            trend = await self._analyze_performance_trend()
            if trend["direction"] == "declining":
                opportunities.append({
                    "type": "performance_decline",
                    "description": "Model performance declining",
                    "metrics": trend["metrics"],
                    "suggested_action": "Increase training data diversity"
                })

        # Check for concept drift
        drift_detected = await self._detect_concept_drift()
        if drift_detected:
            opportunities.append({
                "type": "concept_drift",
                "description": "Data distribution has changed",
                "confidence": drift_detected["confidence"],
                "suggested_action": "Retrain model with recent data"
            })

        return opportunities

    async def _extract_patterns(
        self,
        validation_record: ValidationRecord
    ) -> List[LearningPattern]:
        """Extract patterns from validation record"""
        patterns = []

        # Text-based patterns
        hypothesis_words = validation_record.hypothesis.lower().split()

        # Length pattern
        if len(hypothesis_words) > 50:
            pattern = LearningPattern(
                pattern_id=hashlib.md5(f"length_{len(hypothesis_words)}".encode()).hexdigest()[:8],
                pattern_type="length",
                description=f"Long hypothesis ({len(hypothesis_words)} words)",
                frequency=1,
                success_rate=1.0 if validation_record.actual_outcome == "success" else 0.0,
                feature_importance={"hypothesis_length": 0.8},
                examples=[validation_record.hypothesis[:100]],
                discovered_at=datetime.now()
            )
            patterns.append(pattern)

        # Keyword patterns
        keywords = ["quantum", "neural", "optimization", "hypothesis", "theory"]
        present_keywords = [kw for kw in keywords if kw in validation_record.hypothesis.lower()]
        if present_keywords:
            pattern = LearningPattern(
                pattern_id=hashlib.md5(f"keywords_{'_'.join(present_keywords)}".encode()).hexdigest()[:8],
                pattern_type="keywords",
                description=f"Contains keywords: {', '.join(present_keywords)}",
                frequency=1,
                success_rate=validation_record.confidence_score,
                feature_importance={f"keyword_{kw}": 0.5 for kw in present_keywords},
                examples=[validation_record.hypothesis[:100]],
                discovered_at=datetime.now()
            )
            patterns.append(pattern)

        return patterns

    async def _extract_features(
        self,
        hypothesis: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Extract features from hypothesis"""
        features = {}

        # Text features
        features["length"] = len(hypothesis.split())
        features["avg_word_length"] = np.mean([len(w) for w in hypothesis.split()])
        features["question_marks"] = hypothesis.count("?")
        features["exclamation_marks"] = hypothesis.count("!")

        # Complexity features
        features["unique_words"] = len(set(hypothesis.lower().split()))
        features["lexical_diversity"] = features["unique_words"] / features["length"] if features["length"] > 0 else 0

        # Context features
        if context:
            features["has_context"] = 1.0
            features["context_size"] = len(str(context))
        else:
            features["has_context"] = 0.0
            features["context_size"] = 0.0

        return features

    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from validation history"""
        X = []
        y = []

        hypotheses = []
        for record in self.validation_history:
            if record.actual_outcome:
                hypotheses.append(record.hypothesis)

        if not hypotheses:
            return np.array([]), np.array([])

        # Fit vectorizer if needed
        if not hasattr(self.text_vectorizer, 'vocabulary_'):
            self.text_vectorizer.fit(hypotheses)

        # Extract features for each record
        for record in self.validation_history:
            if record.actual_outcome:
                # Text features
                text_features = self.text_vectorizer.transform([record.hypothesis]).toarray()[0]

                # Additional features
                add_features = await self._extract_features(record.hypothesis)
                add_features_array = np.array([add_features.get(k, 0) for k in sorted(add_features.keys())])

                # Combine features
                feature_vector = np.concatenate([text_features, add_features_array])
                X.append(feature_vector)

                # Label (simplified to binary for this example)
                y.append(1 if record.actual_outcome == "success" else 0)

        X = np.array(X)
        y = np.array(y)

        # Fit scaler if needed
        if X.shape[0] > 0 and not hasattr(self.feature_scaler, 'mean_'):
            self.feature_scaler.fit(X)
            X = self.feature_scaler.transform(X)

        return X, y

    async def _find_similar_patterns(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Find patterns similar to the given hypothesis"""
        similar = []

        for pattern_id, pattern in self.pattern_database.items():
            # Simple similarity check based on keywords
            pattern_keywords = set(pattern.description.lower().split())
            hypothesis_keywords = set(hypothesis.lower().split())

            overlap = len(pattern_keywords & hypothesis_keywords)
            if overlap > 0:
                similarity = overlap / len(pattern_keywords | hypothesis_keywords)
                similar.append({
                    "pattern_id": pattern_id,
                    "description": pattern.description,
                    "similarity": similarity,
                    "success_rate": pattern.success_rate
                })

        # Sort by similarity
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:5]  # Top 5 similar patterns

    async def _perform_meta_learning(self) -> List[MetaLearningInsight]:
        """Perform meta-learning analysis"""
        insights = []

        # Analyze learning curves
        if len(self.performance_history) > 3:
            recent_performance = [p.accuracy for p in self.performance_history[-5:]]
            if all(recent_performance[i] <= recent_performance[i-1] for i in range(1, len(recent_performance))):
                insight = MetaLearningInsight(
                    insight_id=hashlib.md5(f"plateau_{datetime.now()}".encode()).hexdigest()[:8],
                    insight_type="performance_plateau",
                    description="Model performance has plateaued",
                    confidence=0.8,
                    supporting_evidence=[f"Accuracy: {recent_performance}"],
                    recommendations=["Consider adding more diverse training data", "Try ensemble methods"],
                    timestamp=datetime.now()
                )
                insights.append(insight)
                self.meta_insights.append(insight)

        # Analyze pattern effectiveness
        effective_patterns = [p for p in self.pattern_database.values() if p.success_rate > 0.8]
        if len(effective_patterns) > 5:
            insight = MetaLearningInsight(
                insight_id=hashlib.md5(f"effective_patterns_{datetime.now()}".encode()).hexdigest()[:8],
                insight_type="effective_patterns",
                description=f"Identified {len(effective_patterns)} highly effective patterns",
                confidence=0.9,
                supporting_evidence=[p.description for p in effective_patterns[:3]],
                recommendations=["Focus on hypotheses matching these patterns"],
                timestamp=datetime.now()
            )
            insights.append(insight)
            self.meta_insights.append(insight)

        return insights

    async def _adjust_learning_from_feedback(
        self,
        record: ValidationRecord,
        feedback: Dict[str, Any],
        feedback_type: FeedbackType
    ) -> Dict[str, Any]:
        """Adjust learning based on feedback"""
        adjustment = {"type": feedback_type.value}

        if feedback_type == FeedbackType.CORRECTIVE:
            # User corrected the prediction
            record.actual_outcome = feedback.get("correct_outcome")
            adjustment["action"] = "corrected_outcome"
            adjustment["impact"] = "high"

        elif feedback_type == FeedbackType.EXPLICIT:
            # User provided rating
            rating = feedback.get("rating", 0)
            if rating < 3:  # Poor rating
                adjustment["action"] = "downweight_example"
                adjustment["weight_adjustment"] = 0.5
            else:
                adjustment["action"] = "upweight_example"
                adjustment["weight_adjustment"] = 1.5

        return adjustment

    async def _update_patterns_from_feedback(
        self,
        record: ValidationRecord,
        feedback: Dict[str, Any]
    ) -> None:
        """Update patterns based on feedback"""
        # Find patterns related to this record
        hypothesis_patterns = await self._find_similar_patterns(record.hypothesis)

        for pattern_info in hypothesis_patterns:
            if pattern_info["pattern_id"] in self.pattern_database:
                pattern = self.pattern_database[pattern_info["pattern_id"]]

                # Update success rate based on feedback
                if "rating" in feedback:
                    rating = feedback["rating"]
                    # Weighted update of success rate
                    pattern.success_rate = 0.9 * pattern.success_rate + 0.1 * (rating / 5.0)

                pattern.frequency += 1

    async def _analyze_failure_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in failed validations"""
        failures = [r for r in self.validation_history if r.actual_outcome == "failure"]

        if not failures:
            return []

        # Group failures by common characteristics
        failure_patterns = []

        # Low confidence failures
        low_conf_failures = [f for f in failures if f.confidence_score < 0.5]
        if len(low_conf_failures) > 5:
            failure_patterns.append({
                "description": "Low confidence predictions failing",
                "frequency": len(low_conf_failures),
                "suggestion": "Improve confidence calibration"
            })

        return failure_patterns

    async def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trend over time"""
        if len(self.performance_history) < 2:
            return {"direction": "unknown", "metrics": {}}

        recent = self.performance_history[-5:]
        accuracies = [p.accuracy for p in recent]

        # Simple trend detection
        if accuracies[-1] < accuracies[0]:
            direction = "declining"
        elif accuracies[-1] > accuracies[0]:
            direction = "improving"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "metrics": {
                "recent_accuracy": accuracies[-1],
                "average_accuracy": np.mean(accuracies),
                "trend_slope": (accuracies[-1] - accuracies[0]) / len(accuracies) if len(accuracies) > 1 else 0
            }
        }

    async def _detect_concept_drift(self) -> Optional[Dict[str, Any]]:
        """Detect if concept drift has occurred"""
        if len(self.validation_history) < 100:
            return None

        # Compare old vs new data distributions
        mid_point = len(self.validation_history) // 2
        old_records = list(self.validation_history)[:mid_point]
        new_records = list(self.validation_history)[mid_point:]

        # Simple drift detection based on outcome distribution
        old_success_rate = sum(1 for r in old_records if r.actual_outcome == "success") / len(old_records)
        new_success_rate = sum(1 for r in new_records if r.actual_outcome == "success") / len(new_records)

        drift_magnitude = abs(old_success_rate - new_success_rate)

        if drift_magnitude > 0.2:
            return {
                "detected": True,
                "confidence": min(drift_magnitude * 2, 1.0),
                "old_success_rate": old_success_rate,
                "new_success_rate": new_success_rate
            }

        return None

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or not hasattr(self.hypothesis_classifier, 'feature_importances_'):
            return {}

        importances = self.hypothesis_classifier.feature_importances_

        # Map to feature names (simplified)
        feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(zip(feature_names, importances.tolist()))

    def _get_current_accuracy(self) -> float:
        """Get current model accuracy"""
        if not self.performance_history:
            return 0.0
        return self.performance_history[-1].accuracy

    async def save_model(self, path: Path) -> None:
        """Save the adaptive learning model"""
        model_data = {
            "hypothesis_classifier": self.hypothesis_classifier,
            "confidence_predictor": self.confidence_predictor,
            "text_vectorizer": self.text_vectorizer,
            "feature_scaler": self.feature_scaler,
            "pattern_database": self.pattern_database,
            "performance_history": self.performance_history,
            "model_version": self.model_version
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    async def load_model(self, path: Path) -> None:
        """Load a saved adaptive learning model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.hypothesis_classifier = model_data["hypothesis_classifier"]
        self.confidence_predictor = model_data["confidence_predictor"]
        self.text_vectorizer = model_data["text_vectorizer"]
        self.feature_scaler = model_data["feature_scaler"]
        self.pattern_database = model_data["pattern_database"]
        self.performance_history = model_data["performance_history"]
        self.model_version = model_data["model_version"]
        self.is_trained = True

        logger.info(f"Model loaded from {path}")