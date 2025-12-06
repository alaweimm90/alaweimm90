"""Legacy-friendly AI method selector facade."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

from Librex.ai.method_selector import MethodRecommendation, MethodSelector
from Librex.ai.models import MethodSelectorModel


class AIMethodSelector:
    """Thin compatibility wrapper that mimics the historical selector API."""

    def __init__(
        self,
        cache_recommendations: bool = True,
        cache_ttl: int = 300,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379/0"
    ) -> None:
        self.cache_enabled = cache_recommendations
        self.cache_ttl = cache_ttl
        self._selector = MethodSelector()
        self.model = MethodSelectorModel()
        # Mark as trained so downstream code can call save/load without errors
        self.model.is_trained = True

        self._local_cache: Dict[str, str] = {}
        self._feedback_stats: Dict[str, Dict[str, float]] = {}
        self._method_bias: Dict[str, float] = {}

        self.use_redis = bool(use_redis and redis is not None)
        self._redis_url = redis_url
        self._redis_client = None
        self._redis_keys: set[str] = set()
        if self.use_redis:
            try:
                self._redis_client = redis.Redis.from_url(redis_url)
            except Exception:  # pragma: no cover - optional dependency
                self.use_redis = False
                self._redis_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(
        self,
        problem: Any,
        top_k: int = 3,
        constraints: Optional[Dict[str, Any]] = None,
        explain: bool = False
    ) -> List[MethodRecommendation]:
        key = self._cache_key(problem, top_k, constraints, explain)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        raw = self._selector.recommend_top_k(problem, k=top_k)
        enriched = [
            self._enrich_recommendation(problem, rec, constraints or {}, explain)
            for rec in raw
        ]

        filtered = self._apply_constraints(enriched, constraints)
        self._write_cache(key, filtered)
        return filtered

    def recommend_batch(
        self,
        problems: Iterable[Any],
        top_k: int = 3,
        constraints: Optional[Dict[str, Any]] = None,
        explain: bool = False
    ) -> List[List[MethodRecommendation]]:
        return [
            self.recommend(problem, top_k=top_k, constraints=constraints, explain=explain)
            for problem in problems
        ]

    def update_from_feedback(self, problem: Any, feedback: Dict[str, Any]) -> None:
        method = feedback.get('method')
        if not method:
            return

        stats = self._feedback_stats.setdefault(method, {'success': 0.0, 'fail': 0.0})
        if feedback.get('success', False):
            stats['success'] += 1
            delta = 0.03
        else:
            stats['fail'] += 1
            delta = -0.03

        current = self._method_bias.get(method, 0.0)
        current = max(-0.25, min(0.25, current + delta))
        self._method_bias[method] = current

    def predict_performance(self, problem: Any) -> Dict[str, Dict[str, float]]:
        predictions: Dict[str, Dict[str, float]] = {}
        recommendations = self._selector._get_all_recommendations(  # noqa: SLF001
            self._selector.feature_extractor.extract_features(problem),
            None,
            'balanced'
        )
        for rec in recommendations:
            confidence = self._apply_bias(rec.method_name, rec.confidence)
            predictions[rec.method_name] = {
                'expected_quality': min(1.0, 0.6 + confidence * 0.35),
                'expected_time': self._estimate_runtime(problem, rec.method_name),
                'confidence': confidence,
            }
        return predictions

    def clear_cache(self) -> None:
        self._local_cache.clear()
        if self._redis_client and self._redis_keys:
            try:  # pragma: no cover - best effort cleanup
                self._redis_client.delete(*self._redis_keys)
            finally:
                self._redis_keys.clear()

    def save_model(self, path: str) -> None:
        state = {
            'bias': self._method_bias,
            'feedback': self._feedback_stats,
        }
        with open(path, 'wb') as handle:
            pickle.dump(state, handle)

    def load_model(self, path: str) -> None:
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        self._method_bias = state.get('bias', {})
        self._feedback_stats = state.get('feedback', {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_constraints(
        self,
        recommendations: List[MethodRecommendation],
        constraints: Optional[Dict[str, Any]]
    ) -> List[MethodRecommendation]:
        if not constraints:
            return recommendations

        excluded = set(constraints.get('excluded_methods', []))
        max_time = constraints.get('max_time')
        min_quality = constraints.get('min_quality', 0.0)

        filtered = []
        for rec in recommendations:
            if rec.method_name in excluded:
                continue
            if max_time is not None and rec.expected_time and rec.expected_time > max_time:
                continue
            if rec.expected_quality is not None and rec.expected_quality < min_quality:
                continue
            filtered.append(rec)
        return filtered or recommendations

    def _enrich_recommendation(
        self,
        problem: Any,
        recommendation: MethodRecommendation,
        constraints: Dict[str, Any],
        explain: bool
    ) -> MethodRecommendation:
        method_name = recommendation.method_name
        confidence = self._apply_bias(method_name, recommendation.confidence)
        expected_time = self._estimate_runtime(problem, method_name)
        expected_quality = min(1.0, 0.65 + confidence * 0.3)

        explanation = recommendation.explanation
        if explain:
            features = self._selector.feature_extractor.extract_features(problem)
            explanation = (
                f"features: dim={features.dimension}, type={features.problem_type}; "
                f"reasoning: {recommendation.explanation}; "
                f"confidence_factors: {', '.join(recommendation.supporting_evidence)}"
            )

        return MethodRecommendation(
            method_name=method_name,
            config=recommendation.config,
            confidence=confidence,
            explanation=explanation,
            supporting_evidence=recommendation.supporting_evidence,
            method=method_name,
            expected_time=expected_time,
            expected_quality=expected_quality,
            metadata={'constraints': constraints, 'explain': explain}
        )

    def _apply_bias(self, method: str, confidence: float) -> float:
        bias = self._method_bias.get(method, 0.0)
        return max(0.0, min(1.0, confidence + bias))

    def _estimate_runtime(self, problem: Any, method: str) -> float:
        size = self._extract_size(problem)
        base = max(size, 1) * 0.05
        if 'genetic' in method:
            base *= 1.2
        if method in {'qaoa', 'quantum'}:
            base *= 1.5
        return round(base, 3)

    def _extract_size(self, problem: Any) -> int:
        if isinstance(problem, dict):
            for key in ('size', 'dimension', 'n'):
                if key in problem:
                    return int(problem[key])
            if 'flow' in problem:
                return int(np.asarray(problem['flow']).shape[0])
            if 'flow_matrix' in problem:
                return int(np.asarray(problem['flow_matrix']).shape[0])
        return 10

    def _cache_key(
        self,
        problem: Any,
        top_k: int,
        constraints: Optional[Dict[str, Any]],
        explain: bool
    ) -> str:
        payload = {
            'problem': self._normalize(problem),
            'top_k': top_k,
            'constraints': constraints or {},
            'explain': explain,
        }
        blob = json.dumps(payload, sort_keys=True)
        return sha256(blob.encode('utf-8')).hexdigest()

    def _read_cache(self, key: str) -> Optional[List[MethodRecommendation]]:
        cached = self._local_cache.get(key)
        if cached is None and self.use_redis and self._redis_client:
            redis_key = f"librex:ai:{key}"
            payload = self._redis_client.get(redis_key)
            if payload is not None:
                cached = payload.decode('utf-8')
        if cached is None:
            return None
        return [self._dict_to_recommendation(item) for item in json.loads(cached)]

    def _write_cache(self, key: str, recommendations: List[MethodRecommendation]) -> None:
        if not self.cache_enabled:
            return
        serialized = json.dumps([asdict(r) for r in recommendations])
        self._local_cache[key] = serialized
        if self.use_redis and self._redis_client:
            redis_key = f"librex:ai:{key}"
            self._redis_client.setex(redis_key, self.cache_ttl, serialized)
            self._redis_keys.add(redis_key)

    def _dict_to_recommendation(self, data: Dict[str, Any]) -> MethodRecommendation:
        return MethodRecommendation(
            method_name=data['method_name'],
            config=data['config'],
            confidence=data['confidence'],
            explanation=data['explanation'],
            supporting_evidence=data.get('supporting_evidence', []),
            method=data.get('method'),
            expected_time=data.get('expected_time'),
            expected_quality=data.get('expected_quality'),
            metadata=data.get('metadata')
        )

    def _normalize(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes)):
            return self._normalize(obj.tolist())
        if isinstance(obj, np.ndarray):
            return self._normalize(obj.tolist())
        return obj
