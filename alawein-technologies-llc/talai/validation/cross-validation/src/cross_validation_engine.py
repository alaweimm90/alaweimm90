"""
Multi-Provider Cross-Validation Engine for TalAI

Core engine that validates hypotheses across multiple LLM providers to ensure
consistency, detect bias, and optimize for cost-effectiveness while maintaining reliability.
"""

import asyncio
import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np
from collections import defaultdict
import aiohttp
import statistics
from scipy import stats

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC_CLAUDE = "anthropic_claude"
    OPENAI_GPT4 = "openai_gpt4"
    GOOGLE_GEMINI = "google_gemini"
    META_LLAMA = "meta_llama"
    COHERE_COMMAND = "cohere_command"
    MISTRAL_LARGE = "mistral_large"
    TOGETHER_AI = "together_ai"
    PERPLEXITY = "perplexity"


class AgreementLevel(Enum):
    """Levels of agreement between providers"""
    UNANIMOUS = "unanimous"  # All providers agree
    STRONG_CONSENSUS = "strong_consensus"  # 80%+ agreement
    MAJORITY = "majority"  # 50-80% agreement
    DISAGREEMENT = "disagreement"  # <50% agreement
    CONFLICT = "conflict"  # Direct contradictions


class BiasType(Enum):
    """Types of bias detected"""
    PROVIDER_BIAS = "provider_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    RECENCY_BIAS = "recency_bias"
    CULTURAL_BIAS = "cultural_bias"
    TECHNICAL_BIAS = "technical_bias"
    SAMPLING_BIAS = "sampling_bias"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: str
    endpoint: str
    model_name: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    latency_ms_avg: float
    reliability_score: float  # 0-1 based on historical uptime
    capabilities: List[str]
    rate_limits: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRequest:
    """Request for hypothesis validation"""
    id: str
    hypothesis: str
    domain: str
    context: Dict[str, Any]
    required_confidence: float
    max_cost_usd: float
    timeout_seconds: int
    priority: int  # 1-10
    providers_requested: Optional[List[LLMProvider]] = None
    fallback_enabled: bool = True


@dataclass
class ProviderResponse:
    """Response from a single provider"""
    request_id: str
    provider: LLMProvider
    valid: bool
    confidence: float
    reasoning: str
    evidence: List[str]
    concerns: List[str]
    latency_ms: float
    tokens_used: int
    cost_usd: float
    timestamp: datetime
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationResult:
    """Aggregated cross-validation result"""
    request_id: str
    hypothesis: str
    final_verdict: bool
    confidence_score: float
    agreement_level: AgreementLevel
    providers_used: List[LLMProvider]
    provider_responses: List[ProviderResponse]
    disagreements: List[Dict[str, Any]]
    biases_detected: List[Dict[str, Any]]
    ensemble_weight: Dict[LLMProvider, float]
    total_cost_usd: float
    total_latency_ms: float
    reliability_score: float
    recommendations: List[str]


class CrossValidationEngine:
    """
    Main engine for multi-provider cross-validation of TalAI hypotheses.
    Ensures consistency across different LLM providers and detects biases.
    """

    def __init__(self,
                 provider_configs: List[ProviderConfig],
                 cache_dir: Optional[Path] = None,
                 async_batch_size: int = 5):
        """
        Initialize the cross-validation engine.

        Args:
            provider_configs: Configurations for LLM providers
            cache_dir: Directory for caching results
            async_batch_size: Number of concurrent provider calls
        """
        self.provider_configs = {cfg.provider: cfg for cfg in provider_configs}
        self.cache_dir = cache_dir or Path("/tmp/talai_crossval_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.async_batch_size = async_batch_size

        # Performance tracking
        self.validation_history: List[CrossValidationResult] = []
        self.provider_performance: Dict[LLMProvider, Dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.0, "avg_latency": 0.0, "total_cost": 0.0}
        )
        self.agreement_matrix: Dict[Tuple[LLMProvider, LLMProvider], float] = {}
        self.bias_tracking: Dict[BiasType, List[Dict[str, Any]]] = defaultdict(list)

        # Cost optimization
        self.budget_used = 0.0
        self.budget_limit = float('inf')

        # Fallback chains
        self.fallback_chains = self._initialize_fallback_chains()

        # Provider weights for ensemble
        self.provider_weights = self._initialize_weights()

        logger.info(f"Initialized CrossValidationEngine with {len(provider_configs)} providers")

    def _initialize_fallback_chains(self) -> Dict[LLMProvider, List[LLMProvider]]:
        """Initialize fallback chains for each provider"""
        chains = {}
        for provider in self.provider_configs:
            # Sort other providers by reliability and cost
            other_providers = [p for p in self.provider_configs if p != provider]
            sorted_providers = sorted(
                other_providers,
                key=lambda p: (
                    -self.provider_configs[p].reliability_score,
                    self.provider_configs[p].cost_per_1k_tokens
                )
            )
            chains[provider] = sorted_providers[:3]  # Top 3 fallbacks

        return chains

    def _initialize_weights(self) -> Dict[LLMProvider, float]:
        """Initialize provider weights based on reliability"""
        weights = {}
        total_reliability = sum(cfg.reliability_score for cfg in self.provider_configs.values())

        for provider, cfg in self.provider_configs.items():
            weights[provider] = cfg.reliability_score / total_reliability

        return weights

    async def validate_hypothesis(self, request: ValidationRequest) -> CrossValidationResult:
        """
        Validate hypothesis across multiple providers.

        Args:
            request: Validation request

        Returns:
            Cross-validation result with aggregated analysis
        """
        start_time = time.time()

        # Select providers
        providers = await self._select_providers(request)

        # Query providers concurrently
        responses = await self._query_providers_concurrent(request, providers)

        # Handle failures with fallbacks
        responses = await self._handle_fallbacks(request, responses, providers)

        # Analyze agreement
        agreement_level, disagreements = self._analyze_agreement(responses)

        # Detect biases
        biases = await self._detect_biases(request, responses)

        # Calculate ensemble verdict
        final_verdict, confidence = self._calculate_ensemble_verdict(responses)

        # Update performance metrics
        self._update_performance_metrics(responses)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            agreement_level, biases, disagreements
        )

        # Calculate totals
        total_cost = sum(r.cost_usd for r in responses if r.error is None)
        total_latency = (time.time() - start_time) * 1000

        result = CrossValidationResult(
            request_id=request.id,
            hypothesis=request.hypothesis,
            final_verdict=final_verdict,
            confidence_score=confidence,
            agreement_level=agreement_level,
            providers_used=[r.provider for r in responses],
            provider_responses=responses,
            disagreements=disagreements,
            biases_detected=biases,
            ensemble_weight=self.provider_weights,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            reliability_score=self._calculate_reliability_score(responses),
            recommendations=recommendations
        )

        # Cache result
        await self._cache_result(result)

        # Update history
        self.validation_history.append(result)
        self.budget_used += total_cost

        logger.info(
            f"Validated {request.id}: verdict={final_verdict}, "
            f"confidence={confidence:.2%}, agreement={agreement_level.value}"
        )

        return result

    async def _select_providers(self, request: ValidationRequest) -> List[LLMProvider]:
        """Select providers based on request requirements and optimization"""
        if request.providers_requested:
            return request.providers_requested

        # Cost-optimized selection
        available_providers = []
        remaining_budget = request.max_cost_usd

        # Sort by cost-effectiveness
        sorted_providers = sorted(
            self.provider_configs.items(),
            key=lambda x: x[1].cost_per_1k_tokens / x[1].reliability_score
        )

        for provider, cfg in sorted_providers:
            estimated_cost = (cfg.cost_per_1k_tokens * 2)  # Estimate 2k tokens
            if estimated_cost <= remaining_budget:
                available_providers.append(provider)
                remaining_budget -= estimated_cost

                # Aim for at least 3 providers for good cross-validation
                if len(available_providers) >= 3:
                    break

        # Ensure minimum diversity
        if len(available_providers) < 2:
            # Add most reliable providers regardless of cost
            for provider in [LLMProvider.ANTHROPIC_CLAUDE, LLMProvider.OPENAI_GPT4]:
                if provider in self.provider_configs and provider not in available_providers:
                    available_providers.append(provider)

        return available_providers

    async def _query_providers_concurrent(self,
                                        request: ValidationRequest,
                                        providers: List[LLMProvider]) -> List[ProviderResponse]:
        """Query multiple providers concurrently"""
        responses = []

        # Batch providers for concurrent execution
        for i in range(0, len(providers), self.async_batch_size):
            batch = providers[i:i + self.async_batch_size]
            batch_tasks = [
                self._query_single_provider(request, provider)
                for provider in batch
            ]

            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for response in batch_responses:
                if isinstance(response, ProviderResponse):
                    responses.append(response)
                else:
                    logger.error(f"Provider query failed: {response}")

        return responses

    async def _query_single_provider(self,
                                   request: ValidationRequest,
                                   provider: LLMProvider) -> ProviderResponse:
        """Query a single provider for validation"""
        config = self.provider_configs[provider]
        start_time = time.time()

        try:
            # Check cache first
            cached = await self._check_cache(request.id, provider)
            if cached:
                return cached

            # Simulate provider API call (in production, would use actual APIs)
            await asyncio.sleep(config.latency_ms_avg / 1000)  # Simulate latency

            # Generate simulated response based on provider characteristics
            # In production, this would call actual provider APIs
            response = await self._simulate_provider_response(request, provider, config)

            response.latency_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            logger.error(f"Error querying {provider.value}: {e}")
            return ProviderResponse(
                request_id=request.id,
                provider=provider,
                valid=False,
                confidence=0.0,
                reasoning="Provider query failed",
                evidence=[],
                concerns=[],
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cost_usd=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _simulate_provider_response(self,
                                        request: ValidationRequest,
                                        provider: LLMProvider,
                                        config: ProviderConfig) -> ProviderResponse:
        """Simulate provider response for testing"""
        import random

        # Base validation probability influenced by provider characteristics
        base_prob = 0.7

        # Adjust based on provider-specific biases (simulated)
        if provider == LLMProvider.ANTHROPIC_CLAUDE:
            base_prob += 0.05  # Slightly more conservative
        elif provider == LLMProvider.OPENAI_GPT4:
            base_prob += 0.08  # Slightly more optimistic
        elif provider == LLMProvider.GOOGLE_GEMINI:
            base_prob += 0.06
        elif provider == LLMProvider.META_LLAMA:
            base_prob += 0.04

        # Add some randomness
        confidence = min(0.99, max(0.1, base_prob + random.gauss(0, 0.1)))
        valid = random.random() < confidence

        # Generate reasoning
        reasoning = self._generate_reasoning(provider, valid, confidence)

        # Generate evidence
        evidence = [
            f"Evidence point {i+1} from {provider.value}"
            for i in range(random.randint(2, 5))
        ]

        # Generate concerns
        concerns = []
        if confidence < 0.8:
            concerns.append("Limited supporting evidence")
        if confidence < 0.6:
            concerns.append("Conflicting prior research")

        # Calculate tokens and cost
        tokens_used = random.randint(500, 2000)
        cost_usd = (tokens_used / 1000) * config.cost_per_1k_tokens

        return ProviderResponse(
            request_id=request.id,
            provider=provider,
            valid=valid,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            concerns=concerns,
            latency_ms=0,  # Will be set by caller
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            timestamp=datetime.now(),
            error=None
        )

    def _generate_reasoning(self, provider: LLMProvider, valid: bool, confidence: float) -> str:
        """Generate provider-specific reasoning"""
        if valid:
            if confidence > 0.8:
                return f"{provider.value}: Strong evidence supports this hypothesis"
            else:
                return f"{provider.value}: Moderate evidence supports this hypothesis with some uncertainties"
        else:
            return f"{provider.value}: Insufficient evidence to validate this hypothesis"

    async def _handle_fallbacks(self,
                              request: ValidationRequest,
                              responses: List[ProviderResponse],
                              original_providers: List[LLMProvider]) -> List[ProviderResponse]:
        """Handle failed providers with fallback chains"""
        failed_providers = [
            p for p in original_providers
            if not any(r.provider == p and r.error is None for r in responses)
        ]

        for failed_provider in failed_providers:
            fallback_chain = self.fallback_chains.get(failed_provider, [])

            for fallback in fallback_chain:
                if fallback not in original_providers:
                    logger.info(f"Trying fallback {fallback.value} for {failed_provider.value}")

                    fallback_response = await self._query_single_provider(request, fallback)

                    if fallback_response.error is None:
                        responses.append(fallback_response)
                        break

        return responses

    def _analyze_agreement(self,
                          responses: List[ProviderResponse]) -> Tuple[AgreementLevel, List[Dict[str, Any]]]:
        """Analyze agreement between provider responses"""
        valid_responses = [r for r in responses if r.error is None]

        if not valid_responses:
            return AgreementLevel.CONFLICT, []

        # Count validations
        valid_count = sum(1 for r in valid_responses if r.valid)
        total_count = len(valid_responses)
        agreement_ratio = valid_count / total_count

        # Determine agreement level
        if agreement_ratio == 1.0 or agreement_ratio == 0.0:
            agreement_level = AgreementLevel.UNANIMOUS
        elif agreement_ratio >= 0.8:
            agreement_level = AgreementLevel.STRONG_CONSENSUS
        elif agreement_ratio >= 0.5:
            agreement_level = AgreementLevel.MAJORITY
        else:
            agreement_level = AgreementLevel.DISAGREEMENT

        # Find specific disagreements
        disagreements = []
        for i, r1 in enumerate(valid_responses):
            for r2 in valid_responses[i+1:]:
                if r1.valid != r2.valid:
                    disagreements.append({
                        "provider1": r1.provider.value,
                        "provider2": r2.provider.value,
                        "provider1_verdict": r1.valid,
                        "provider2_verdict": r2.valid,
                        "confidence_delta": abs(r1.confidence - r2.confidence)
                    })

        # Check for conflicts in reasoning
        if disagreements and any(d["confidence_delta"] > 0.5 for d in disagreements):
            agreement_level = AgreementLevel.CONFLICT

        # Update agreement matrix
        for i, r1 in enumerate(valid_responses):
            for r2 in valid_responses[i+1:]:
                key = (r1.provider, r2.provider)
                if key not in self.agreement_matrix:
                    self.agreement_matrix[key] = []
                self.agreement_matrix[key].append(1.0 if r1.valid == r2.valid else 0.0)

        return agreement_level, disagreements

    async def _detect_biases(self,
                           request: ValidationRequest,
                           responses: List[ProviderResponse]) -> List[Dict[str, Any]]:
        """Detect various types of bias in provider responses"""
        biases = []
        valid_responses = [r for r in responses if r.error is None]

        if not valid_responses:
            return biases

        # Provider bias - systematic differences
        provider_verdicts = defaultdict(list)
        for r in valid_responses:
            provider_verdicts[r.provider].append(r.valid)

        for provider, verdicts in provider_verdicts.items():
            if len(self.validation_history) > 10:
                historical_rate = self._get_historical_validation_rate(provider)
                current_rate = sum(verdicts) / len(verdicts)

                if abs(historical_rate - current_rate) > 0.3:
                    biases.append({
                        "type": BiasType.PROVIDER_BIAS.value,
                        "provider": provider.value,
                        "description": "Significant deviation from historical validation rate",
                        "historical_rate": historical_rate,
                        "current_rate": current_rate
                    })

        # Confirmation bias - providers agreeing too readily
        confidence_scores = [r.confidence for r in valid_responses]
        if len(confidence_scores) > 2:
            confidence_std = statistics.stdev(confidence_scores)
            if confidence_std < 0.05:  # Very low variance
                biases.append({
                    "type": BiasType.CONFIRMATION_BIAS.value,
                    "description": "Suspiciously similar confidence scores across providers",
                    "confidence_scores": confidence_scores,
                    "standard_deviation": confidence_std
                })

        # Technical bias - providers with similar architectures agreeing
        architecture_groups = {
            "transformer": [LLMProvider.OPENAI_GPT4, LLMProvider.ANTHROPIC_CLAUDE],
            "mixture": [LLMProvider.MISTRAL_LARGE, LLMProvider.META_LLAMA]
        }

        for arch_name, providers in architecture_groups.items():
            arch_responses = [r for r in valid_responses if r.provider in providers]
            if len(arch_responses) >= 2:
                arch_agreement = sum(1 for r in arch_responses if r.valid) / len(arch_responses)
                overall_agreement = sum(1 for r in valid_responses if r.valid) / len(valid_responses)

                if abs(arch_agreement - overall_agreement) > 0.3:
                    biases.append({
                        "type": BiasType.TECHNICAL_BIAS.value,
                        "architecture": arch_name,
                        "description": f"{arch_name} models show different validation pattern",
                        "architecture_agreement": arch_agreement,
                        "overall_agreement": overall_agreement
                    })

        # Sampling bias - if certain providers consistently not queried
        if len(self.validation_history) > 20:
            provider_usage = defaultdict(int)
            for result in self.validation_history[-20:]:
                for provider in result.providers_used:
                    provider_usage[provider] += 1

            avg_usage = sum(provider_usage.values()) / len(self.provider_configs)
            for provider in self.provider_configs:
                usage = provider_usage.get(provider, 0)
                if usage < avg_usage * 0.3:  # Significantly underused
                    biases.append({
                        "type": BiasType.SAMPLING_BIAS.value,
                        "provider": provider.value,
                        "description": "Provider significantly underrepresented in recent validations",
                        "usage_count": usage,
                        "average_usage": avg_usage
                    })

        # Track biases
        for bias in biases:
            self.bias_tracking[BiasType(bias["type"])].append({
                "timestamp": datetime.now(),
                "request_id": request.id,
                "details": bias
            })

        return biases

    def _get_historical_validation_rate(self, provider: LLMProvider) -> float:
        """Get historical validation rate for a provider"""
        provider_validations = []
        for result in self.validation_history:
            for response in result.provider_responses:
                if response.provider == provider and response.error is None:
                    provider_validations.append(response.valid)

        if not provider_validations:
            return 0.5  # Default to neutral

        return sum(provider_validations) / len(provider_validations)

    def _calculate_ensemble_verdict(self,
                                  responses: List[ProviderResponse]) -> Tuple[bool, float]:
        """Calculate weighted ensemble verdict"""
        valid_responses = [r for r in responses if r.error is None]

        if not valid_responses:
            return False, 0.0

        # Weighted voting
        weighted_score = 0.0
        total_weight = 0.0

        for response in valid_responses:
            weight = self.provider_weights.get(response.provider, 1.0)

            # Adjust weight based on confidence
            adjusted_weight = weight * response.confidence

            if response.valid:
                weighted_score += adjusted_weight

            total_weight += adjusted_weight

        if total_weight == 0:
            return False, 0.0

        ensemble_confidence = weighted_score / total_weight
        ensemble_verdict = ensemble_confidence >= 0.5

        return ensemble_verdict, ensemble_confidence

    def _calculate_reliability_score(self, responses: List[ProviderResponse]) -> float:
        """Calculate overall reliability score for the validation"""
        if not responses:
            return 0.0

        factors = []

        # Success rate
        success_rate = sum(1 for r in responses if r.error is None) / len(responses)
        factors.append(success_rate)

        # Average confidence of successful responses
        valid_responses = [r for r in responses if r.error is None]
        if valid_responses:
            avg_confidence = statistics.mean(r.confidence for r in valid_responses)
            factors.append(avg_confidence)

        # Provider reliability scores
        avg_reliability = statistics.mean(
            self.provider_configs[r.provider].reliability_score
            for r in valid_responses
        ) if valid_responses else 0.0
        factors.append(avg_reliability)

        return statistics.mean(factors)

    def _generate_recommendations(self,
                                agreement_level: AgreementLevel,
                                biases: List[Dict[str, Any]],
                                disagreements: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Agreement-based recommendations
        if agreement_level == AgreementLevel.CONFLICT:
            recommendations.append("CRITICAL: Direct conflicts detected - manual review required")
        elif agreement_level == AgreementLevel.DISAGREEMENT:
            recommendations.append("Significant disagreement - consider additional validation")
        elif agreement_level == AgreementLevel.MAJORITY:
            recommendations.append("Moderate agreement - results should be interpreted cautiously")

        # Bias-based recommendations
        for bias in biases:
            if bias["type"] == BiasType.PROVIDER_BIAS.value:
                recommendations.append(f"Provider bias detected in {bias['provider']} - consider alternative providers")
            elif bias["type"] == BiasType.CONFIRMATION_BIAS.value:
                recommendations.append("Potential confirmation bias - increase provider diversity")
            elif bias["type"] == BiasType.TECHNICAL_BIAS.value:
                recommendations.append(f"Technical bias in {bias['architecture']} models - use diverse architectures")
            elif bias["type"] == BiasType.SAMPLING_BIAS.value:
                recommendations.append(f"Include {bias['provider']} more frequently for balanced validation")

        # Disagreement-based recommendations
        if len(disagreements) > 2:
            recommendations.append("Multiple disagreements found - hypothesis may be ambiguous")

        # Cost optimization recommendations
        if hasattr(self, 'budget_used') and self.budget_used > self.budget_limit * 0.8:
            recommendations.append("Approaching budget limit - consider using cheaper providers")

        return recommendations

    def _update_performance_metrics(self, responses: List[ProviderResponse]) -> None:
        """Update provider performance metrics"""
        for response in responses:
            provider = response.provider
            metrics = self.provider_performance[provider]

            # Update success rate
            if response.error is None:
                metrics["success_rate"] = (
                    metrics["success_rate"] * 0.9 + 1.0 * 0.1
                )  # Exponential moving average
            else:
                metrics["success_rate"] = metrics["success_rate"] * 0.9

            # Update average latency
            if response.error is None:
                metrics["avg_latency"] = (
                    metrics["avg_latency"] * 0.9 + response.latency_ms * 0.1
                )

            # Update total cost
            metrics["total_cost"] += response.cost_usd

        # Update provider weights based on performance
        self._update_provider_weights()

    def _update_provider_weights(self) -> None:
        """Update provider weights based on recent performance"""
        for provider in self.provider_configs:
            metrics = self.provider_performance[provider]

            # Combine success rate and base reliability
            base_reliability = self.provider_configs[provider].reliability_score
            recent_success = metrics["success_rate"]

            # Weighted average favoring recent performance
            new_weight = base_reliability * 0.3 + recent_success * 0.7

            self.provider_weights[provider] = new_weight

        # Normalize weights
        total_weight = sum(self.provider_weights.values())
        if total_weight > 0:
            for provider in self.provider_weights:
                self.provider_weights[provider] /= total_weight

    async def _check_cache(self, request_id: str, provider: LLMProvider) -> Optional[ProviderResponse]:
        """Check cache for previous response"""
        cache_key = f"{request_id}_{provider.value}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to ProviderResponse
                    return ProviderResponse(
                        request_id=data["request_id"],
                        provider=LLMProvider(data["provider"]),
                        valid=data["valid"],
                        confidence=data["confidence"],
                        reasoning=data["reasoning"],
                        evidence=data["evidence"],
                        concerns=data["concerns"],
                        latency_ms=data["latency_ms"],
                        tokens_used=data["tokens_used"],
                        cost_usd=data["cost_usd"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        error=data.get("error")
                    )
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")

        return None

    async def _cache_result(self, result: CrossValidationResult) -> None:
        """Cache validation result"""
        for response in result.provider_responses:
            if response.error is None:
                cache_key = f"{result.request_id}_{response.provider.value}"
                cache_file = self.cache_dir / f"{cache_key}.json"

                try:
                    with open(cache_file, 'w') as f:
                        json.dump({
                            "request_id": response.request_id,
                            "provider": response.provider.value,
                            "valid": response.valid,
                            "confidence": response.confidence,
                            "reasoning": response.reasoning,
                            "evidence": response.evidence,
                            "concerns": response.concerns,
                            "latency_ms": response.latency_ms,
                            "tokens_used": response.tokens_used,
                            "cost_usd": response.cost_usd,
                            "timestamp": response.timestamp.isoformat(),
                            "error": response.error
                        }, f)
                except Exception as e:
                    logger.warning(f"Failed to cache result for {cache_key}: {e}")

    async def optimize_provider_selection(self,
                                        budget: float,
                                        required_confidence: float,
                                        time_constraint_ms: Optional[float] = None) -> List[LLMProvider]:
        """
        Optimize provider selection for given constraints.

        Args:
            budget: Maximum budget in USD
            required_confidence: Required confidence level
            time_constraint_ms: Optional time constraint in milliseconds

        Returns:
            Optimized list of providers
        """
        # Dynamic programming approach for provider selection
        providers = []
        remaining_budget = budget
        current_confidence = 0.0

        # Sort providers by value (reliability * speed / cost)
        provider_value = []
        for provider, cfg in self.provider_configs.items():
            speed_factor = 1.0 / (cfg.latency_ms_avg / 1000) if time_constraint_ms else 1.0
            value = (cfg.reliability_score * speed_factor) / cfg.cost_per_1k_tokens
            provider_value.append((provider, value, cfg))

        provider_value.sort(key=lambda x: x[1], reverse=True)

        for provider, value, cfg in provider_value:
            estimated_cost = cfg.cost_per_1k_tokens * 2  # Estimate 2k tokens

            if estimated_cost <= remaining_budget:
                if time_constraint_ms and cfg.latency_ms_avg > time_constraint_ms:
                    continue

                providers.append(provider)
                remaining_budget -= estimated_cost
                current_confidence = 1 - (1 - current_confidence) * (1 - cfg.reliability_score)

                if current_confidence >= required_confidence:
                    break

        return providers

    async def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.validation_history:
            return {"error": "No validation history available"}

        # Calculate statistics
        total_validations = len(self.validation_history)
        total_cost = sum(r.total_cost_usd for r in self.validation_history)
        avg_latency = statistics.mean(r.total_latency_ms for r in self.validation_history)

        # Agreement statistics
        agreement_counts = defaultdict(int)
        for result in self.validation_history:
            agreement_counts[result.agreement_level] += 1

        # Bias statistics
        bias_counts = defaultdict(int)
        for bias_list in self.bias_tracking.values():
            bias_counts[bias_list[0]["details"]["type"]] = len(bias_list) if bias_list else 0

        # Provider performance
        provider_stats = {}
        for provider, metrics in self.provider_performance.items():
            provider_stats[provider.value] = {
                "success_rate": f"{metrics['success_rate']:.2%}",
                "avg_latency_ms": f"{metrics['avg_latency']:.1f}",
                "total_cost_usd": f"${metrics['total_cost']:.2f}"
            }

        # Agreement matrix analysis
        agreement_pairs = {}
        for (p1, p2), agreements in self.agreement_matrix.items():
            if agreements:
                agreement_rate = statistics.mean(agreements)
                agreement_pairs[f"{p1.value}-{p2.value}"] = f"{agreement_rate:.2%}"

        report = {
            "summary": {
                "total_validations": total_validations,
                "total_cost_usd": f"${total_cost:.2f}",
                "average_latency_ms": f"{avg_latency:.1f}",
                "providers_active": len(self.provider_configs)
            },
            "agreement_distribution": {
                level.value: count for level, count in agreement_counts.items()
            },
            "bias_detections": dict(bias_counts),
            "provider_performance": provider_stats,
            "provider_agreement_matrix": agreement_pairs,
            "recommendations": {
                "most_reliable_provider": max(
                    self.provider_weights.items(),
                    key=lambda x: x[1]
                )[0].value if self.provider_weights else "N/A",
                "most_cost_effective": min(
                    self.provider_configs.items(),
                    key=lambda x: x[1].cost_per_1k_tokens / x[1].reliability_score
                )[0].value if self.provider_configs else "N/A"
            }
        }

        return report

    async def export_results(self, output_path: Path) -> None:
        """Export validation results to file"""
        report = await self.generate_validation_report()

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "report": report,
            "validation_history": [
                {
                    "request_id": r.request_id,
                    "hypothesis": r.hypothesis[:200],  # Truncate for readability
                    "final_verdict": r.final_verdict,
                    "confidence_score": r.confidence_score,
                    "agreement_level": r.agreement_level.value,
                    "providers_used": [p.value for p in r.providers_used],
                    "total_cost_usd": r.total_cost_usd,
                    "total_latency_ms": r.total_latency_ms,
                    "biases_detected": len(r.biases_detected),
                    "recommendations": r.recommendations
                }
                for r in self.validation_history
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported results to {output_path}")


if __name__ == "__main__":
    # Example usage
    async def main():
        # Configure providers
        provider_configs = [
            ProviderConfig(
                provider=LLMProvider.ANTHROPIC_CLAUDE,
                api_key="sk-ant-...",
                endpoint="https://api.anthropic.com/v1",
                model_name="claude-3-opus",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.015,
                latency_ms_avg=2000,
                reliability_score=0.95,
                capabilities=["reasoning", "analysis", "code"],
                rate_limits={"requests_per_minute": 50}
            ),
            ProviderConfig(
                provider=LLMProvider.OPENAI_GPT4,
                api_key="sk-...",
                endpoint="https://api.openai.com/v1",
                model_name="gpt-4-turbo",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.01,
                latency_ms_avg=1500,
                reliability_score=0.92,
                capabilities=["reasoning", "analysis", "creativity"],
                rate_limits={"requests_per_minute": 60}
            ),
            ProviderConfig(
                provider=LLMProvider.GOOGLE_GEMINI,
                api_key="AIza...",
                endpoint="https://generativelanguage.googleapis.com/v1",
                model_name="gemini-pro",
                max_tokens=4096,
                temperature=0.7,
                cost_per_1k_tokens=0.005,
                latency_ms_avg=1800,
                reliability_score=0.88,
                capabilities=["reasoning", "multimodal"],
                rate_limits={"requests_per_minute": 40}
            )
        ]

        # Initialize engine
        engine = CrossValidationEngine(provider_configs)

        # Create validation request
        request = ValidationRequest(
            id="val_001",
            hypothesis="Quantum entanglement can be used for faster-than-light communication",
            domain="physics",
            context={"subfield": "quantum_mechanics"},
            required_confidence=0.8,
            max_cost_usd=1.0,
            timeout_seconds=30,
            priority=8
        )

        # Validate hypothesis
        result = await engine.validate_hypothesis(request)

        print(f"Final Verdict: {result.final_verdict}")
        print(f"Confidence: {result.confidence_score:.2%}")
        print(f"Agreement Level: {result.agreement_level.value}")
        print(f"Total Cost: ${result.total_cost_usd:.3f}")
        print(f"Biases Detected: {len(result.biases_detected)}")
        print(f"Recommendations: {result.recommendations}")

        # Generate report
        report = await engine.generate_validation_report()
        print("\nValidation Report:")
        print(json.dumps(report, indent=2))

        # Export results
        await engine.export_results(Path("../reports/crossval_results.json"))

    asyncio.run(main())