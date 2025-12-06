"""
Intelligent routing logic for model selection
"""

import os
from typing import List, Optional
from enum import Enum
from atlas_orchestrator.core.task import Task, TaskType, ModelCapabilities
from atlas_orchestrator.core.config import OrchestratorConfig


class RoutingStrategy(str, Enum):
    """Strategy for selecting models"""
    COST_OPTIMIZED = "cost_optimized"  # Choose cheapest suitable model
    QUALITY_OPTIMIZED = "quality_optimized"  # Choose best model
    BALANCED = "balanced"  # Balance cost and quality
    FASTEST = "fastest"  # Choose fastest model
    CUSTOM = "custom"  # Use custom routing rules


class Router:
    """
    Intelligent router for selecting the best model for a task

    Features:
    - Task-type based routing
    - Cost optimization
    - Quality optimization
    - Fallback chains
    - Custom routing rules
    """

    def __init__(self, config: OrchestratorConfig, strategy: RoutingStrategy = RoutingStrategy.BALANCED):
        self.config = config
        self.strategy = strategy
        self.model_configs = config.model_configs
        self.routing_rules = config.routing_rules
        self._debug_routing = os.getenv("ORCHESTRATOR_DEBUG_ROUTING", "0").lower() not in {"0", "false", "no"}

    def select_model(self, task: Task) -> str:
        """
        Select the best model for a task

        Args:
            task: The task to route

        Returns:
            Model ID to use
        """
        # Use custom fallback chain if provided
        if task.fallback_chain:
            return task.fallback_chain[0]

        # Get candidates based on task type
        candidates = self._get_candidates_for_task_type(task.task_type)

        if not candidates:
            # Fall back to default model
            return self.config.default_model

        # Apply routing strategy
        if self.strategy == RoutingStrategy.COST_OPTIMIZED:
            selected = self._select_cheapest(candidates, task)
        elif self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            selected = self._select_best_quality(candidates, task)
        elif self.strategy == RoutingStrategy.BALANCED:
            selected = self._select_balanced(candidates, task)
        elif self.strategy == RoutingStrategy.FASTEST:
            selected = self._select_fastest(candidates, task)
        else:
            # CUSTOM or default
            selected = candidates[0]

        self._log_routing(task, selected, f"strategy={self.strategy.value}")
        return selected

    def get_fallback_chain(self, task: Task, primary_model: Optional[str] = None) -> List[str]:
        """
        Get fallback chain for a task

        Args:
            task: The task
            primary_model: Primary model (if None, will be selected)

        Returns:
            List of model IDs in fallback order
        """
        # Use custom fallback if provided
        if task.fallback_chain:
            return task.fallback_chain

        if task.task_type in self.routing_rules:
            chain = list(self.routing_rules[task.task_type])
        else:
            chain = [self.config.default_model]

        # Ensure primary model leads the chain
        model = primary_model or self.select_model(task)
        if model:
            chain = [model] + [m for m in chain if m != model]
        return chain

    def _get_candidates_for_task_type(self, task_type: TaskType) -> List[str]:
        """Get candidate models for a task type"""
        if task_type in self.routing_rules:
            return self.routing_rules[task_type]
        return [self.config.default_model]

    def _select_cheapest(self, candidates: List[str], task: Task) -> str:
        """Select cheapest model from candidates"""
        cheapest = None
        min_cost = float('inf')

        for model_id in candidates:
            if model_id not in self.model_configs:
                continue

            model = self.model_configs[model_id]
            if not model.available:
                continue

            # Estimate cost (assume avg 500 input, 500 output tokens)
            estimated_cost = model.calculate_cost(500, 500)

            if estimated_cost < min_cost:
                min_cost = estimated_cost
                cheapest = model_id

        return cheapest or candidates[0]

    def _select_best_quality(self, candidates: List[str], task: Task) -> str:
        """Select highest quality model"""
        # Quality proxy: higher cost usually means better quality
        best = None
        max_cost = 0.0

        for model_id in candidates:
            if model_id not in self.model_configs:
                continue

            model = self.model_configs[model_id]
            if not model.available:
                continue

            # Check if model is good at this task type
            if task.task_type in model.strengths:
                estimated_cost = model.calculate_cost(500, 500)
                if estimated_cost > max_cost:
                    max_cost = estimated_cost
                    best = model_id

        return best or candidates[0]

    def _select_balanced(self, candidates: List[str], task: Task) -> str:
        """Balance cost and quality"""
        scores = {}

        for model_id in candidates:
            if model_id not in self.model_configs:
                continue

            model = self.model_configs[model_id]
            if not model.available:
                continue

            # Calculate score
            score = 0.0

            # Quality score (if model is strong at this task type)
            if task.task_type in model.strengths:
                score += 10.0

            # Cost score (inverse of cost)
            estimated_cost = model.calculate_cost(500, 500)
            if estimated_cost > 0:
                score += (1.0 / estimated_cost) * 100

            # Context window score
            if model.max_tokens >= 100000:
                score += 5.0

            scores[model_id] = score

        if not scores:
            return candidates[0]

        # Return model with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def _select_fastest(self, candidates: List[str], task: Task) -> str:
        """Select fastest model (usually cheaper models are faster)"""
        # For now, use cheapest as proxy for fastest
        return self._select_cheapest(candidates, task)

    def estimate_cost(self, task: Task, model_id: str) -> float:
        """
        Estimate cost for a task with a specific model

        Args:
            task: The task
            model_id: Model to use

        Returns:
            Estimated cost in USD
        """
        if model_id not in self.model_configs:
            return 0.0

        model = self.model_configs[model_id]

        # Estimate input tokens from prompt length
        input_tokens = len(task.prompt.split()) * 1.3  # rough estimate
        if task.context:
            input_tokens += len(task.context.split()) * 1.3

        # Use max_tokens as output estimate
        output_tokens = task.max_tokens

        return model.calculate_cost(int(input_tokens), output_tokens)

    def get_model_info(self, model_id: str) -> Optional[ModelCapabilities]:
        """Get information about a model"""
        return self.model_configs.get(model_id)

    def _log_routing(self, task: Task, model_id: str, details: str) -> None:
        """Emit debug info when routing tracing is enabled."""
        if not self._debug_routing:
            return
        task_type = getattr(task.task_type, "value", str(task.task_type))
        print(f"[orchestrator] routed task_type={task_type} task_id={task.task_id or 'n/a'} -> {model_id} ({details})")
