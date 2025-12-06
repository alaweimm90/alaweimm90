#!/usr/bin/env python3
"""
Prompt Engine - Self-Learning AI Infrastructure
Selects, composes, and evolves superprompts based on task analysis and feedback.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

import yaml


class ModelTier(Enum):
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    HEAVYWEIGHT = "heavyweight"


class TaskOutcome(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class CompositionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


@dataclass
class PromptSelection:
    """Result of prompt selection."""
    primary: str
    secondary: list[str]
    strategy: CompositionStrategy
    confidence: float
    reasoning: str
    composed_guidance: str
    recommended_tier: str = "standard"
    estimated_tokens: int = 0


@dataclass
class TaskRecord:
    """Record of a completed task for learning."""
    task_id: str
    timestamp: str
    query: str
    extracted_intent: str
    selected_prompts: list[str]
    composition_strategy: str
    confidence: float
    outcome: Optional[str] = None
    duration_seconds: Optional[int] = None
    user_feedback: Optional[str] = None
    context_snapshot: dict = field(default_factory=dict)


class PromptEngine:
    """
    Self-learning prompt selection and composition engine.

    This engine:
    1. Analyzes tasks to extract intent
    2. Selects appropriate superprompts based on scoring
    3. Composes guidance from multiple prompts
    4. Learns from outcomes to improve future selection
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or self._find_base_path()
        # Check both legacy and new paths for superprompts
        new_path = self.base_path / ".config" / "ai" / "superprompts"
        legacy_path = self.base_path / ".ai" / "superprompts"
        self.superprompts_dir = new_path if new_path.exists() else legacy_path

        # Learning data directory
        new_learning = self.base_path / ".config" / "ai" / "learning" / "data"
        legacy_learning = self.base_path / ".ai" / "learning" / "data"
        self.learning_dir = new_learning if new_learning.parent.exists() else legacy_learning
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.selector_config = self._load_selector_config()
        self.prompts = self._load_superprompts()
        self.prompt_stats = self._load_prompt_stats()
        self.tiering_config = self._load_tiering_config()

        # Thresholds
        self.PRIMARY_THRESHOLD = 0.6
        self.SECONDARY_THRESHOLD = 0.3
        self.EXPLORATION_RATE = 0.1

        # Token estimation (chars per token)
        self.CHARS_PER_TOKEN = 4

    def _find_base_path(self) -> Path:
        """Find the repository root."""
        current = Path.cwd()
        while current != current.parent:
            if (current / ".ai").exists() or (current / ".metaHub").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _load_selector_config(self) -> dict:
        """Load the selector configuration."""
        # Check both paths
        new_path = self.base_path / ".config" / "ai" / "prompt-engine" / "selector.yaml"
        legacy_path = self.base_path / ".ai" / "prompt-engine" / "selector.yaml"
        config_path = new_path if new_path.exists() else legacy_path
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_tiering_config(self) -> dict:
        """Load the model tiering configuration."""
        config_path = self.base_path / ".config" / "ai" / "model-tiering.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_superprompts(self) -> dict[str, dict]:
        """Load all available superprompts."""
        prompts = {}
        if self.superprompts_dir.exists():
            for prompt_file in self.superprompts_dir.glob("*.yaml"):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = yaml.safe_load(f)
                    if prompt_data and "metadata" in prompt_data:
                        prompt_id = prompt_data["metadata"].get("id", prompt_file.stem)
                        prompts[prompt_id] = prompt_data
        return prompts

    def _load_prompt_stats(self) -> dict:
        """Load prompt effectiveness statistics."""
        stats_path = self.learning_dir / "prompt-stats.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_prompt_stats(self):
        """Save prompt effectiveness statistics."""
        stats_path = self.learning_dir / "prompt-stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_stats, f, indent=2)

    def _extract_intent(self, query: str) -> tuple[str, dict[str, float]]:
        """Extract intent from query and score against each intent pattern."""
        query_lower = query.lower()
        intent_scores: dict[str, float] = {}

        patterns = self.selector_config.get("task_classification", {}).get("intent_patterns", {})

        for intent, config in patterns.items():
            keywords = config.get("keywords", [])
            weight = config.get("weight", 1.0)

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                # Normalize by number of keywords
                score = (matches / len(keywords)) * weight
                intent_scores[intent] = score

        # Get primary intent
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "general"

        return primary_intent, intent_scores

    def _calculate_prompt_score(
        self,
        prompt_id: str,
        query: str,
        intent_scores: dict[str, float]
    ) -> float:
        """Calculate score for a prompt based on query and learned stats."""
        prompt = self.prompts.get(prompt_id, {})
        triggers = prompt.get("metadata", {}).get("triggers", [])

        # Base score from trigger keyword matching
        query_lower = query.lower()
        trigger_matches = sum(1 for t in triggers if t in query_lower)
        base_score = trigger_matches / max(len(triggers), 1)

        # Boost from intent matching
        patterns = self.selector_config.get("task_classification", {}).get("intent_patterns", {})
        intent_boost = 0.0
        for intent, config in patterns.items():
            if prompt_id in config.get("prompts", []):
                intent_boost += intent_scores.get(intent, 0) * 0.5

        # Apply learned adjustments
        stats = self.prompt_stats.get(prompt_id, {})
        success_rate = stats.get("success_rate", 0.5)
        learned_adjustment = (success_rate - 0.5) * 0.3  # +/- 0.15 max

        # Final score
        final_score = min(1.0, base_score + intent_boost + learned_adjustment)

        return final_score

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        return len(text) // self.CHARS_PER_TOKEN

    def _select_tier(self, query: str, primary_intent: str, estimated_tokens: int) -> ModelTier:
        """Select the appropriate model tier based on task complexity."""
        query_lower = query.lower()

        # Check tiering config
        tiers = self.tiering_config.get("tiers", {})

        # Check lightweight triggers
        lightweight = tiers.get("lightweight", {})
        lightweight_patterns = lightweight.get("triggers", {}).get("query_patterns", [])
        if any(p in query_lower for p in lightweight_patterns):
            if estimated_tokens < 2000:
                return ModelTier.LIGHTWEIGHT

        # Check heavyweight triggers
        heavyweight = tiers.get("heavyweight", {})
        heavyweight_patterns = heavyweight.get("triggers", {}).get("query_patterns", [])
        heavyweight_tasks = heavyweight.get("triggers", {}).get("task_types", [])

        if any(p in query_lower for p in heavyweight_patterns):
            return ModelTier.HEAVYWEIGHT
        if primary_intent in heavyweight_tasks:
            return ModelTier.HEAVYWEIGHT
        if estimated_tokens > 12000:
            return ModelTier.HEAVYWEIGHT

        # Check standard triggers
        standard = tiers.get("standard", {})
        standard_tasks = standard.get("triggers", {}).get("task_types", [])

        if primary_intent in standard_tasks:
            return ModelTier.STANDARD
        if 2000 <= estimated_tokens <= 12000:
            return ModelTier.STANDARD

        # Default to standard
        return ModelTier.STANDARD

    def _determine_composition_strategy(
        self,
        primary: str,
        secondary: list[str],
        query: str
    ) -> CompositionStrategy:
        """Determine how to compose multiple prompts."""
        # Keywords suggesting sequential execution
        sequential_keywords = ["then", "after", "followed by", "step by step", "phases"]
        if any(kw in query.lower() for kw in sequential_keywords):
            return CompositionStrategy.SEQUENTIAL

        # If security is involved, use hierarchical with security as advisor
        if "security-auditor" in secondary and primary != "security-auditor":
            return CompositionStrategy.HIERARCHICAL

        # Default to parallel for independent concerns
        return CompositionStrategy.PARALLEL

    def _compose_guidance(
        self,
        primary: str,
        secondary: list[str],
        strategy: CompositionStrategy
    ) -> str:
        """Compose guidance from selected prompts."""
        lines = []

        # Add primary prompt guidance
        primary_prompt = self.prompts.get(primary, {})
        primary_identity = primary_prompt.get("identity", {})
        lines.append(f"## Primary Mode: {primary_identity.get('role', primary)}")
        lines.append(f"Philosophy: {primary_identity.get('philosophy', '')}")
        lines.append(f"North Star: {primary_identity.get('north_star', '')}")
        lines.append("")

        # Add key principles from primary
        laws = primary_prompt.get("laws", [])
        if laws:
            lines.append("### Active Principles:")
            for law in laws[:3]:  # Top 3 laws
                if isinstance(law, dict):
                    lines.append(f"- **{law.get('name', '')}**: {law.get('principle', '')}")
        lines.append("")

        # Add secondary prompt considerations
        if secondary:
            lines.append(f"### Secondary Considerations ({strategy.value} mode):")
            for sec_id in secondary:
                sec_prompt = self.prompts.get(sec_id, {})
                sec_identity = sec_prompt.get("identity", {})
                lines.append(f"- {sec_identity.get('role', sec_id)}: {sec_identity.get('north_star', '')}")
        lines.append("")

        # Add thinking phases if available
        phases = primary_prompt.get("thinking_phases", [])
        if phases:
            lines.append("### Execution Phases:")
            for phase in phases:
                if isinstance(phase, dict):
                    lines.append(f"1. **{phase.get('phase', '')}**: {phase.get('question', '')}")

        return "\n".join(lines)

    def select(self, query: str, context: Optional[dict] = None) -> PromptSelection:
        """
        Select and compose prompts for a given task.

        Args:
            query: The user's task description
            context: Optional additional context

        Returns:
            PromptSelection with selected prompts and composed guidance
        """
        # Extract intent
        primary_intent, intent_scores = self._extract_intent(query)

        # Estimate tokens from query and context
        estimated_tokens = self._estimate_tokens(query)
        if context:
            context_str = json.dumps(context) if isinstance(context, dict) else str(context)
            estimated_tokens += self._estimate_tokens(context_str)

        # Score all prompts
        prompt_scores: dict[str, float] = {}
        for prompt_id in self.prompts:
            score = self._calculate_prompt_score(prompt_id, query, intent_scores)
            prompt_scores[prompt_id] = score

        # Select primary (highest score above threshold)
        sorted_prompts = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_prompts[0][0] if sorted_prompts and sorted_prompts[0][1] >= self.PRIMARY_THRESHOLD else "codebase-sentinel"
        primary_score = prompt_scores.get(primary, 0)

        # Select secondary (above secondary threshold, excluding primary)
        secondary = [
            pid for pid, score in sorted_prompts[1:4]  # Top 3 after primary
            if score >= self.SECONDARY_THRESHOLD and pid != primary
        ]

        # Determine composition strategy
        strategy = self._determine_composition_strategy(primary, secondary, query)

        # Compose guidance
        composed = self._compose_guidance(primary, secondary, strategy)

        # Select model tier based on complexity
        tier = self._select_tier(query, primary_intent, estimated_tokens)

        # Build reasoning
        reasoning_parts = [
            f"Primary intent: {primary_intent}",
            f"Scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted_prompts[:3])}",
            f"Strategy: {strategy.value}",
            f"Tier: {tier.value}",
            f"Est. tokens: {estimated_tokens}"
        ]

        return PromptSelection(
            primary=primary,
            secondary=secondary,
            strategy=strategy,
            confidence=primary_score,
            reasoning=" | ".join(reasoning_parts),
            composed_guidance=composed,
            recommended_tier=tier.value,
            estimated_tokens=estimated_tokens
        )

    def record_outcome(
        self,
        query: str,
        selected_prompts: list[str],
        outcome: TaskOutcome,
        duration_seconds: Optional[int] = None,
        feedback: Optional[str] = None
    ):
        """
        Record task outcome for learning.

        Args:
            query: Original task query
            selected_prompts: Prompts that were selected
            outcome: The task outcome
            duration_seconds: Time taken
            feedback: Optional user feedback
        """
        # Create task record
        record = TaskRecord(
            task_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            query=query,
            extracted_intent=self._extract_intent(query)[0],
            selected_prompts=selected_prompts,
            composition_strategy="recorded",
            confidence=0.0,
            outcome=outcome.value,
            duration_seconds=duration_seconds,
            user_feedback=feedback
        )

        # Append to task history
        history_path = self.learning_dir / "task-history.jsonl"
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record)) + "\n")

        # Update prompt stats
        for prompt_id in selected_prompts:
            if prompt_id not in self.prompt_stats:
                self.prompt_stats[prompt_id] = {
                    "total_selections": 0,
                    "success_count": 0,
                    "partial_count": 0,
                    "failure_count": 0,
                    "success_rate": 0.5
                }

            stats = self.prompt_stats[prompt_id]
            stats["total_selections"] += 1

            if outcome == TaskOutcome.SUCCESS:
                stats["success_count"] += 1
            elif outcome == TaskOutcome.PARTIAL:
                stats["partial_count"] += 1
            else:
                stats["failure_count"] += 1

            # Recalculate success rate with decay
            total = stats["success_count"] + stats["partial_count"] * 0.5 + stats["failure_count"] * 0
            stats["success_rate"] = total / stats["total_selections"]

        self._save_prompt_stats()

    def get_effectiveness_report(self) -> dict:
        """Generate an effectiveness report for all prompts."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "prompts": {}
        }

        for prompt_id, stats in self.prompt_stats.items():
            report["prompts"][prompt_id] = {
                "total_uses": stats.get("total_selections", 0),
                "success_rate": f"{stats.get('success_rate', 0) * 100:.1f}%",
                "successes": stats.get("success_count", 0),
                "partials": stats.get("partial_count", 0),
                "failures": stats.get("failure_count", 0)
            }

        return report

    def execute_sentinel_audit(self) -> dict:
        """Execute a Codebase Sentinel audit on the repository."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "laws_checked": [],
            "findings": [],
            "summary": {}
        }

        sentinel = self.prompts.get("codebase-sentinel", {})
        laws = sentinel.get("laws", [])

        for law in laws:
            if not isinstance(law, dict):
                continue

            law_id = law.get("id", "unknown")
            law_name = law.get("name", "Unknown")
            detection = law.get("detection", {})

            law_result = {
                "id": law_id,
                "name": law_name,
                "checked": True,
                "violations": []
            }

            # Check patterns if defined
            patterns = detection.get("patterns", [])
            for pattern in patterns:
                # This would scan files - simplified for now
                law_result["violations"].append({
                    "pattern": pattern,
                    "status": "requires_scan"
                })

            results["laws_checked"].append(law_result)

        # Summary
        results["summary"] = {
            "total_laws": len(laws),
            "laws_checked": len(results["laws_checked"]),
            "requires_manual_review": True
        }

        return results


def main():
    """CLI interface for the prompt engine."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Prompt Engine")
    subparsers = parser.add_subparsers(dest="command")

    # Select command
    select_parser = subparsers.add_parser("select", help="Select prompts for a task")
    select_parser.add_argument("query", help="Task description")

    # Report command
    subparsers.add_parser("report", help="Generate effectiveness report")

    # Audit command
    subparsers.add_parser("audit", help="Run Codebase Sentinel audit")

    # List command
    subparsers.add_parser("list", help="List available prompts")

    args = parser.parse_args()

    engine = PromptEngine()

    if args.command == "select":
        selection = engine.select(args.query)
        print(f"\n{'='*60}")
        print(f"PROMPT SELECTION + MODEL TIERING")
        print(f"{'='*60}")
        print(f"Primary: {selection.primary}")
        print(f"Secondary: {', '.join(selection.secondary) or 'None'}")
        print(f"Strategy: {selection.strategy.value}")
        print(f"Confidence: {selection.confidence:.2f}")
        print(f"\n--- Token Optimization ---")
        print(f"Recommended Tier: {selection.recommended_tier.upper()}")
        print(f"Estimated Tokens: {selection.estimated_tokens:,}")
        print(f"Reasoning: {selection.reasoning}")
        print(f"\n{'='*60}")
        print("COMPOSED GUIDANCE")
        print(f"{'='*60}")
        print(selection.composed_guidance)

    elif args.command == "report":
        report = engine.get_effectiveness_report()
        print(json.dumps(report, indent=2))

    elif args.command == "audit":
        results = engine.execute_sentinel_audit()
        print(json.dumps(results, indent=2))

    elif args.command == "list":
        print("\nAvailable Superprompts:")
        print("-" * 40)
        for prompt_id, prompt in engine.prompts.items():
            meta = prompt.get("metadata", {})
            identity = prompt.get("identity", {})
            print(f"\n  {prompt_id}:")
            print(f"    Role: {identity.get('role', 'N/A')}")
            print(f"    Purpose: {meta.get('purpose', 'N/A')}")
            print(f"    Triggers: {', '.join(meta.get('triggers', []))}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
