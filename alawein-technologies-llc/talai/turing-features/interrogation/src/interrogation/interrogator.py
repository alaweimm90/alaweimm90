"""
Interrogator

Asks questions and gets answers from AI models.
Supports single-model and multi-model consensus approaches.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from interrogation.core.models import Answer, ConsensusAnswer


class Interrogator:
    """
    Ask questions and get answers from AI models

    Supports:
    - Single-model answering
    - Multi-model consensus
    - Adaptive question selection
    """

    def __init__(self, orchestrator=None, **config):
        """
        Initialize interrogator

        Args:
            orchestrator: AI Orchestrator instance for LLM calls
            **config: Configuration options
                - use_consensus: Use multiple models (default: False)
                - consensus_models: List of model IDs (default: ['claude-sonnet-4', 'gpt-4', 'gemini-pro'])
                - parallel: Ask questions in parallel (default: True)
                - max_concurrent: Max concurrent questions (default: 5)
        """
        self.orchestrator = orchestrator
        self.config = config

        # Default configuration
        self.use_consensus = config.get("use_consensus", False)
        self.consensus_models = config.get(
            "consensus_models",
            ["claude-sonnet-4", "gpt-4", "gemini-pro"]
        )
        self.parallel = config.get("parallel", True)
        self.max_concurrent = config.get("max_concurrent", 5)

    async def ask_question(
        self,
        question: str,
        hypothesis: Any,
        context: Optional[str] = None,
        model: Optional[str] = None
    ) -> Answer:
        """
        Ask a single question about the hypothesis

        Args:
            question: The question to ask
            hypothesis: The hypothesis being interrogated
            context: Additional context
            model: Specific model to use (None = use orchestrator's router)

        Returns:
            Answer object
        """
        if not self.orchestrator:
            # Fallback: Return placeholder answer
            return Answer(
                question=question,
                answer="[No orchestrator available - placeholder answer]",
                model="none",
                confidence=0.5,
                tokens_used=0,
            )

        # Build prompt
        prompt = self._build_question_prompt(question, hypothesis, context)

        # Execute via orchestrator
        from atlas_orchestrator import Task, TaskType

        task = Task(
            prompt=prompt,
            task_type=TaskType.ANALYSIS,
            max_tokens=500,
        )

        if model:
            # Force specific model
            result = await self.orchestrator.execute_with_model(task, model)
        else:
            # Let router decide
            result = await self.orchestrator.execute(task)

        # Extract answer and confidence
        answer_text = result.content if result.success else "[Error generating answer]"
        confidence = self._estimate_confidence(answer_text)

        return Answer(
            question=question,
            answer=answer_text,
            model=result.metadata.get("model", "unknown"),
            confidence=confidence,
            tokens_used=result.metadata.get("tokens_used", 0),
        )

    async def ask_with_consensus(
        self,
        question: str,
        hypothesis: Any,
        context: Optional[str] = None
    ) -> ConsensusAnswer:
        """
        Ask question using multiple models and build consensus

        Args:
            question: The question to ask
            hypothesis: The hypothesis being interrogated
            context: Additional context

        Returns:
            ConsensusAnswer with synthesis
        """
        # Ask all models
        tasks = [
            self.ask_question(question, hypothesis, context, model=model)
            for model in self.consensus_models
        ]

        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_answers = [
            a for a in answers
            if isinstance(a, Answer)
        ]

        if not valid_answers:
            # Fallback if all failed
            return ConsensusAnswer(
                question=question,
                answers=[],
                consensus_answer="[All models failed to answer]",
                agreement_score=0.0,
                final_confidence=0.0,
            )

        # Synthesize consensus
        consensus_text, agreement = self._synthesize_consensus(valid_answers)

        return ConsensusAnswer(
            question=question,
            answers=valid_answers,
            consensus_answer=consensus_text,
            agreement_score=agreement,
            final_confidence=sum(a.confidence for a in valid_answers) / len(valid_answers),
        )

    async def ask_questions(
        self,
        questions: List[str],
        hypothesis: Any,
        context: Optional[str] = None
    ) -> List[Answer]:
        """
        Ask multiple questions

        Args:
            questions: List of questions
            hypothesis: The hypothesis being interrogated
            context: Additional context

        Returns:
            List of Answer objects
        """
        if self.use_consensus:
            # Use consensus for all questions
            consensus_answers = await self._ask_questions_with_consensus(
                questions, hypothesis, context
            )
            # Convert to regular answers (take best from consensus)
            return [ca.get_best_answer() for ca in consensus_answers]

        if self.parallel:
            # Ask in parallel with concurrency limit
            return await self._ask_questions_parallel(questions, hypothesis, context)
        else:
            # Ask sequentially
            return await self._ask_questions_sequential(questions, hypothesis, context)

    async def _ask_questions_parallel(
        self,
        questions: List[str],
        hypothesis: Any,
        context: Optional[str] = None
    ) -> List[Answer]:
        """Ask questions in parallel with concurrency control"""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def ask_with_limit(q):
            async with semaphore:
                return await self.ask_question(q, hypothesis, context)

        tasks = [ask_with_limit(q) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and return valid answers
        return [
            a if isinstance(a, Answer) else Answer(
                question=questions[i],
                answer="[Error]",
                model="error",
                confidence=0.0,
                tokens_used=0,
            )
            for i, a in enumerate(answers)
        ]

    async def _ask_questions_sequential(
        self,
        questions: List[str],
        hypothesis: Any,
        context: Optional[str] = None
    ) -> List[Answer]:
        """Ask questions one by one"""
        answers = []
        for question in questions:
            answer = await self.ask_question(question, hypothesis, context)
            answers.append(answer)
        return answers

    async def _ask_questions_with_consensus(
        self,
        questions: List[str],
        hypothesis: Any,
        context: Optional[str] = None
    ) -> List[ConsensusAnswer]:
        """Ask all questions using consensus"""
        tasks = [
            self.ask_with_consensus(q, hypothesis, context)
            for q in questions
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _build_question_prompt(
        self,
        question: str,
        hypothesis: Any,
        context: Optional[str] = None
    ) -> str:
        """Build prompt for asking question"""
        # Extract hypothesis details
        if hasattr(hypothesis, 'claim'):
            claim = hypothesis.claim
            domain = getattr(hypothesis, 'domain', 'unknown')
            hyp_context = getattr(hypothesis, 'context', '')
        else:
            claim = str(hypothesis)
            domain = 'unknown'
            hyp_context = ''

        prompt = f"""You are evaluating a scientific hypothesis. Answer the following question critically and objectively.

Hypothesis: {claim}
Domain: {domain}
Context: {hyp_context or 'None provided'}
{f'Additional Context: {context}' if context else ''}

Question: {question}

Provide a clear, concise answer focusing on:
1. Direct answer to the question
2. Evidence or reasoning
3. Any concerns or limitations

Be critical but fair. If information is missing, state what's needed.
Keep your answer under 300 words.
"""
        return prompt

    def _estimate_confidence(self, answer_text: str) -> float:
        """
        Estimate confidence from answer text

        Uses heuristics based on language markers.
        """
        answer_lower = answer_text.lower()

        # High confidence markers
        high_conf = ["clearly", "definitely", "certainly", "undoubtedly", "proven", "confirmed"]
        # Medium confidence markers
        med_conf = ["likely", "probably", "appears", "seems", "suggests", "indicates"]
        # Low confidence markers
        low_conf = ["unclear", "uncertain", "possibly", "might", "maybe", "insufficient", "unknown"]

        high_count = sum(1 for marker in high_conf if marker in answer_lower)
        med_count = sum(1 for marker in med_conf if marker in answer_lower)
        low_count = sum(1 for marker in low_conf if marker in answer_lower)

        if low_count > high_count + med_count:
            return 0.3
        elif high_count > med_count + low_count:
            return 0.9
        elif med_count > high_count:
            return 0.6
        else:
            return 0.7  # Default

    def _synthesize_consensus(self, answers: List[Answer]) -> tuple[str, float]:
        """
        Synthesize consensus from multiple answers

        Returns:
            (consensus_text, agreement_score)
        """
        if not answers:
            return ("[No answers]", 0.0)

        if len(answers) == 1:
            return (answers[0].answer, 1.0)

        # Simple synthesis: Use highest confidence answer as base
        best_answer = max(answers, key=lambda a: a.confidence)

        # Calculate agreement based on answer similarity (simplified)
        # In production, use semantic similarity
        agreement = sum(a.confidence for a in answers) / len(answers)

        # Synthesize
        if agreement > 0.7:
            consensus = f"{best_answer.answer}\n\n(High agreement across {len(answers)} models)"
        elif agreement > 0.5:
            consensus = f"{best_answer.answer}\n\n(Moderate agreement across {len(answers)} models)"
        else:
            consensus = f"{best_answer.answer}\n\n(Low agreement - models diverge on this question)"

        return (consensus, agreement)
