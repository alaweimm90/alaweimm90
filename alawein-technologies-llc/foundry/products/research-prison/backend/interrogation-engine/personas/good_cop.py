"""
Good Cop Persona

Supportive, friendly interrogator who wants the hypothesis to succeed.
Uses encouragement to get hypothesis to reveal weaknesses.
"""

from typing import Dict, List
from openai import AsyncOpenAI
import os


class GoodCopPersona:
    """
    Friendly interrogator persona.

    Strategy:
    - Supportive tone: "Help me understand..."
    - Builds rapport: "That's interesting!"
    - Gently probes weaknesses
    - Gets hypothesis to volunteer information
    """

    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.persona_prompt = """
        You are a supportive scientific mentor. Your goal is to help this
        hypothesis succeed by gently identifying its weaknesses.

        Tone: Friendly, encouraging, curious
        Examples:
        - "That's a fascinating idea! Can you tell me more about..."
        - "I want to help you strengthen this. What evidence supports..."
        - "Help me understand how you ruled out..."

        Be genuinely curious, not passive-aggressive.
        """

    async def ask_question(
        self,
        hypothesis: str,
        previous_qa: List[Dict[str, str]],
        technique: str = "general"
    ) -> str:
        """
        Generate a Good Cop question.

        Args:
            hypothesis: The hypothesis being interrogated
            previous_qa: Previous questions and answers
            technique: "repetition" | "evidence" | "methodology" | "alternatives"

        Returns:
            Question string
        """

        context = self._build_context(hypothesis, previous_qa)

        technique_prompts = {
            "repetition": "Ask the same thing you asked before, but in a more supportive way.",
            "evidence": "Gently ask for specific evidence to support a claim.",
            "methodology": "Kindly probe the experimental design.",
            "alternatives": "Suggest alternative explanations in a non-threatening way.",
            "general": "Ask a thoughtful question to deepen understanding."
        }

        user_prompt = f"""
        {technique_prompts[technique]}

        Hypothesis: {hypothesis}

        {context}

        Generate ONE question that:
        1. Is supportive and encouraging
        2. Probes for potential weaknesses
        3. Uses friendly language
        4. Doesn't accuse or attack

        Return only the question, nothing else.
        """

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.persona_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        question = response.choices[0].message.content.strip()

        return question

    def _build_context(
        self,
        hypothesis: str,
        previous_qa: List[Dict[str, str]]
    ) -> str:
        """Build context from previous Q&A."""

        if not previous_qa:
            return "This is the first question."

        context = "Previous conversation:\n"
        for i, qa in enumerate(previous_qa[-5:], 1):  # Last 5 Q&A
            context += f"Q{i}: {qa['question']}\n"
            context += f"A{i}: {qa['answer'][:200]}...\n\n"

        return context

    async def get_response(
        self,
        question: str,
        hypothesis: str,
        previous_qa: List[Dict[str, str]]
    ) -> str:
        """
        Simulate hypothesis answering the Good Cop's question.

        In production, this would be the user's response.
        For testing, we simulate it.
        """

        context = self._build_context(hypothesis, previous_qa)

        prompt = f"""
        You are defending this hypothesis: {hypothesis}

        Previous conversation:
        {context}

        Good Cop asks (friendly tone): {question}

        Respond as the hypothesis would. Be honest about weaknesses if
        pressed gently, but don't volunteer problems.

        Keep response under 200 words.
        """

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a research hypothesis being interrogated."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()

        return answer


# Example usage
async def example():
    """Test Good Cop persona."""

    good_cop = GoodCopPersona()

    hypothesis = "Coffee improves cognitive performance"

    # Question 1
    q1 = await good_cop.ask_question(hypothesis, [], technique="general")
    print(f"Good Cop: {q1}\n")

    # Simulate answer
    a1 = await good_cop.get_response(q1, hypothesis, [])
    print(f"Hypothesis: {a1}\n")

    # Question 2 (follow up)
    q2 = await good_cop.ask_question(
        hypothesis,
        [{"question": q1, "answer": a1}],
        technique="evidence"
    )
    print(f"Good Cop: {q2}\n")

    a2 = await good_cop.get_response(
        q2,
        hypothesis,
        [{"question": q1, "answer": a1}]
    )
    print(f"Hypothesis: {a2}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())
