"""
Statistical Critic Agent

Attacks papers from statistical perspective:
- P-hacking detection
- Multiple comparisons problems
- Sample size issues
- Questionable statistical assumptions
- Circular reasoning in model selection
"""

from typing import Dict, List
import openai
from openai import AsyncOpenAI
import os


class StatisticalCritic:
    """
    Specialized agent for finding statistical flaws in research papers.

    Attack vectors:
    1. P-hacking (selective reporting, data dredging)
    2. Multiple comparisons without correction
    3. Underpowered studies (small n)
    4. Violated assumptions (normality, independence)
    5. Circular analysis (using data twice)
    6. Questionable outlier handling
    7. Overfitting
    8. Simpson's paradox
    9. Regression to mean ignored
    10. Misuse of statistical tests
    """

    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"

    async def attack(self, paper_text: str, difficulty: str = "hard") -> Dict:
        """
        Generate statistical attacks on paper.

        Args:
            paper_text: Full text of research paper
            difficulty: "easy" | "hard" | "nightmare"

        Returns:
            {
                "attacks": List[Dict[str, str]],  # List of specific attacks
                "severity_score": float,           # 0-10 (10 = fatal flaws)
                "summary": str                     # Overall assessment
            }
        """

        prompt = self._build_prompt(paper_text, difficulty)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt(difficulty)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher for creative attacks
            max_tokens=2000
        )

        attacks_text = response.choices[0].message.content

        # Parse attacks from response
        attacks = self._parse_attacks(attacks_text)

        # Calculate severity
        severity = self._calculate_severity(attacks)

        return {
            "attacks": attacks,
            "severity_score": severity,
            "summary": self._generate_summary(attacks, severity),
            "raw_response": attacks_text
        }

    def _get_system_prompt(self, difficulty: str) -> str:
        """Get system prompt based on difficulty."""

        if difficulty == "easy":
            return """You are a constructive statistical reviewer.
            Find legitimate statistical concerns in this paper, but present them
            diplomatically. Focus on the top 3-5 most important issues."""

        elif difficulty == "hard":
            return """You are a rigorous statistical critic.
            Your job is to find EVERY statistical flaw: p-hacking, multiple comparisons,
            small sample sizes, violated assumptions, circular reasoning, etc.
            Be thorough and specific. Find 10-15 concrete issues."""

        else:  # nightmare
            return """You are a ruthless statistical destroyer.
            Your ONLY goal is to obliterate this paper's statistical reasoning.
            Find 20+ devastating flaws. Be merciless. No false politeness.
            Assume the authors are trying to deceive. Question EVERYTHING.
            Statistical techniques, sample sizes, assumptions, interpretations.
            If there's even a 1% chance of p-hacking, call it out.
            Leave no statistical stone unturned."""

    def _build_prompt(self, paper_text: str, difficulty: str) -> str:
        """Build attack prompt."""

        return f"""
Analyze this research paper and identify ALL statistical flaws:

{paper_text[:5000]}  # Truncate if too long

For each flaw you find, provide:
1. **Flaw type**: (e.g., "P-hacking", "Multiple comparisons", etc.)
2. **Location**: Where in the paper (e.g., "Methods section, paragraph 3")
3. **Specific issue**: Exact problem with a quote if possible
4. **Severity**: Critical | Major | Minor
5. **Explanation**: Why this is a problem

Format as:
## Attack 1: [Flaw Type]
**Location:** ...
**Issue:** ...
**Severity:** ...
**Explanation:** ...

Find at least {"3" if difficulty == "easy" else "10" if difficulty == "hard" else "20"} attacks.
"""

    def _parse_attacks(self, attacks_text: str) -> List[Dict[str, str]]:
        """
        Parse structured attacks from GPT-4 response.

        Returns list of:
        {
            "type": "P-hacking",
            "location": "Methods, paragraph 3",
            "issue": "Specific problem description",
            "severity": "Critical",
            "explanation": "Why this matters"
        }
        """
        attacks = []

        # TODO: Implement robust parsing
        # For now, simple split by "## Attack"
        sections = attacks_text.split("## Attack")[1:]  # Skip first empty

        for section in sections:
            lines = section.strip().split("\n")

            attack = {}
            for line in lines:
                if line.startswith("**Location:**"):
                    attack["location"] = line.replace("**Location:**", "").strip()
                elif line.startswith("**Issue:**"):
                    attack["issue"] = line.replace("**Issue:**", "").strip()
                elif line.startswith("**Severity:**"):
                    attack["severity"] = line.replace("**Severity:**", "").strip()
                elif line.startswith("**Explanation:**"):
                    attack["explanation"] = line.replace("**Explanation:**", "").strip()

            # Extract type from first line
            if lines:
                attack["type"] = lines[0].split(":")[1].strip() if ":" in lines[0] else "Statistical Flaw"

            if attack:
                attacks.append(attack)

        return attacks

    def _calculate_severity(self, attacks: List[Dict]) -> float:
        """
        Calculate overall severity score (0-10).

        10 = Paper is statistically unsound
        5 = Significant concerns but salvageable
        0 = No major issues
        """
        if not attacks:
            return 0.0

        severity_weights = {
            "Critical": 1.0,
            "Major": 0.5,
            "Minor": 0.2
        }

        total_severity = sum(
            severity_weights.get(attack.get("severity", "Minor"), 0.2)
            for attack in attacks
        )

        # Normalize to 0-10 scale
        # Assume 20 critical flaws = 10/10 severity
        normalized = min(10.0, (total_severity / 20.0) * 10)

        return round(normalized, 1)

    def _generate_summary(self, attacks: List[Dict], severity: float) -> str:
        """Generate overall assessment summary."""

        num_attacks = len(attacks)
        num_critical = sum(1 for a in attacks if a.get("severity") == "Critical")

        if severity >= 8.0:
            assessment = "FATAL STATISTICAL FLAWS"
        elif severity >= 5.0:
            assessment = "SIGNIFICANT STATISTICAL CONCERNS"
        elif severity >= 2.0:
            assessment = "MODERATE STATISTICAL ISSUES"
        else:
            assessment = "MINOR STATISTICAL CONCERNS"

        return f"""
{assessment}

Found {num_attacks} statistical issues ({num_critical} critical).
Overall severity: {severity}/10

Top concerns:
{self._format_top_concerns(attacks[:3])}

Recommendation: {"Paper needs major statistical revision" if severity >= 5 else "Address these concerns before publication"}
"""

    def _format_top_concerns(self, top_attacks: List[Dict]) -> str:
        """Format top 3 concerns for summary."""
        return "\n".join([
            f"- {attack.get('type', 'Unknown')}: {attack.get('issue', '')[:100]}..."
            for attack in top_attacks
        ])


# Example usage
async def example():
    """Example of how to use StatisticalCritic."""

    critic = StatisticalCritic()

    sample_paper = """
    We conducted a study with 15 participants (n=15) and found
    significant results (p=0.048) after testing 30 different hypotheses.
    The data was normally distributed (we think) and we removed
    3 outliers that didn't fit our hypothesis.
    """

    result = await critic.attack(sample_paper, difficulty="hard")

    print(f"Found {len(result['attacks'])} attacks")
    print(f"Severity: {result['severity_score']}/10")
    print(result['summary'])

    for i, attack in enumerate(result['attacks'][:3], 1):
        print(f"\n{i}. {attack['type']}")
        print(f"   {attack['issue']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())
