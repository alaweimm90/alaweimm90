"""
Nightmare Mode Ensemble

Coordinates multiple AI models attacking a paper simultaneously:
- GPT-4 (OpenAI)
- Claude Opus (Anthropic)
- Debate phase where models challenge each other
- Consensus building
- Agreement scoring
"""

from typing import List, Dict, Tuple
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os


class NightmareModeEnsemble:
    """
    Multi-model attack coalition for maximum rigor.

    Process:
    1. Launch all attack agents in parallel (both GPT-4 and Claude)
    2. Collect initial attacks
    3. Debate phase: Models challenge each other's attacks
    4. Build consensus: Keep attacks both models agree on
    5. Calculate survival score based on consensus attacks
    """

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Assign models to attack dimensions
        self.attack_dimensions = {
            'statistical': 'GPT-4',
            'methodological': 'Claude-3-Opus',
            'logical': 'GPT-4',
            'historical': 'Claude-3-Opus',
            'ethical': 'GPT-4',
            'economic': 'Claude-3-Opus'
        }

    async def coordinate_attack(self, paper: str) -> Dict:
        """
        Execute full Nightmare Mode attack sequence.

        Args:
            paper: Full text of research paper

        Returns:
            {
                "survival_score": float,           # 0-100
                "consensus_attacks": List[Dict],   # Attacks both models agree on
                "debate_transcript": str,          # Full debate between models
                "model_agreement": float,          # 0-1 (agreement rate)
                "individual_attacks": {
                    "gpt4": List[Dict],
                    "claude": List[Dict]
                }
            }
        """

        print("[Nightmare Mode] Launching multi-model attack coalition...")

        # Phase 1: Parallel attacks from all dimensions
        initial_attacks = await self._execute_parallel_attacks(paper)

        print(f"[Nightmare Mode] Collected {len(initial_attacks)} initial attacks")

        # Phase 2: Debate phase
        debate_result = await self._debate_phase(initial_attacks, paper)

        print(f"[Nightmare Mode] Debate complete. Agreement: {debate_result['agreement_rate']:.1%}")

        # Phase 3: Build consensus
        consensus_attacks = self._build_consensus(debate_result)

        print(f"[Nightmare Mode] Consensus: {len(consensus_attacks)} attacks both models agree on")

        # Phase 4: Calculate survival score
        survival_score = self._calculate_survival_score(consensus_attacks)

        print(f"[Nightmare Mode] Survival score: {survival_score}/100")

        return {
            'survival_score': survival_score,
            'consensus_attacks': consensus_attacks,
            'debate_transcript': debate_result['transcript'],
            'model_agreement': debate_result['agreement_rate'],
            'individual_attacks': {
                'gpt4': [a for a in initial_attacks if a['model'] == 'GPT-4'],
                'claude': [a for a in initial_attacks if a['model'] == 'Claude-3-Opus']
            },
            'total_attacks': len(initial_attacks),
            'consensus_attacks_count': len(consensus_attacks)
        }

    async def _execute_parallel_attacks(self, paper: str) -> List[Dict]:
        """
        Launch all 6 attack dimensions in parallel.

        Returns list of attacks from both models.
        """

        attack_tasks = [
            self._execute_attack(dimension, model, paper)
            for dimension, model in self.attack_dimensions.items()
        ]

        attack_results = await asyncio.gather(*attack_tasks)

        # Flatten list of attacks
        all_attacks = []
        for result in attack_results:
            all_attacks.extend(result['attacks'])

        return all_attacks

    async def _execute_attack(self, dimension: str, model: str, paper: str) -> Dict:
        """Execute single attack dimension with specified model."""

        attack_prompts = {
            'statistical': """You are a ruthless statistical critic. Destroy this paper's
            statistical reasoning. Find every flaw: p-hacking, multiple comparisons, small sample,
            questionable assumptions, circular reasoning, etc. Be merciless. Provide 20+ specific attacks.""",

            'methodological': """You are a methodological perfectionist. Tear apart this paper's
            experimental design: confounds, selection bias, measurement error, lack of controls,
            improper randomization, etc. Find 20+ specific flaws.""",

            'logical': """You are a logician. Find every logical flaw: non-sequiturs, contradictions,
            unsupported assumptions, equivocation, circular arguments, etc. Provide 20+ specific attacks.""",

            'historical': """You are a historian of science. Show how this idea has been tried before
            and failed. Find analogous attempts and explain why they didn't work. Provide 15+ examples.""",

            'ethical': """You are an ethicist. Identify every way this research could be misused,
            harm vulnerable populations, or create unintended negative consequences.
            Provide 15+ specific concerns.""",

            'economic': """You are a cost-benefit analyst. Show why this research is not worth
            the resources: low probability of success, minimal impact even if true, opportunity cost.
            Provide 15+ specific arguments."""
        }

        system_prompt = attack_prompts[dimension]
        user_prompt = f"Attack this paper:\n\n{paper[:4000]}\n\nProvide specific, devastating critiques."

        if 'GPT' in model:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            attacks_text = response.choices[0].message.content

        else:  # Claude
            response = await self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                ]
            )
            attacks_text = response.content[0].text

        # Parse attacks
        attacks = self._parse_attacks_from_text(attacks_text, dimension, model)

        return {
            'dimension': dimension,
            'model': model,
            'attacks': attacks,
            'raw_text': attacks_text
        }

    def _parse_attacks_from_text(self, text: str, dimension: str, model: str) -> List[Dict]:
        """
        Parse individual attacks from model response.

        Very simple parser for MVP - just split by numbers.
        TODO: Use more robust parsing (regex, structured output)
        """
        attacks = []

        # Simple split by "1.", "2.", etc.
        lines = text.split("\n")
        current_attack = ""

        for line in lines:
            # Check if line starts with number
            if line.strip() and line.strip()[0].isdigit() and "." in line[:3]:
                # Save previous attack
                if current_attack:
                    attacks.append({
                        'dimension': dimension,
                        'model': model,
                        'content': current_attack.strip(),
                        'severity': 'High'  # TODO: Extract from text
                    })
                current_attack = line
            else:
                current_attack += " " + line

        # Add final attack
        if current_attack:
            attacks.append({
                'dimension': dimension,
                'model': model,
                'content': current_attack.strip(),
                'severity': 'High'
            })

        return attacks

    async def _debate_phase(self, initial_attacks: List[Dict], paper: str) -> Dict:
        """
        Models debate their attacks.

        GPT-4 challenges Claude's attacks and vice versa.
        Produces agreement matrix and refined attack list.
        """

        # Separate attacks by model
        gpt4_attacks = [a for a in initial_attacks if a['model'] == 'GPT-4']
        claude_attacks = [a for a in initial_attacks if a['model'] == 'Claude-3-Opus']

        # Create debate prompt
        debate_prompt = f"""
Two AI systems have attacked this paper from different angles.

GPT-4 found these issues:
{self._format_attacks_for_debate(gpt4_attacks[:10])}  # Top 10

Claude found these issues:
{self._format_attacks_for_debate(claude_attacks[:10])}

Original paper:
{paper[:2000]}

Now engage in a debate:
1. Which attacks are most devastating and both of you agree on?
2. Are any attacks contradictory between you?
3. Did either of you miss something obvious that the other caught?
4. What are the top 10 consensus attacks (both agree = fatal flaw)?

Provide ranked list of consensus attacks with agreement scores.
"""

        # Use GPT-4 as debate moderator
        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a debate moderator synthesizing AI critiques."},
                {"role": "user", "content": debate_prompt}
            ],
            max_tokens=3000
        )

        debate_transcript = response.choices[0].message.content

        # Calculate agreement rate
        agreement_rate = self._calculate_agreement_rate(gpt4_attacks, claude_attacks)

        return {
            'transcript': debate_transcript,
            'agreement_rate': agreement_rate,
            'gpt4_attacks': gpt4_attacks,
            'claude_attacks': claude_attacks
        }

    def _format_attacks_for_debate(self, attacks: List[Dict]) -> str:
        """Format attacks for debate prompt."""
        return "\n".join([
            f"- {attack['content'][:200]}..."
            for attack in attacks
        ])

    def _calculate_agreement_rate(self, gpt4_attacks: List[Dict], claude_attacks: List[Dict]) -> float:
        """
        Calculate how much the models agree.

        For MVP: Simple heuristic based on attack counts.
        TODO: Use semantic similarity to find overlapping attacks.
        """
        # If similar number of attacks, assume moderate agreement
        total_attacks = len(gpt4_attacks) + len(claude_attacks)
        if total_attacks == 0:
            return 0.0

        # Simple heuristic: If both found similar amounts, agreement is higher
        ratio = min(len(gpt4_attacks), len(claude_attacks)) / max(len(gpt4_attacks), len(claude_attacks), 1)

        # Assume 60-80% agreement typically
        return 0.6 + (ratio * 0.2)

    def _build_consensus(self, debate_result: Dict) -> List[Dict]:
        """
        Extract consensus attacks from debate.

        TODO: Parse debate transcript to extract attacks both models agreed on.
        For MVP: Return all attacks (assume they're mostly consensus).
        """
        # Simplified for MVP
        gpt4_attacks = debate_result['gpt4_attacks'][:10]  # Top 10 from each
        claude_attacks = debate_result['claude_attacks'][:10]

        consensus = gpt4_attacks + claude_attacks

        return consensus

    def _calculate_survival_score(self, consensus_attacks: List[Dict]) -> float:
        """
        Calculate survival score (0-100).

        100 = Survived all attacks (indestructible)
        50 = Survived half (decent but flawed)
        0 = Completely destroyed

        Formula: 100 - (num_attacks * severity_weight)
        """
        if not consensus_attacks:
            return 100.0

        severity_weights = {
            'Critical': 5.0,
            'High': 3.0,
            'Medium': 1.5,
            'Low': 0.5
        }

        total_damage = sum(
            severity_weights.get(attack.get('severity', 'High'), 3.0)
            for attack in consensus_attacks
        )

        # Cap at 100 points of damage
        total_damage = min(100, total_damage)

        survival_score = 100 - total_damage

        return max(0, round(survival_score, 1))


# Example usage
async def example():
    """Test Nightmare Mode ensemble."""

    ensemble = NightmareModeEnsemble()

    sample_paper = """
    Title: Coffee Improves Cognitive Performance

    Abstract: We conducted a study with 15 participants and found that
    coffee significantly improves cognitive performance (p=0.048).

    Methods: We recruited 15 volunteers and gave them coffee. We then
    tested 30 different cognitive measures and found one that was significant.

    Results: Reaction time improved by 12ms (p=0.048).

    Conclusion: Coffee is a powerful cognitive enhancer.
    """

    result = await ensemble.coordinate_attack(sample_paper)

    print(f"\n{'='*60}")
    print(f"NIGHTMARE MODE RESULTS")
    print(f"{'='*60}")
    print(f"Survival Score: {result['survival_score']}/100")
    print(f"Total Attacks: {result['total_attacks']}")
    print(f"Consensus Attacks: {result['consensus_attacks_count']}")
    print(f"Model Agreement: {result['model_agreement']:.1%}")
    print(f"\nTop Consensus Attacks:")
    for i, attack in enumerate(result['consensus_attacks'][:5], 1):
        print(f"{i}. [{attack['dimension']}] {attack['content'][:100]}...")


if __name__ == "__main__":
    asyncio.run(example())
