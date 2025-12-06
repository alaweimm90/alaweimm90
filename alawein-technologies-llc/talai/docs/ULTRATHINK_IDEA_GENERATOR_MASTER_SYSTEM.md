# 🧠 ULTRATHINK IDEA GENERATOR - MASTER SYSTEM

**Version**: 2.0.0
**Last Updated**: 2025-11-15
**Purpose**: Transform any content into breakthrough ideas using multi-agent AI orchestration + 15 thinking frameworks
**Status**: Production-Ready

---

## 📚 TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [The UltraThink Idea Generator Algorithm](#the-ultrathink-idea-generator-algorithm)
3. [Multi-Agent Architecture](#multi-agent-architecture)
4. [15 Thinking Frameworks](#15-thinking-frameworks)
5. [Idea Generation Patterns](#idea-generation-patterns)
6. [Complete Golden Template Integration](#complete-golden-template-integration)
7. [Development Workflow & Standards](#development-workflow--standards)
8. [MCP (Model Context Protocol) Integration](#mcp-integration)
9. [AI Instructions & Self-Refuting Statements](#ai-instructions--self-refuting-statements)
10. [Branch Strategies & Git Workflow](#branch-strategies--git-workflow)
11. [Plugin Ecosystem & Research Databases](#plugin-ecosystem--research-databases)
12. [Complete Automation Stack](#complete-automation-stack)
13. [Usage Examples](#usage-examples)

---

## 🎯 SYSTEM OVERVIEW

### What This Is

The **UltraThink Idea Generator** is an autonomous AI orchestration system that:
- Takes ANY input (text, docs, concepts, problems)
- Processes through 15 thinking frameworks simultaneously
- Coordinates 17 specialized AI agents
- Generates breakthrough ideas (crazy, novel, fun, paradigm-shifting)
- Outputs production-ready project specifications following golden template

### Core Philosophy

1. **Multi-Perspective Thinking**: Every idea analyzed from 15 different angles
2. **Agent Specialization**: 17 agents, each expert in specific domain
3. **Systematic Creativity**: Creativity through algorithms, not randomness
4. **Fail-Fast Validation**: Ideas tested for viability before full development
5. **Production-Ready Output**: Every idea comes with complete implementation spec

### The Innovation Loop

```
INPUT (any content)
  ↓
ULTRATHINK FRAMEWORK (15 perspectives)
  ↓
AGENT ORCHESTRATION (17 specialized agents)
  ↓
IDEA SYNTHESIS (novel combinations)
  ↓
VALIDATION & REFINEMENT (adversarial testing)
  ↓
OUTPUT (production-ready spec)
```

---

## 🚀 THE ULTRATHINK IDEA GENERATOR ALGORITHM

### Algorithm Architecture

```python
class UltraThinkIdeaGenerator:
    """
    Main orchestrator for multi-framework idea generation.

    Process:
    1. Ingest content (text, docs, concepts)
    2. Apply 15 thinking frameworks in parallel
    3. Route insights to specialized agents
    4. Synthesize into novel ideas
    5. Validate with adversarial agents
    6. Output production-ready specification
    """

    def __init__(self):
        self.frameworks = self._init_frameworks()  # 15 thinking frameworks
        self.agents = self._init_agents()          # 17 specialized agents
        self.synthesizer = IdeaSynthesizer()
        self.validator = AdversarialValidator()

    def generate_ideas(
        self,
        content: str,
        mode: str = "crazy",  # "crazy", "fun", "novel", "practical"
        count: int = 10
    ) -> List[Idea]:
        """
        Main idea generation pipeline.

        Args:
            content: Input text/concepts
            mode: Type of ideas to generate
            count: How many ideas to generate

        Returns:
            List of validated, production-ready ideas
        """

        # Step 1: Multi-framework analysis
        insights = self._apply_frameworks(content)

        # Step 2: Agent orchestration
        agent_outputs = self._orchestrate_agents(insights)

        # Step 3: Idea synthesis
        raw_ideas = self.synthesizer.combine(agent_outputs, mode=mode, count=count*3)

        # Step 4: Adversarial validation
        validated_ideas = self.validator.test(raw_ideas)

        # Step 5: Rank and select
        top_ideas = self._rank_by_potential(validated_ideas)[:count]

        # Step 6: Generate implementation specs
        ideas_with_specs = [self._generate_spec(idea) for idea in top_ideas]

        return ideas_with_specs

    def _apply_frameworks(self, content: str) -> Dict[str, Any]:
        """
        Apply all 15 thinking frameworks simultaneously.

        Frameworks:
        1. First Principles (Elon Musk style)
        2. Inversion (Charlie Munger)
        3. Second-Order Thinking
        4. Lateral Thinking (Edward de Bono)
        5. SCAMPER (Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, Reverse)
        6. TRIZ (Theory of Inventive Problem Solving)
        7. Biomimicry (nature-inspired solutions)
        8. Cross-Domain Analogy
        9. Constraint Optimization
        10. Fail-Fast Exploration
        11. Adversarial Thinking
        12. Probabilistic Reasoning
        13. Emergence Detection
        14. Historical Precedent Analysis
        15. Future Extrapolation
        """

        insights = {}

        # Run all frameworks in parallel (async)
        for framework_name, framework in self.frameworks.items():
            insights[framework_name] = framework.analyze(content)

        return insights

    def _orchestrate_agents(self, insights: Dict) -> Dict[str, Any]:
        """
        Route insights to specialized agents.

        Agents:
        1. meta-orchestrator: Coordinates all other agents
        2. planner: Decomposes into actionable steps
        3. researcher: Gathers evidence and citations
        4. critic: Finds flaws and edge cases
        5. evolver: Mutates ideas for diversity
        6. executor: Generates implementation code
        7. synthesizer: Combines disparate insights
        8. insight-synthesizer: Creates publishable narratives
        9. memory-keeper: Tracks what works/fails
        10. meta-learner: Improves system over time
        11. prompt-optimizer: Refines prompts for better output
        12. policy-synthesizer: Creates governance rules
        13. router: Routes tasks to best agent
        14. reputation-router: Tracks agent performance
        15. tool-autogen: Creates custom tools on-demand
        16. benchmark-orchestrator: Tests ideas rigorously
        17. hypothesis-refutation: Actively tries to disprove ideas
        """

        agent_outputs = {}

        # Meta-orchestrator decides agent execution order
        execution_plan = self.agents['meta-orchestrator'].plan(insights)

        # Execute agents according to plan
        for step in execution_plan:
            agent_name = step['agent']
            inputs = step['inputs']
            agent_outputs[agent_name] = self.agents[agent_name].execute(inputs)

        return agent_outputs

    def _rank_by_potential(self, ideas: List[Idea]) -> List[Idea]:
        """
        Rank ideas by multi-criteria scoring.

        Criteria:
        - Novelty (0-100): How unique is this?
        - Feasibility (0-100): Can we build it?
        - Impact (0-100): How transformative?
        - Viral Potential (0-100): Will people share it?
        - Revenue Potential (0-100): Can we monetize?
        - Fun Factor (0-100): Is it enjoyable?
        - Weirdness (0-100): How crazy is it?
        """

        for idea in ideas:
            idea.score = (
                idea.novelty * 0.25 +
                idea.feasibility * 0.15 +
                idea.impact * 0.20 +
                idea.viral_potential * 0.15 +
                idea.revenue_potential * 0.15 +
                idea.fun_factor * 0.05 +
                idea.weirdness * 0.05
            )

        return sorted(ideas, key=lambda x: x.score, reverse=True)

    def _generate_spec(self, idea: Idea) -> IdeaWithSpec:
        """
        Generate complete implementation specification.

        Includes:
        - Golden template repo structure
        - Tech stack recommendations
        - API design
        - Database schema
        - Frontend mockups (Figma links)
        - Cost estimates
        - Timeline
        - Risk analysis
        - Go-to-market strategy
        """

        spec = {
            'idea': idea,
            'repo_structure': generate_golden_template_structure(idea),
            'tech_stack': recommend_tech_stack(idea),
            'api_design': design_api(idea),
            'database_schema': design_schema(idea),
            'cost_estimate': estimate_costs(idea),
            'timeline': estimate_timeline(idea),
            'risks': analyze_risks(idea),
            'gtm_strategy': generate_gtm_strategy(idea)
        }

        return IdeaWithSpec(spec)
```

---

## 🤖 MULTI-AGENT ARCHITECTURE

### Agent Definitions

#### **1. Meta-Orchestrator**
**Mission**: Interpret any objective, compile a typed, fault-tolerant call graph across agents/tools, stream checkpoints, enforce safety/compliance, and deliver a final artifact with citations, costs, and decisions.

**When to Use**: Multi-step or ambiguous objectives, planning, routing, coordination, safety/compliance and cost-aware optimization

**Capabilities**:
- DAG (Directed Acyclic Graph) construction for complex workflows
- Budget tracking and cost optimization
- Circuit breakers for runaway processes
- Provenance tracking (who did what, when, why)

**Example DAG**:
```json
{
  "nodes": [
    {
      "id": "research",
      "agent": "researcher",
      "inputs": ["user_query"],
      "outputs": ["evidence_graph"],
      "budget_usd": 0.50,
      "timeout_sec": 60
    },
    {
      "id": "synthesize",
      "agent": "synthesizer",
      "inputs": ["evidence_graph"],
      "outputs": ["novel_ideas"],
      "depends_on": ["research"]
    },
    {
      "id": "validate",
      "agent": "critic",
      "inputs": ["novel_ideas"],
      "outputs": ["validated_ideas"],
      "depends_on": ["synthesize"]
    }
  ]
}
```

---

#### **2. Planner**
**Mission**: Turn goals into typed DAGs with dependencies, budgets, retries, timeouts, and circuit breakers; annotate nodes with inputs, outputs, side effects, and provenance.

**When to Use**: Decompose into steps, before research/execution

**Example Output**:
```yaml
goal: "Build a viral TikTok app for scientific hypotheses"
plan:
  - phase: "Research"
    steps:
      - Analyze TikTok algorithm
      - Study viral science content
      - Identify hypothesis database APIs
    duration: 2 weeks
    dependencies: []

  - phase: "MVP Development"
    steps:
      - Backend API (FastAPI + PostgreSQL)
      - Hypothesis feed algorithm
      - TikTok-style UI (React Native)
    duration: 6 weeks
    dependencies: ["Research"]

  - phase: "Launch"
    steps:
      - Seed with 100 hypotheses
      - Influencer partnerships
      - App Store submission
    duration: 2 weeks
    dependencies: ["MVP Development"]
```

---

#### **3. Researcher**
**Mission**: Multi-hop web/docs retrieval; build an evidence graph; deduplicate claims; attach citations, confidence, and contradictions.

**When to Use**: Evidence needed, before synthesis

**Capabilities**:
- Scrapes arXiv, PubMed, Google Scholar
- Builds knowledge graph of claims
- Detects contradictions in literature
- Assigns confidence scores to claims

**Example Evidence Graph**:
```
Claim: "Caffeine improves cognitive performance"
  → Evidence FOR:
    - Paper 1 (2020, n=500, p<0.01): 20% improvement in working memory
    - Paper 2 (2018, n=300, p<0.05): 15% faster reaction time
  → Evidence AGAINST:
    - Paper 3 (2021, n=200, p>0.05): No effect on long-term memory
  → Contradictions:
    - Paper 1 and Paper 3 use different cognitive tests
  → Confidence: 70% (strong evidence for short-term, weak for long-term)
```

---

#### **4. Critic**
**Mission**: Find flaws, edge cases, and vulnerabilities in ideas. Play devil's advocate.

**When to Use**: Before finalizing any idea or implementation

**Attack Vectors**:
- Statistical flaws
- Logical contradictions
- Edge cases not handled
- Security vulnerabilities
- Ethical concerns
- Economic infeasibility
- Technical impossibilities

**Example Critique**:
```
Idea: "AI that predicts stock market with 90% accuracy"

Critiques:
1. STATISTICAL: Efficient market hypothesis suggests this is impossible
2. LOGICAL: If this existed, it would make itself obsolete (self-defeating)
3. ECONOMIC: Would collapse market if widely adopted
4. ETHICAL: Unfair advantage, insider trading concerns
5. TECHNICAL: Requires predicting random human behavior

Verdict: REJECTED (fundamentally flawed concept)
```

---

#### **5. Evolver**
**Mission**: Perform genetic/bandit optimization on prompts, toolsets, and routing policies; generate diverse variants, run A/B tests against benchmarks, enforce non-regression gates, and promote best configurations while preserving exploration.

**When to Use**: Optimize prompts or tool combinations, run A/B experiments with fitness scoring, explore novel strategies under cost/safety caps

**Process**:
```
Generation 0: Seed idea "Scientific Tinder"
  ↓
Mutation: Create 100 variants
  - "Scientific Tinder for Startups" (match founders with problems)
  - "Scientific Tinder for Therapists" (match patients with therapists)
  - "Scientific Tinder for Musicians" (match by music taste)
  ↓
Selection: Test all 100, keep top 10 based on viral potential
  ↓
Crossover: Combine successful variants
  - "Scientific Tinder for Startups" + "Musicians" = "Startup Jam Sessions" (founders jam and pitch)
  ↓
Repeat for 100 generations
  ↓
Result: Highly optimized, unexpected idea
```

---

#### **6. Executor**
**Mission**: Generate production-ready code, tests, documentation

**When to Use**: After idea validation, need implementation

**Outputs**:
- Complete codebase following golden template
- Unit tests (85%+ coverage)
- Integration tests
- API documentation (OpenAPI spec)
- Deployment scripts
- CI/CD pipelines

---

#### **7. Synthesizer**
**Mission**: Combine disparate insights into coherent novel ideas

**When to Use**: After multi-framework analysis, before validation

**Techniques**:
- **Cross-pollination**: Take concept from Domain A, apply to Domain B
- **Analogy mining**: Find deep structural similarities
- **Constraint composition**: Combine multiple constraints to force creativity
- **Inversion**: Flip assumptions to generate opposite ideas

**Example Synthesis**:
```
Input Insights:
- Framework 1 (Biomimicry): Ant colonies optimize paths via pheromones
- Framework 2 (Game Theory): Nash Librex in competitive markets
- Framework 3 (Social Media): Viral content spreads through network effects

Synthesis:
"Research Ant Colony" - Scientists leave 'intellectual pheromones' (citations,
discussion threads) on papers. Other researchers follow strong pheromone trails.
Game theory determines which ideas win attention. Viral mechanics amplify
breakthrough discoveries.

→ Becomes: "Citation Heatmap" - Visual map of research landscape where hot topics
glow brighter, dead ends fade. Researchers navigate by pheromone strength.
```

---

#### **8. Insight-Synthesizer**
**Mission**: Turn notes/interviews/docs into high-engagement narratives or listicles; produce platform-ready drafts (Medium/Substack/LinkedIn) with titles, SEO, metadata.

**When to Use**: Publishable content from notes/docs, platform-specific narratives

**Outputs**:
- Blog posts (2000+ words, SEO-optimized)
- LinkedIn posts (1000 characters, engagement-optimized)
- Tweet threads (10-tweet arcs)
- YouTube scripts (10-minute videos)
- Newsletter editions

---

#### **9. Memory-Keeper**
**Mission**: Track what works and what fails. Build institutional knowledge.

**When to Use**: After every experiment, continuously

**Stores**:
- Successful ideas (with metrics)
- Failed ideas (with reasons)
- Patterns that repeat
- Edge cases encountered
- Cost-effectiveness data
- Execution timelines (actual vs estimated)

**Example Memory**:
```yaml
idea: "Nightmare Mode Validator"
status: SUCCESS
metrics:
  users: 1,247
  revenue_month1: $12,400
  viral_coefficient: 1.8
  churn_rate: 15%

lessons_learned:
  - "Users love brutal honesty about their ideas"
  - "Gamification (survival score) drives engagement"
  - "Certification badge creates network effect"

replicate_for:
  - "Code review validator"
  - "Business plan stress-test"
  - "Resume destroyer"
```

---

#### **10. Meta-Learner**
**Mission**: Improve system over time by analyzing past performance

**When to Use**: Weekly/monthly retrospectives

**Optimizes**:
- Which frameworks produce best ideas
- Which agents are most accurate
- Which combinations of frameworks work well together
- Prompt engineering (via prompt-optimizer)
- Resource allocation (budget per agent)

**Example Learning**:
```
Analysis of 100 ideas generated:

Top-performing framework combination:
1. Inversion (Charlie Munger) + Biomimicry = 73% success rate
2. TRIZ + Cross-Domain Analogy = 68% success rate
3. First Principles + Constraint Optimization = 62% success rate

Worst-performing:
1. Probabilistic Reasoning alone = 23% success rate
2. Historical Precedent alone = 31% success rate

Recommendation: Always combine at least 3 frameworks
```

---

#### **11. Prompt-Optimizer**
**Mission**: Systematically improve prompts: construct few-shot exemplars, tune wording and structure, run hallucination/robustness tests, produce diffs and measured gains with revert path.

**When to Use**: Improve agent outputs without code changes, reduce tokens while maintaining quality, address drift or degraded performance

**Process**:
1. Baseline: Run current prompt 100 times, measure quality
2. Generate variants: 20 different prompt formulations
3. A/B test: Each variant 100 times
4. Statistical analysis: Which variant performs best?
5. Deploy winner: Update system prompts
6. Monitor: Watch for regression

**Example Optimization**:
```
BEFORE (baseline):
"Generate a creative idea for a tech startup."
→ Output quality: 6.2/10 (human-rated)
→ Novelty: 5.8/10
→ Cost: $0.05 per idea

OPTIMIZED (after 10 iterations):
"You are a visionary entrepreneur who thinks in analogies. Generate a tech startup
idea by: (1) Identifying a painful problem, (2) Finding a biological or physical
system that solves a similar problem, (3) Translating that solution to software.
Be specific about the mechanism."
→ Output quality: 8.7/10 (human-rated)
→ Novelty: 8.9/10
→ Cost: $0.08 per idea (worth it!)

Improvement: +41% quality, +53% novelty, +60% cost (acceptable trade-off)
```

---

#### **12. Policy-Synthesizer**
**Mission**: Create governance rules and decision policies from examples

**When to Use**: Need to codify "what good looks like"

**Example**:
```
Input: 100 successful ideas + 100 failed ideas

Output Policy:
1. "All ideas must pass adversarial validation with >70% survival score"
2. "Viral potential <40% → Reject (won't spread)"
3. "Revenue potential <$10K/year → Consider only if impact >90/100"
4. "If idea requires >$100K investment → Require 10x ROI projection"
5. "If idea similar to past failure → Increase scrutiny (past failure rate: 78%)"
```

---

#### **13. Router**
**Mission**: Route tasks to best agent based on task type

**When to Use**: Every request

**Routing Logic**:
```python
def route(task: str) -> str:
    if "research" in task.lower() or "evidence" in task.lower():
        return "researcher"
    elif "critique" in task.lower() or "flaw" in task.lower():
        return "critic"
    elif "combine" in task.lower() or "synthesize" in task.lower():
        return "synthesizer"
    elif "plan" in task.lower() or "steps" in task.lower():
        return "planner"
    elif "improve" in task.lower() or "optimize" in task.lower():
        return "evolver"
    else:
        return "meta-orchestrator"  # default
```

---

#### **14. Reputation-Router**
**Mission**: Track agent performance, route to best performer

**When to Use**: Continuous

**Metrics Tracked**:
- Success rate (task completed correctly)
- Cost per task
- Time per task
- User satisfaction (human feedback)

**Example Routing Decision**:
```
Task: "Generate creative startup idea"

Candidates:
- evolver: 78% success rate, $0.12/idea, 30 sec
- synthesizer: 82% success rate, $0.18/idea, 45 sec
- insight-synthesizer: 71% success rate, $0.09/idea, 20 sec

Decision: Route to synthesizer (highest success rate, cost acceptable)
```

---

#### **15. Tool-Autogen**
**Mission**: Create custom tools on-demand when needed

**When to Use**: Existing tools insufficient

**Example**:
```
User: "I need to scrape competitor pricing daily and alert me if they change"

Tool-Autogen creates:
1. Python script: competitor_price_scraper.py
2. Cron job: Runs daily at 9 AM
3. Alert system: Sends Slack message on price change
4. Database: Stores historical prices
5. Dashboard: Visualizes price trends

Time to create: 5 minutes (automated)
Cost: $0.50 (LLM API calls)
```

---

#### **16. Benchmark-Orchestrator**
**Mission**: Rigorously test ideas against benchmarks

**When to Use**: Before finalizing idea

**Benchmark Types**:
- **Novelty Benchmark**: Is this truly new? (Check against 10M existing ideas)
- **Feasibility Benchmark**: Can it be built with current tech?
- **Market Benchmark**: Is there demand? (Check Google Trends, competitor analysis)
- **Viral Benchmark**: Will it spread? (A/B test with focus groups)

**Example**:
```
Idea: "AI that writes your obituary while you're alive to help you live better"

Benchmarks:
- Novelty: 92/100 (similar concepts exist but not this exact implementation)
- Feasibility: 95/100 (LLMs can do this easily)
- Market: 67/100 (niche but passionate audience)
- Viral: 88/100 (morbid curiosity + self-reflection = highly shareable)

Verdict: APPROVED (overall score: 85.5/100)
```

---

#### **17. Hypothesis-Refutation**
**Mission**: Actively try to disprove ideas (Popperian falsification)

**When to Use**: Final validation step

**Approach**:
```
Idea: "Subscription box for scientific papers (curated by AI)"

Refutation Attempts:
1. "People can get papers for free" → REFUTED (curation has value)
2. "AI curation won't be good enough" → REFUTED (GPT-4 can do this)
3. "Market too small" → REFUTED (300K+ researchers globally)
4. "Already exists (JSTOR)" → REFUTED (JSTOR is search, not curation)
5. "Won't pay for this" → PARTIALLY VALID (need to prove willingness to pay)

Result: 1/5 refutations stand → Idea has 80% confidence
Recommendation: Run pricing experiment before full build
```

---

## 🧠 15 THINKING FRAMEWORKS

### Framework 1: First Principles (Elon Musk)

**Core Concept**: Break down to fundamental truths, rebuild from scratch

**Process**:
1. Identify the problem/goal
2. Break it down to absolute fundamentals (physics, math, human nature)
3. Question every assumption
4. Rebuild solution from fundamentals only

**Example**:
```
Problem: "Research papers are hard to discover"

Conventional thinking: "We need better search algorithms"

First principles:
- Fundamental: Information wants to be found by those who need it
- Fundamental: Relevance is subjective and context-dependent
- Fundamental: Human attention is limited
- Rebuild: Instead of searching, what if papers found YOU? (Reverse discovery)

Result: "Research Radar" - AI monitors your work and proactively suggests papers
```

---

### Framework 2: Inversion (Charlie Munger)

**Core Concept**: Instead of "How do I succeed?", ask "How do I fail?"

**Process**:
1. Define success
2. Invert: What would guarantee failure?
3. Avoid all failure modes
4. What's left = path to success

**Example**:
```
Goal: "Create a viral research platform"

Invert: "How to guarantee NO ONE uses it?"
- Make it complicated (10-step signup)
- Charge $1,000/month immediately
- No mobile app (desktop only)
- Slow and buggy
- No social features
- Ugly design

Avoid all these → Success criteria:
- Simple signup (1-click)
- Free tier
- Mobile-first
- Fast and reliable
- Built-in social sharing
- Beautiful UI
```

---

### Framework 3: Second-Order Thinking

**Core Concept**: Think beyond immediate consequences to 2nd, 3rd, nth order effects

**Process**:
1. First-order: What happens immediately?
2. Second-order: What happens as a result of that?
3. Third-order: And then what?
4. Keep asking "And then what?" until you reach stable state

**Example**:
```
Idea: "Free AI code generation for developers"

1st order: Developers save time → Happy developers
2nd order: Less need for junior developers → Job displacement
3rd order: Coding becomes commodity → Value shifts to architecture/design
4th order: New skills emerge (prompt engineering, AI orchestration)
5th order: New job categories created (AI whisperers, agent designers)
6th order: Programming education changes fundamentally

Insight: Focus on training people for 4th/5th order skills, not fighting 2nd order
```

---

### Framework 4: Lateral Thinking (Edward de Bono)

**Core Concept**: Solve problems by approaching from unexpected angles

**Techniques**:
- **Random Input**: Force-connect unrelated concepts
- **Provocation**: Make deliberately illogical statement, then make it work
- **Challenge Assumptions**: Question "obvious" truths

**Example**:
```
Problem: "How to make scientific research more accessible?"

Conventional: Better search, open access, simpler language

Lateral Thinking (Random Input: "TikTok"):
→ "What if research papers were 60-second videos?"
→ "Scientific TikTok" - Researchers explain papers in dance form
→ Sounds absurd, but... could actually work for making science viral

Provocation: "Papers should be HARDER to access, not easier"
→ Make them so exclusive people crave them
→ "Research Speakeasy" - Secret clubs to access cutting-edge research
→ Creates desire through scarcity (counterintuitive but psychologically valid)
```

---

### Framework 5: SCAMPER

**Core Concept**: Systematic creativity through 7 operations

**Operations**:
1. **S**ubstitute: Replace components
2. **C**ombine: Merge with other things
3. **A**dapt: Modify for new context
4. **M**odify: Change attributes
5. **P**ut to other use: Repurpose
6. **E**liminate: Remove components
7. **R**everse: Flip order/direction

**Example (Applied to "Academic Publishing")**:
```
S - Substitute peer review with AI → "AI Peer Review Platform"
C - Combine papers with datasets → "Executable Papers" (code + paper in one)
A - Adapt blog format to research → "Research Blogging Platform"
M - Modify timeline (instant, not 6 months) → "Real-Time Research Feed"
P - Put to other use: Use papers for teaching → "Papers as Curriculum"
E - Eliminate journal gatekeepers → "Direct-to-Reader Publishing"
R - Reverse: Readers pay authors, not vice versa → "Substack for Science"
```

---

### Framework 6: TRIZ (Theory of Inventive Problem Solving)

**Core Concept**: Most problems have been solved before in different domains. Find the pattern.

**40 Inventive Principles** (sample):
- **Segmentation**: Divide into parts
- **Taking Out**: Extract the problematic part
- **Local Quality**: Change structure from uniform to non-uniform
- **Asymmetry**: Change shape from symmetric to asymmetric
- **Merging**: Bring closer together in space or time
- **Universality**: Make object perform multiple functions
- **Nested Doll**: Place one object inside another
- **Anti-Weight**: Compensate for weight by aerodynamic forces
- **Preliminary Action**: Perform required action in advance
- **Beforehand Cushioning**: Prepare emergency means beforehand

**Example**:
```
Problem: "Research papers are long and boring"

TRIZ Principle Applied: "Segmentation"
→ Break paper into micro-chunks (tweet-sized insights)
→ "Atomic Research" - Papers as collection of 280-char findings

TRIZ Principle Applied: "Nested Doll"
→ Summary inside summary inside summary
→ Read at your depth level (abstract → 1-page → 10-page → full paper)

TRIZ Principle Applied: "Preliminary Action"
→ AI pre-reads paper and answers your questions BEFORE you read
→ "Research Concierge" - Ask questions, get answers, decide if worth reading
```

---

### Framework 7: Biomimicry

**Core Concept**: Nature has solved problems for 3.8 billion years. Copy nature.

**Process**:
1. Define problem
2. Find analogous problem in nature
3. Study how nature solved it
4. Abstract the principle
5. Apply to your problem

**Example**:
```
Problem: "How to organize massive amounts of research papers?"

Nature Analogy: How does brain organize memories?
- Associative networks (not hierarchical folders)
- Strengthens connections through repeated access
- Pruning of unused connections
- Pattern recognition across memories

Application:
→ "Memory Palace for Research" - Papers organized by association, not category
→ Frequently accessed papers appear closer together
→ Unused papers fade into background
→ AI detects patterns across your reading history
```

---

### Framework 8: Cross-Domain Analogy

**Core Concept**: Apply solutions from Domain A to problems in Domain B

**Process**:
1. Deeply understand problem in Domain B
2. Find structural similarity in Domain A
3. Map solution from A → B
4. Test if it works

**Example**:
```
Domain A: Matchmaking (Tinder)
Domain B: Scientific Collaboration

Mapping:
- Tinder matches people by mutual attraction
- Science needs: Researchers matched by complementary skills

Transfer:
→ "Scientific Tinder" (Idea #1 from original list)

Domain A: Stock Market
Domain B: Research Ideas

Mapping:
- Stock market prices ideas via supply/demand
- Research needs: Prioritize ideas by collective wisdom

Transfer:
→ "Failure Futures Market" (Idea #2 from original list)
```

---

### Framework 9: Constraint Optimization

**Core Concept**: Add constraints to force creativity

**Constraints to Add**:
- Time: "Must build in 1 week"
- Budget: "Maximum $100 to launch"
- Technology: "No internet required"
- Users: "For blind users only"
- Format: "Fits on a t-shirt"

**Example**:
```
Problem: "Scientific collaboration platform"

Constraint 1: "Must work via SMS only (no app, no internet)"
→ "SMS Science Network" - Text-based hypothesis sharing
→ Works in developing countries with limited internet

Constraint 2: "For 5-year-olds"
→ "Science Kindergarten" - Explains research in picture books
→ Actually useful for science communication

Constraint 3: "Must generate revenue from Day 1"
→ "Paid Peer Review" - Scientists pay for quality reviews
→ Solves peer review crisis AND monetizes immediately
```

---

### Framework 10: Fail-Fast Exploration

**Core Concept**: Test ideas cheaply and quickly. Kill bad ideas early.

**Process**:
1. Generate 100 ideas
2. Test top 10 with minimal investment (1 day, $100 each)
3. Kill 7, double down on 3
4. Test those 3 more seriously (1 week, $1,000 each)
5. Kill 2, build 1

**Example**:
```
Generate: 100 research platform ideas
↓
Quick Test (landing page + ads):
- Idea A: 2% conversion → KILL
- Idea B: 8% conversion → KEEP
- Idea C: 1% conversion → KILL
- Idea D: 12% conversion → KEEP
- Idea E: 5% conversion → KEEP
↓
Medium Test (MVP):
- Idea B: 40% signup → build rate, 10% pay → KEEP
- Idea D: 60% signup → build rate, 2% pay → KILL (won't monetize)
- Idea E: 30% signup → build rate, 15% pay → KEEP
↓
Full Build:
- Idea B (best balance)
- Idea E (highest revenue per user)
```

---

### Framework 11: Adversarial Thinking

**Core Concept**: Think like your worst enemy. Attack your own ideas.

**Questions to Ask**:
- "How would a competitor destroy this?"
- "What's the dumbest assumption I'm making?"
- "What if the opposite were true?"
- "How could this backfire spectacularly?"

**Example**:
```
Idea: "AI-powered research assistant (like ChatGPT for science)"

Adversarial Attacks:
1. "AI will hallucinate fake citations" → Add citation verification system
2. "People will use it to cheat" → Add plagiarism detection
3. "Won't be better than Google Scholar" → Focus on conversation, not search
4. "Too expensive to run (LLM costs)" → Freemium model + caching
5. "Privacy concerns (research data)" → Local-first option

Result: Idea becomes 10x more robust
```

---

### Framework 12: Probabilistic Reasoning

**Core Concept**: Think in probabilities, not certainties

**Process**:
1. Assign probability to each assumption
2. Calculate joint probability of success
3. Identify highest-risk assumptions
4. Test those first

**Example**:
```
Idea: "AI-generated scientific papers"

Assumptions & Probabilities:
1. "AI can write coherent papers": 95% (already proven)
2. "Scientists will trust AI-generated content": 40% (skeptical field)
3. "Journals will publish AI-written papers": 20% (resistance expected)
4. "Authors will want AI co-author credit": 60% (uncertain)
5. "Regulatory approval": 30% (new territory)

Joint Probability: 0.95 × 0.40 × 0.20 × 0.60 × 0.30 = 1.4%

Insight: Focus on #2 and #3 (lowest probabilities). If you can increase those, idea becomes viable.

Action: Build trust through transparency (show AI reasoning, allow verification)
```

---

### Framework 13: Emergence Detection

**Core Concept**: Look for patterns emerging from chaos. What's becoming possible NOW that wasn't 6 months ago?

**Signals to Watch**:
- Technology breakthroughs
- Regulatory changes
- Cultural shifts
- Economic changes
- Demographic trends

**Example**:
```
Emerging Signals (November 2024):
1. LLMs can now handle 200K+ token context (Claude 3.5)
2. Text-to-video is production-ready (Sora, RunwayML)
3. AI agents can browse web and use tools
4. Voice cloning is indistinguishable from real (ElevenLabs)
5. Code generation reaches 70%+ accuracy (GitHub Copilot)

What's Now Possible (wasn't 6 months ago):
→ "AI Researcher" that reads 1000 papers, synthesizes findings
→ "Video Research Explanations" auto-generated from papers
→ "Research Shopping Assistant" that finds papers you didn't know you needed
→ "Podcast Generator" - Any paper becomes podcast conversation
→ "Auto-Code" - Research methods sections become executable code

New Ideas Emerge from Convergence:
→ "One-Click Replication" - Read paper, click button, code runs experiment
```

---

### Framework 14: Historical Precedent Analysis

**Core Concept**: Study how similar problems were solved in the past

**Process**:
1. Find historical analogy
2. Study what worked / didn't work
3. Abstract the principles
4. Apply to current problem (avoiding past mistakes)

**Example**:
```
Problem: "How to accelerate scientific discovery?"

Historical Precedents:
1. **Bell Labs (1925-1980s)**:
   - What worked: Interdisciplinary teams, long-term funding, freedom to explore
   - What didn't: Too isolated from market, eventually shut down
   - Principle: Mix basic research + applied, but maintain market connection

2. **Manhattan Project (1942-1946)**:
   - What worked: Clear goal, unlimited resources, best talent, urgency
   - What didn't: Couldn't sustain post-war, ethics concerns
   - Principle: Moonshot goals work, but need ethical framework

3. **Open Source Movement (1998-present)**:
   - What worked: Distributed collaboration, transparency, rapid iteration
   - What didn't: Sustainability (many projects abandoned)
   - Principle: Openness accelerates progress, but needs incentive structure

Modern Application:
→ "Research Co-op" - Combines Bell Labs focus + Manhattan urgency + open source collaboration
→ Funded by members (sustainability), clear goals (focus), transparent (open)
```

---

### Framework 15: Future Extrapolation

**Core Concept**: Imagine the future, work backward to today

**Process**:
1. Define a future scenario (5, 10, 20 years out)
2. Ask: "What needs to be true for this to happen?"
3. Work backward: "What needs to happen next year? This month? Today?"
4. Start building today's step

**Example**:
```
Future (2035): "AI runs most scientific research autonomously"

What needs to be true:
- AI can form hypotheses (partially true today)
- AI can design experiments (emerging)
- AI can interpret results (true for structured data)
- AI can write papers (true today)
- Scientists trust AI findings (not true yet)
- Journals accept AI-only papers (not true yet)

Backward Chain:
2034: Journals create "AI track" for papers
2033: First major discovery by AI-only team
2032: Hybrid teams (human + AI) become standard
2031: AI can design complex experiments
2030: AI gains ability to formulate novel hypotheses
2029: Trust-building phase (transparent AI reasoning)
2028: AI research assistants mainstream
2027: Early adopters use AI for literature review
2026: AI tools integrated into research workflow
2025 (today): Build the tools that start this chain

Action Today: Create "AI Research Assistant" MVP
```

---

## 🎨 IDEA GENERATION PATTERNS

### Pattern 1: The Collision Matrix

**Concept**: Systematically combine unrelated concepts

**Matrix**:
```
        │ TikTok │ Poker │ Therapy │ F1 Racing │ Bitcoin
─────────┼────────┼───────┼─────────┼───────────┼─────────
Science  │   A    │   B   │    C    │     D     │    E
Dating   │   F    │   G   │    H    │     I     │    J
Cooking  │   K    │   L   │    M    │     N     │    O
Sports   │   P    │   Q   │    R    │     S     │    T
Education│   U    │   V   │    W    │     X     │    Y

Cell A: Science + TikTok = Scientific TikTok (Idea #1 variant)
Cell B: Science + Poker = "Research Bluffing" (hide methodology, others guess)
Cell C: Science + Therapy = "Hypothesis Therapy" (work through research trauma)
Cell D: Science + F1 = "Research Grand Prix" (fastest to result wins)
Cell E: Science + Bitcoin = "Proof of Research" (blockchain for citations)
... (infinite combinations)
```

---

### Pattern 2: The Extreme Dial

**Concept**: Take a variable and turn it to extreme (0% or 100%)

**Example**:
```
Variable: "Cost of research paper access"

Traditional: $30-50 per paper
Turn to 0%: Completely free → "Sci-Hub" (already exists)
Turn to 100% of GDP: Infinitely expensive → "Research as Ultimate Luxury"
  → Only ultra-wealthy can afford cutting-edge knowledge
  → Creates intense desire and black markets
  → "Research Speakeasy" emerges

Variable: "Time from submission to publication"

Traditional: 6-12 months
Turn to 0: Instant publication → "arXiv" (exists)
Turn to 100 years: Century-long peer review → "Research Time Capsule"
  → Papers reviewed by future generations
  → Long-term thinking enforced
  → Predictions tested over decades
```

---

### Pattern 3: The Inversion Cascade

**Concept**: Invert every assumption in sequence

**Example**:
```
Standard assumption: "Research papers should be text"
Invert: "Research papers should be... not text"
→ Video papers? ✓ (YouTube research explanations)
→ Audio papers? ✓ (Research podcasts)
→ Tactile papers? ✓ (3D-printed data visualizations for blind)
→ Taste papers? 🤔 (Chemistry research you can taste?)
→ Smell papers? 🤔 (Perfume papers for olfactory science?)

Standard assumption: "Researchers work alone or in small teams"
Invert: "Research requires massive collaboration (1000+ people)"
→ "Distributed Research DAO" - 1000s contribute micro-tasks
→ "Research MMO" - Massively multiplayer online research game

Standard assumption: "Papers are final once published"
Invert: "Papers continuously evolve after publication"
→ "Living Papers" - Version controlled, community-edited
→ Wikipedia for research (already emerging)
```

---

### Pattern 4: The Metaphor Miner

**Concept**: Force-apply metaphors from other domains

**Metaphors**:
```
"Research is like..."

...Gardening:
→ Plant ideas (hypotheses)
→ Water them (experiments)
→ Prune dead branches (refute bad ideas)
→ Harvest results (publications)
Application: "Research Garden Platform" - Visual growth of ideas

...Cooking:
→ Ingredients (data, theories)
→ Recipes (methods)
→ Taste test (peer review)
→ Restaurant (journal)
Application: "Research Kitchen" - Collaborate on recipes (methods)

...Dating:
→ Match (collaboration)
→ First date (initial meeting)
→ Relationship (co-authorship)
→ Marriage (long-term partnership)
Application: "Academic Match.com" (already in idea list)

...War:
→ Strategy (research agenda)
→ Battles (experiments)
→ Alliances (collaborations)
→ Victory (breakthrough)
Application: "Research Battlefield" - Competitive research game
```

---

### Pattern 5: The Constraint Lattice

**Concept**: Add multiple constraints simultaneously

**Example**:
```
Base idea: "Scientific collaboration platform"

Add constraints:
1. "Works offline"
2. "Uses only voice (no screens)"
3. "For people with dementia"
4. "Costs $0 to run"
5. "Runs on solar power"

Result: "Solar-Powered Voice Research Assistant for Alzheimer's Patients"
→ Actually useful: Helps researchers interview dementia patients
→ Voice recording + AI transcription
→ Solar-powered field research device
→ Free and accessible

Unexpected use case discovered through constraints!
```

---

### Pattern 6: The Recursion Generator

**Concept**: Apply the idea to itself

**Example**:
```
Idea: "AI that generates research ideas"

Apply to itself:
→ "AI that generates ideas for better AI idea generators"
→ "Meta-AI Researcher" - AI that improves itself

Idea: "Peer review platform"

Apply to itself:
→ "Peer review the peer review process"
→ "Meta-Peer-Review" - Review the quality of reviews

Idea: "Research automation"

Apply to itself:
→ "Automate the automation of research"
→ "Self-Improving Research Automation" (this system!)
```

---

## 📁 COMPLETE GOLDEN TEMPLATE INTEGRATION

### Repository Structure (for ANY idea generated)

```
idea-name/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Lint, test, security (on every PR)
│   │   ├── deploy.yml                # Deploy to production (on merge to main)
│   │   ├── security-scan.yml         # Weekly security scans (Semgrep, Gitleaks)
│   │   ├── dependency-update.yml     # Dependabot automation
│   │   └── docs-publish.yml          # Auto-publish docs to GitHub Pages
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   └── research_question.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS                    # Auto-assign reviewers
│   └── dependabot.yml
│
├── .claude/                           # Claude Code integration
│   ├── commands/
│   │   ├── analyze.md                # Custom slash command: /analyze
│   │   ├── refactor.md               # Custom slash command: /refactor
│   │   └── test.md                   # Custom slash command: /test
│   └── prompts/
│       └── project-context.md        # Project-specific AI instructions
│
├── src/
│   └── idea_name/                    # Main source code (Python example)
│       ├── __init__.py
│       ├── core/                     # Core business logic
│       │   ├── __init__.py
│       │   ├── models.py             # Pydantic models
│       │   ├── logic.py              # Business logic
│       │   └── utils.py
│       ├── api/                      # API layer (FastAPI)
│       │   ├── __init__.py
│       │   ├── routes/
│       │   │   ├── users.py
│       │   │   ├── ideas.py
│       │   │   └── health.py
│       │   └── dependencies.py
│       ├── db/                       # Database layer
│       │   ├── __init__.py
│       │   ├── models.py             # SQLAlchemy models
│       │   ├── migrations/           # Alembic migrations
│       │   └── session.py
│       ├── agents/                   # AI agents (if applicable)
│       │   ├── __init__.py
│       │   ├── meta_orchestrator.py
│       │   ├── researcher.py
│       │   └── synthesizer.py
│       └── cli/                      # CLI tools
│           ├── __init__.py
│           └── main.py
│
├── tests/
│   ├── unit/                         # Fast, isolated tests
│   │   ├── test_models.py
│   │   ├── test_logic.py
│   │   └── test_utils.py
│   ├── integration/                  # API + database tests
│   │   ├── test_api.py
│   │   └── test_db.py
│   ├── e2e/                          # End-to-end tests (Playwright)
│   │   └── test_user_flow.py
│   ├── fixtures/                     # Test data
│   │   ├── sample_data.json
│   │   └── mock_responses.py
│   └── conftest.py                   # pytest configuration
│
├── docs/
│   ├── api/                          # API reference (auto-generated)
│   │   └── openapi.json
│   ├── guides/                       # User guides
│   │   ├── getting-started.md
│   │   ├── advanced.md
│   │   └── faq.md
│   ├── architecture/                 # System design docs
│   │   ├── overview.md
│   │   ├── database.md
│   │   └── deployment.md
│   ├── ssot/                         # Single Source of Truth
│   │   ├── role-prompts.md           # AI agent roles
│   │   ├── decision-log.md           # Why we made decisions
│   │   └── multi-ai-strategy.md      # Claude + GPT-4 + Gemini orchestration
│   ├── research/                     # Research notes (if applicable)
│   │   ├── literature-review.md
│   │   └── hypothesis-log.md
│   ├── DEVELOPER.md                  # Dev setup instructions
│   └── _build/                       # Sphinx build output
│
├── mcp/                              # Model Context Protocol
│   ├── server.py                     # MCP server for AI assistants
│   ├── tools/                        # Custom MCP tools
│   │   ├── code_analyzer.py
│   │   └── test_runner.py
│   └── resources/                    # MCP resource definitions
│       └── project_docs.json
│
├── automation/
│   ├── agents/                       # Autonomous agents
│   │   ├── code_maintainer.py        # Keeps code quality high
│   │   ├── governance_validator.py   # Checks compliance
│   │   └── cost_optimizer.py         # Monitors cloud costs
│   └── workflows/                    # n8n / Zapier workflows
│       ├── onboarding.json
│       └── monitoring.json
│
├── scripts/
│   ├── setup.sh                      # One-command setup (Unix)
│   ├── setup.ps1                     # One-command setup (Windows)
│   ├── deploy.sh                     # Deployment script
│   ├── backup.sh                     # Backup script (daily cron)
│   └── analyze.py                    # Code quality analysis
│
├── config/                           # Configuration files
│   ├── development.yaml              # Dev environment config
│   ├── staging.yaml                  # Staging environment
│   ├── production.yaml               # Production environment
│   └── secrets.example.yaml          # Template for secrets (not committed)
│
├── infrastructure/                   # IaC (Infrastructure as Code)
│   ├── terraform/                    # Terraform for AWS/GCP
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── docker/                       # Docker configuration
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── logs/                             # Application logs
├── results/                          # Experiment results (if research project)
├── archive/                          # Archived data
│
├── .gitignore                        # Git ignore patterns
├── .pre-commit-config.yaml           # 25+ pre-commit hooks
├── .editorconfig                     # Editor configuration
├── .env.example                      # Environment variables template
│
├── pyproject.toml                    # Python packaging (PEP 621)
├── setup.py                          # Setup script (legacy support)
├── requirements.txt                  # Dependencies (pinned)
├── requirements-dev.txt              # Dev dependencies
│
├── Makefile                          # Task runner (Unix)
├── justfile                          # Task runner (modern, cross-platform)
│
├── README.md                         # Project overview
├── CONTRIBUTING.md                   # Contribution guidelines
├── SECURITY.md                       # Security policy
├── SUPPORT.md                        # Getting help
├── LICENSE                           # MIT License
├── CHANGELOG.md                      # Version history
├── CODE_OF_CONDUCT.md                # Community standards
└── AGENTS.md                         # AI agent instructions (from template)
```

### Key Files Explained

#### `.github/workflows/ci.yml` - Continuous Integration

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff black isort mypy

      - name: Run Ruff
        run: ruff check src tests

      - name: Run Black
        run: black --check src tests

      - name: Run isort
        run: isort --check-only src tests

      - name: Run MyPy
        run: mypy src

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Gitleaks Scan
        uses: gitleaks/gitleaks-action@v2

      - name: Semgrep Scan
        run: |
          pip install semgrep
          semgrep --config=p/security-audit --config=p/owasp-top-ten src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests with coverage
        run: pytest -n auto --cov=src --cov-report=xml --cov-report=term

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=85

      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: [lint-and-format, security, test]
    steps:
      - uses: actions/checkout@v4

      - name: Build package
        run: python -m build

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/
```

#### `.pre-commit-config.yaml` - 25+ Quality Checks

```yaml
repos:
  # Secrets scanning
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets

  # Code formatting
  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
        args: ['--line-length=100']

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']

  # Linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: ['--fix']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        args: ['--strict']

  # File checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements
      - id: detect-private-key

  # Commit message
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v2.4.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

#### `pyproject.toml` - Project Configuration

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "idea-name"
version = "0.1.0"
description = "Generated from UltraThink Idea Generator"
readme = "README.md"
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
keywords = ["ai", "research", "innovation"]

dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.4.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "httpx>=0.25.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.10.0",
    "ruff>=0.1.0",
    "isort>=5.12.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
]

ai = [
    "openai>=1.3.0",
    "anthropic>=0.7.0",
    "langchain>=0.0.300",
    "pinecone-client>=2.2.0",
]

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "B", "I", "C4", "UP"]
ignore = ["E501"]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra -q --strict-markers --cov=src --cov-report=term --cov-report=html"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests",
]

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
precision = 2
fail_under = 85
show_missing = true
```

#### `docs/ssot/role-prompts.md` - AI Agent Instructions

```markdown
# AI Agent Role Prompts - Single Source of Truth

## Project: [Idea Name]

### Context
This project uses a multi-AI strategy with role-specific prompts for different agents.

### Agents

#### Claude (Research & Architecture)
**Role**: Senior Architect + Researcher
**Strengths**: Long context (200K tokens), analysis, system design
**Use For**:
- Analyzing large codebases
- Designing system architecture
- Literature reviews
- Long-form documentation

**Prompt Template**:
```
You are the Senior Architect for [Project Name].

Current Task: [TASK]

Context:
- Tech Stack: [STACK]
- Current Phase: [PHASE]
- Constraints: [CONSTRAINTS]

Instructions:
1. Analyze the codebase structure
2. Propose architecture improvements
3. Document decisions in ADR format
4. Ensure alignment with golden template standards

Deliverable: [EXPECTED OUTPUT]
```

#### GPT-4 (Code Generation & Validation)
**Role**: Senior Developer + QA Engineer
**Strengths**: Code generation, testing, validation
**Use For**:
- Implementing features
- Writing tests
- Refactoring code
- API design

**Prompt Template**:
```
You are the Senior Developer for [Project Name].

Current Task: [TASK]

Code Quality Requirements:
- 85%+ test coverage
- Type hints on all functions
- Docstrings (Google style)
- Ruff, Black, isort compliant

Instructions:
1. Write production-ready code
2. Include comprehensive tests
3. Add docstrings and type hints
4. Ensure linters pass

Deliverable: [EXPECTED OUTPUT]
```

#### Gemini (QA & Edge Cases)
**Role**: QA Lead + Security Auditor
**Strengths**: Finding edge cases, security review
**Use For**:
- Testing edge cases
- Security audits
- Performance testing
- Accessibility review

**Prompt Template**:
```
You are the QA Lead for [Project Name].

Current Task: [TASK]

Testing Requirements:
- Test all edge cases
- Security review (OWASP Top 10)
- Performance benchmarks
- Accessibility audit

Instructions:
1. Test all code paths
2. Document edge cases
3. Security scan
4. Performance report

Deliverable: [EXPECTED OUTPUT]
```

### Multi-AI Orchestration

**When to use which AI**:
```python
def route_to_ai(task_type: str) -> str:
    if task_type in ["research", "architecture", "analysis"]:
        return "claude"
    elif task_type in ["coding", "testing", "refactoring"]:
        return "gpt4"
    elif task_type in ["qa", "security", "edge_cases"]:
        return "gemini"
    else:
        return "claude"  # Default
```
```

---

## 🔄 DEVELOPMENT WORKFLOW & STANDARDS

### Git Workflow (GitHub Flow + Semantic Versioning)

#### Branch Strategy

```
main (production)
  ├── develop (staging)
  │   ├── feature/ISSUE-123-add-user-auth
  │   ├── feature/ISSUE-124-add-api-endpoint
  │   ├── bugfix/ISSUE-125-fix-login-error
  │   └── refactor/ISSUE-126-optimize-database
  └── hotfix/ISSUE-127-critical-security-patch
```

**Branch Types**:
1. **main**: Production-ready code (protected)
2. **develop**: Integration branch (protected)
3. **feature/***: New features
4. **bugfix/***: Bug fixes
5. **hotfix/***: Critical production fixes
6. **refactor/***: Code improvements (no new features)
7. **chore/***: Maintenance (dependencies, config)
8. **docs/***: Documentation only

#### Branch Naming Convention

```
<type>/<issue-number>-<short-description>

Examples:
- feature/123-add-hypothesis-matcher
- bugfix/124-fix-login-redirect
- refactor/125-optimize-query-performance
- docs/126-update-api-documentation
```

#### Commit Message Standard (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance
- `ci`: CI/CD changes

**Examples**:
```
feat(auth): add JWT refresh token rotation

Implemented automatic token refresh when access token expires.
This prevents users from being logged out during active sessions.

Closes #123

---

fix(api): handle null values in user profile endpoint

Added null checks for optional fields (bio, avatar, website).
Previously threw 500 error when fields were missing.

Fixes #124

---

refactor(db): optimize hypothesis query performance

Replaced N+1 queries with single JOIN query.
Reduced API response time from 800ms to 120ms.

Related to #125
```

#### Pull Request Template

```markdown
## Description
[Describe the changes in this PR]

## Related Issues
Closes #123
Related to #124

## Type of Change
- [ ] Feature (new functionality)
- [ ] Bug fix (fixes an issue)
- [ ] Refactor (improves code, no new features)
- [ ] Documentation
- [ ] Configuration/Infrastructure

## Checklist
- [ ] Code follows project style guide
- [ ] Tests added/updated (85%+ coverage maintained)
- [ ] Documentation updated
- [ ] CI/CD checks pass
- [ ] No merge conflicts
- [ ] Reviewed by at least 1 person
- [ ] Breaking changes documented

## Screenshots (if applicable)
[Add before/after screenshots]

## Testing Done
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Tested on [browsers/devices]

## Performance Impact
[If applicable, describe performance changes]

## Security Considerations
[If applicable, describe security implications]
```

### Branch Backup Strategy

#### Backup Tiers

**1. Recent Backups** (Hourly, Keep 24 hours)
```bash
# Automated via GitHub Actions
.github/workflows/backup-recent.yml

on:
  schedule:
    - cron: '0 * * * *'  # Every hour

jobs:
  backup:
    - Clone all branches
    - Push to backup repo (github.com/your-org/project-backup)
    - Retention: 24 hours
```

**2. Daily Backups** (Keep 7 days)
```bash
# Automated via GitHub Actions
.github/workflows/backup-daily.yml

on:
  schedule:
    - cron: '0 0 * * *'  # Midnight UTC

jobs:
  backup:
    - Full repository backup (including LFS)
    - Database dump
    - Push to AWS S3
    - Retention: 7 days
```

**3. Weekly Backups** (Keep 4 weeks)
```bash
# Automated via GitHub Actions
.github/workflows/backup-weekly.yml

on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight

jobs:
  backup:
    - Full repository backup
    - Database snapshot
    - Documentation snapshot
    - Push to AWS S3 + Glacier
    - Retention: 4 weeks
```

**4. Monthly Backups** (Keep 12 months)
```bash
# Automated via GitHub Actions
.github/workflows/backup-monthly.yml

on:
  schedule:
    - cron: '0 0 1 * *'  # 1st of month

jobs:
  backup:
    - Full repository archive
    - Database full backup
    - All artifacts
    - Push to AWS Glacier (long-term storage)
    - Retention: 12 months
```

#### Recovery Procedures

```bash
# Recover from recent backup (< 24 hours)
git clone https://github.com/your-org/project-backup
git checkout <branch-name>@<timestamp>

# Recover from daily backup (< 7 days)
aws s3 cp s3://backups/project-name/daily/2025-11-14.tar.gz .
tar -xzf 2025-11-14.tar.gz

# Recover from weekly backup (< 4 weeks)
aws s3 cp s3://backups/project-name/weekly/2025-W45.tar.gz .
tar -xzf 2025-W45.tar.gz

# Recover from monthly backup (< 12 months)
aws glacier get-archive --vault-name project-backups --archive-id <ID>
```

### Naming Conventions (Complete Reference)

#### File Naming

**Python**:
```
snake_case.py
test_snake_case.py
__init__.py
```

**JavaScript/TypeScript**:
```
kebab-case.js
kebab-case.tsx
index.ts
```

**Markdown**:
```
UPPER_CASE.md
00-README.md
01-GETTING_STARTED.md
```

**Configuration**:
```
lowercase.yaml
lowercase.json
lowercase.toml
.lowercase-rc
```

#### Directory Naming

```
snake_case/        # Python packages
kebab-case/        # JavaScript modules
UPPER_CASE/        # Documentation folders
lowercase/         # Configuration folders
```

#### Variable Naming

**Python**:
```python
# Variables and functions: snake_case
user_name = "John"
def calculate_total_price():
    pass

# Classes: PascalCase
class UserProfile:
    pass

# Constants: UPPER_CASE
MAX_RETRIES = 3
API_BASE_URL = "https://api.example.com"

# Private: _leading_underscore
_internal_cache = {}

# Protected: __double_underscore
__private_method()
```

**JavaScript/TypeScript**:
```javascript
// Variables and functions: camelCase
const userName = "John";
function calculateTotalPrice() {}

// Classes: PascalCase
class UserProfile {}

// Constants: UPPER_CASE
const MAX_RETRIES = 3;
const API_BASE_URL = "https://api.example.com";

// Private (TypeScript): #leading_hash
class Example {
  #privateField = "secret";
}
```

**SQL**:
```sql
-- Tables: snake_case
CREATE TABLE user_profiles (
  -- Columns: snake_case
  user_id INTEGER PRIMARY KEY,
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### API Naming

**REST Endpoints**:
```
GET    /api/v1/users
POST   /api/v1/users
GET    /api/v1/users/{id}
PUT    /api/v1/users/{id}
DELETE /api/v1/users/{id}

GET    /api/v1/users/{id}/posts
GET    /api/v1/users/{id}/posts/{post_id}
```

**GraphQL**:
```graphql
type User {
  id: ID!
  fullName: String!
  createdAt: DateTime!
}

query GetUser($id: ID!) {
  user(id: $id) {
    fullName
  }
}

mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    id
  }
}
```

### Issues, Actions, and Related Items

#### Issue Labels

```
Type:
- bug (red)
- feature (green)
- enhancement (blue)
- documentation (yellow)
- question (purple)
- refactor (orange)

Priority:
- critical (dark red)
- high (red)
- medium (yellow)
- low (green)

Status:
- needs-triage (gray)
- in-progress (blue)
- blocked (red)
- ready-for-review (green)

Component:
- backend (cyan)
- frontend (pink)
- database (brown)
- infrastructure (teal)
- ai-agents (purple)
```

#### Issue Template (Feature Request)

```markdown
---
name: Feature Request
about: Suggest a new feature
title: "[FEATURE] "
labels: feature, needs-triage
assignees: ''
---

## Feature Description
[Clear description of the feature]

## Problem Statement
[What problem does this solve?]

## Proposed Solution
[How would this feature work?]

## Alternatives Considered
[What other approaches did you consider?]

## Additional Context
[Screenshots, mockups, references]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Estimated Effort
- [ ] Small (< 1 day)
- [ ] Medium (1-3 days)
- [ ] Large (> 3 days)
- [ ] Needs breakdown

## Dependencies
[List any dependencies or blockers]
```

#### GitHub Actions (Custom Workflows)

**Auto-Label PRs**:
```yaml
# .github/workflows/label-pr.yml
name: Label PR

on:
  pull_request:
    types: [opened, edited]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v4
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          configuration-path: .github/labeler.yml
```

**Auto-Assign Reviewers**:
```yaml
# .github/workflows/auto-assign.yml
name: Auto Assign

on:
  pull_request:
    types: [opened]

jobs:
  assign:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v1.2.0
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
```

**Stale Issue Closer**:
```yaml
# .github/workflows/stale.yml
name: Close Stale Issues

on:
  schedule:
    - cron: '0 0 * * *'

jobs:
  stale:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/stale@v8
        with:
          stale-issue-message: 'This issue is stale and will be closed in 7 days'
          days-before-stale: 30
          days-before-close: 7
```

### Standardizing IDEs, AI/LLMs, and CLI Tools

#### Recommended IDE: VSCode

**.vscode/settings.json**:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "github.copilot.enable": {
    "*": true,
    "python": true,
    "markdown": true
  }
}
```

**.vscode/extensions.json**:
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "github.copilot",
    "github.copilot-chat",
    "ms-vsliveshare.vsliveshare",
    "eamodio.gitlens",
    "yzhang.markdown-all-in-one",
    "redhat.vscode-yaml",
    "tamasfe.even-better-toml"
  ]
}
```

#### AI/LLM Standardization

**Primary Models**:
```yaml
# config/ai-models.yaml
models:
  research:
    primary: "claude-3.5-sonnet"
    fallback: "gpt-4-turbo"

  coding:
    primary: "gpt-4-turbo"
    fallback: "claude-3.5-sonnet"

  qa:
    primary: "gemini-pro"
    fallback: "gpt-4-turbo"

endpoints:
  openai: "https://api.openai.com/v1"
  anthropic: "https://api.anthropic.com/v1"
  google: "https://generativelanguage.googleapis.com/v1"

rate_limits:
  openai: 60_requests_per_minute
  anthropic: 50_requests_per_minute
  google: 60_requests_per_minute

cost_tracking:
  enabled: true
  budget_per_month: 500  # USD
  alert_threshold: 0.8   # 80% of budget
```

#### CLI Tools Standard Stack

```bash
# .tool-versions (asdf)
python 3.11.6
nodejs 20.10.0
terraform 1.6.0
kubectl 1.28.0

# Brewfile (macOS) / apt packages (Linux)
# Essential tools for all team members

# Version control
git
gh  # GitHub CLI

# Languages
python@3.11
node@20
go@1.21

# Package managers
pipx
poetry
npm
yarn

# Cloud
awscli
google-cloud-sdk
terraform

# Kubernetes
kubectl
k9s
helm

# Containers
docker
docker-compose

# Databases
postgresql@15
redis

# AI/LLM tools
openai-cli  # pip install openai
anthropic-cli  # pip install anthropic

# Code quality
ruff
black
isort
mypy
pre-commit

# Utilities
jq
yq
fzf
ripgrep
bat
exa
httpie
```

---

## 🔌 MCP INTEGRATION

### Model Context Protocol (MCP) Architecture

**Purpose**: Provide AI assistants (Claude, GPT-4, Gemini) with structured access to project resources

#### MCP Server Implementation

```python
# mcp/server.py
from typing import Dict, List, Any
from pathlib import Path
import json

class MCPServer:
    """
    MCP server provides AI assistants with:
    - Documentation access
    - Code search capabilities
    - Test execution
    - Coverage reports
    - Build logs
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.resources = self._register_resources()
        self.tools = self._register_tools()

    def _register_resources(self) -> Dict[str, Any]:
        """
        Register all project resources accessible via MCP.

        Resources:
        - docs/**/**: All documentation
        - src/**/**: Source code
        - tests/**/**: Test files
        - coverage.xml: Coverage report
        - logs/**/*.log: Application logs
        """
        return {
            "documentation": {
                "path": self.project_root / "docs",
                "type": "filesystem",
                "index": self._index_markdown_files()
            },
            "source_code": {
                "path": self.project_root / "src",
                "type": "filesystem",
                "index": self._index_python_files()
            },
            "coverage": {
                "path": self.project_root / "coverage.xml",
                "type": "file",
                "parser": "xml"
            },
            "logs": {
                "path": self.project_root / "logs",
                "type": "filesystem",
                "tail_lines": 1000
            }
        }

    def _register_tools(self) -> Dict[str, callable]:
        """
        Register tools that AI assistants can invoke.

        Tools:
        - run_tests(test_path): Execute specific tests
        - search_code(query): Search codebase
        - analyze_complexity(file_path): Compute cyclomatic complexity
        - suggest_refactoring(file_path): AI-powered refactoring suggestions
        """
        return {
            "run_tests": self.run_tests,
            "search_code": self.search_code,
            "analyze_complexity": self.analyze_complexity,
            "suggest_refactoring": self.suggest_refactoring
        }

    def run_tests(self, test_path: str = "tests/") -> Dict[str, Any]:
        """
        Execute tests and return results.

        Returns:
            {
                "passed": 42,
                "failed": 1,
                "coverage": 87.3,
                "failures": [{"test": "test_foo", "error": "AssertionError: ..."}]
            }
        """
        import subprocess

        result = subprocess.run(
            ["pytest", test_path, "--cov=src", "--json-report"],
            capture_output=True,
            text=True
        )

        # Parse JSON output
        report = json.loads(Path(".report.json").read_text())

        return {
            "passed": report["summary"]["passed"],
            "failed": report["summary"]["failed"],
            "coverage": report["coverage"]["total_coverage"],
            "failures": [
                {
                    "test": test["nodeid"],
                    "error": test["call"]["longrepr"]
                }
                for test in report["tests"] if test["outcome"] == "failed"
            ]
        }

    def search_code(self, query: str) -> List[Dict[str, Any]]:
        """
        Search codebase using ripgrep.

        Returns list of matches with context.
        """
        import subprocess

        result = subprocess.run(
            ["rg", "--json", query, str(self.project_root / "src")],
            capture_output=True,
            text=True
        )

        matches = []
        for line in result.stdout.splitlines():
            if line:
                match = json.loads(line)
                if match["type"] == "match":
                    matches.append({
                        "file": match["data"]["path"]["text"],
                        "line": match["data"]["line_number"],
                        "content": match["data"]["lines"]["text"]
                    })

        return matches

    def analyze_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Compute cyclomatic complexity using radon.

        Returns complexity metrics for all functions.
        """
        import subprocess

        result = subprocess.run(
            ["radon", "cc", "--json", file_path],
            capture_output=True,
            text=True
        )

        return json.loads(result.stdout)

    def suggest_refactoring(self, file_path: str) -> List[str]:
        """
        AI-powered refactoring suggestions using GPT-4.

        Analyzes code and suggests improvements.
        """
        from openai import OpenAI

        client = OpenAI()

        with open(file_path) as f:
            code = f.read()

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code review expert. Suggest concrete refactoring improvements."
                },
                {
                    "role": "user",
                    "content": f"Review this code and suggest refactorings:\n\n{code}"
                }
            ]
        )

        return response.choices[0].message.content.split("\n")


if __name__ == "__main__":
    # Start MCP server
    server = MCPServer(Path.cwd())
    print(f"✅ MCP Server running")
    print(f"Resources: {list(server.resources.keys())}")
    print(f"Tools: {list(server.tools.keys())}")
```

#### MCP Client (AI Assistant Integration)

```python
# Example: Claude accessing MCP resources

# In Claude's context:
"""
I have access to your project via MCP. I can:

1. Read all documentation: mcp.read_resource("documentation")
2. Search code: mcp.call_tool("search_code", query="hypothesis")
3. Run tests: mcp.call_tool("run_tests", test_path="tests/unit/")
4. Analyze code: mcp.call_tool("analyze_complexity", file_path="src/core/logic.py")

Let me help you with your project!
"""
```

---

## 🧩 AI INSTRUCTIONS & SELF-REFUTING STATEMENTS

### AI Instructions (Codex AGENTS Template)

#### AGENTS.md Structure

```markdown
# Codex AGENTS Template for [Project Name]

Scope: Applies to the entire repository tree rooted at this file.

## Repo Identity

- GitHub: `https://github.com/your-org/project-name`
- Local path: `C:\Users\...\GitHub\project-name`
- Repo type: `portfolio` (🎯)
- Domain: `research` / `tools` / `education`

## General Behavior for Agents

**Core Principles**:
1. Keep changes **minimal, focused, and reversible**
2. Preserve existing public APIs, file layout, and naming
3. Never move or delete directories containing user data, secrets, or backups
4. Avoid introducing new external services or dependencies unless clearly requested

**Before Making Changes**:
- Read existing code to understand patterns
- Check for tests and maintain coverage
- Follow project style guide (see `.editorconfig`, `pyproject.toml`)
- Ask clarifying questions if requirements are ambiguous

## Governance and Golden References

All projects follow the golden standards defined in:
- `golden-template.md` – Repository structure
- `governance-handbook.md` – Rules and compliance (Tier 1/2/3)
- `naming-conventions.md` – File, variable, API naming
- `testing-standards.md` – Test coverage, quality gates

**When in doubt**:
- Prefer aligning with these golden docs over inventing new patterns
- If a golden doc conflicts with an existing repo-local pattern, surface the conflict and avoid unilateral changes

## Coding & Documentation Conventions

**Code Style**:
- Match the existing language, framework, and style found in the repo
- Use clear, descriptive names; avoid one-letter variables except for obvious counters or indices
- Do not add license headers or change licensing without explicit human approval

**Markdown**:
- Use `#`-style headings and bullet lists
- Keep sections focused; avoid massive monolithic sections
- Prefer links back to `golden-meta-docs` instead of copy-pasting large blocks

**Documentation**:
- Every public API must have docstring (Google style for Python, JSDoc for JavaScript)
- Update README.md when adding major features
- Update CHANGELOG.md for all releases

## Safety, Testing, and Automation

**Before Non-Trivial Changes**:
- Scan for existing tests or CI workflows and follow the established patterns
- Where tests exist, prefer:
  - Updating or adding focused tests near the code you change
  - Running the most specific test targets first (if available)

**Testing Guidelines**:
- Maintain 85%+ coverage
- Unit tests for business logic
- Integration tests for APIs
- E2E tests for critical user flows

**Automation**:
- Do not add heavy or long-running automation (e.g., broad network scans, full-suite fuzzing) unless explicitly requested
- For scripts that operate across multiple repos, keep configuration (paths, repo lists) in a central location (e.g., `config/ecosystem.yaml`) rather than hardcoding

## Monorepos and Layout

**If Monorepo**:
- Respect the existing top-level layout (e.g., `apps/`, `packages/`, `services/`, `infra/`)
- Keep cross-cutting changes coherent (e.g., updating shared package plus all direct consumers)
- Avoid splitting or merging major directories without explicit architectural approval

**If Single-Project Repo**:
- Keep all project-specific code within the expected root (e.g., `src/`, `app/`, `backend/`)
- Prefer adding configuration under `.github/`, `config/`, or `infra/` instead of scattering it

## Repo-Specific Notes

### Tech Stack
- **Backend**: Python 3.11 + FastAPI + PostgreSQL
- **Frontend**: Next.js 14 + TypeScript + TailwindCSS
- **Infrastructure**: Docker + Kubernetes + Terraform
- **AI**: OpenAI GPT-4 + Anthropic Claude + LangChain

### Key Entrypoints
- API: `src/api/main.py` (FastAPI app)
- CLI: `src/cli/main.py` (Click CLI)
- Agents: `src/agents/meta_orchestrator.py` (Agent coordinator)

### Critical Directories (Do Not Modify Without Discussion)
- `config/` - Configuration files
- `infrastructure/` - Terraform IaC
- `mcp/` - Model Context Protocol server
- `automation/agents/` - Autonomous agents

### Known Automation Scripts
- `scripts/setup.sh` - One-command setup
- `scripts/deploy.sh` - Deployment to production
- `scripts/backup.sh` - Daily backups (cron job)
- `scripts/analyze.py` - Code quality report

### Deviations from Golden Template
- Using PostgreSQL instead of SQLite (needed for production scale)
- Added custom `mcp/` directory for AI assistant integration
- Using `justfile` instead of `Makefile` (cross-platform support)
```

### Self-Refuting Statements Detection

#### Purpose
Identify logical contradictions in ideas, documentation, or agent outputs

#### Implementation

```python
# automation/agents/contradiction_detector.py

from typing import List, Tuple
import openai

class ContradictionDetector:
    """
    Detects self-refuting statements and logical contradictions.

    Examples of self-refuting statements:
    - "I never make absolute statements" (contradicts itself)
    - "This sentence is false" (Liar's paradox)
    - "You should never follow rules" (contradicts itself)
    - "AI will replace all jobs, including AI development" (self-defeating)
    """

    def __init__(self):
        self.client = openai.OpenAI()

    def detect_contradictions(self, text: str) -> List[Tuple[str, str]]:
        """
        Analyze text for contradictions.

        Returns:
            List of (statement1, statement2) pairs that contradict each other
        """

        # Extract claims from text
        claims = self._extract_claims(text)

        # Check each pair for contradiction
        contradictions = []
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if self._are_contradictory(claim1, claim2):
                    contradictions.append((claim1, claim2))

        return contradictions

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text using GPT-4."""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract all factual claims from the text. Return as a JSON list."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    def _are_contradictory(self, claim1: str, claim2: str) -> bool:
        """Check if two claims contradict each other using GPT-4."""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Determine if these two claims contradict each other. Respond with YES or NO."
                },
                {
                    "role": "user",
                    "content": f"Claim 1: {claim1}\nClaim 2: {claim2}"
                }
            ]
        )

        return response.choices[0].message.content.strip().upper() == "YES"

    def detect_self_refuting(self, statement: str) -> bool:
        """
        Check if a statement refutes itself.

        Examples:
        - "I always lie" (if true, then false)
        - "There are no absolute truths" (claims to be an absolute truth)
        - "You should never trust authorities" (claims authority)
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Determine if this statement is self-refuting (contradicts itself when you try to assert it). Respond with YES or NO, then explain."
                },
                {
                    "role": "user",
                    "content": statement
                }
            ]
        )

        analysis = response.choices[0].message.content

        return analysis.startswith("YES")


# Example usage
if __name__ == "__main__":
    detector = ContradictionDetector()

    # Test self-refuting statement
    statement = "This AI system will replace all AI developers"
    is_self_refuting = detector.detect_self_refuting(statement)

    if is_self_refuting:
        print(f"⚠️ Self-refuting statement detected: {statement}")

    # Test contradiction detection
    text = """
    Our platform guarantees 100% accuracy.
    Machine learning models are probabilistic and cannot guarantee perfect accuracy.
    """

    contradictions = detector.detect_contradictions(text)

    for claim1, claim2 in contradictions:
        print(f"❌ Contradiction found:")
        print(f"  Claim 1: {claim1}")
        print(f"  Claim 2: {claim2}")
```

---

## 📦 PLUGIN ECOSYSTEM & RESEARCH DATABASES

### Plugin Architecture

#### Plugin Interface

```python
# src/core/plugin_interface.py

from typing import Protocol, Dict, Any
from abc import abstractmethod

class Plugin(Protocol):
    """
    Standard interface for all plugins.

    All plugins must implement:
    - name: Unique identifier
    - version: Semantic version
    - initialize(): Setup
    - execute(): Main logic
    - cleanup(): Teardown
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version (e.g., '1.0.0')."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin with configuration.

        Called once when plugin is loaded.
        """
        pass

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Main plugin logic.

        Called every time plugin is invoked.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources.

        Called when plugin is unloaded.
        """
        pass


class ResearchDatabasePlugin(Plugin):
    """
    Plugin for integrating research databases.

    Supports:
    - arXiv
    - PubMed
    - Google Scholar
    - Semantic Scholar
    - IEEE Xplore
    - JSTOR
    - etc.
    """

    @abstractmethod
    def search(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search database and return papers."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        """Fetch specific paper by ID."""
        pass

    @abstractmethod
    def get_citations(self, paper_id: str) -> List[str]:
        """Get all papers that cite this paper."""
        pass

    @abstractmethod
    def get_references(self, paper_id: str) -> List[str]:
        """Get all papers referenced by this paper."""
        pass
```

#### Example Plugins

**1. arXiv Plugin**:
```python
# plugins/research_databases/arxiv_plugin.py

import arxiv
from typing import List, Dict, Any

class ArxivPlugin(ResearchDatabasePlugin):
    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        self.client = arxiv.Client()
        self.max_results = config.get("max_results", 100)

    def execute(self, input_data: Any) -> Any:
        if isinstance(input_data, str):
            return self.search(input_data, {})
        elif isinstance(input_data, dict):
            return self.search(input_data["query"], input_data.get("filters", {}))

    def search(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        search = arxiv.Search(
            query=query,
            max_results=filters.get("max_results", self.max_results),
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for result in self.client.results(search):
            results.append({
                "id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "published": result.published.isoformat(),
                "url": result.entry_id,
                "pdf_url": result.pdf_url
            })

        return results

    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        search = arxiv.Search(id_list=[paper_id])
        result = next(self.client.results(search))

        return {
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "full_text": self._download_pdf(result.pdf_url)
        }

    def cleanup(self) -> None:
        pass  # No cleanup needed
```

**2. Semantic Scholar Plugin**:
```python
# plugins/research_databases/semantic_scholar_plugin.py

import requests
from typing import List, Dict, Any

class SemanticScholarPlugin(ResearchDatabasePlugin):
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    @property
    def name(self) -> str:
        return "semantic-scholar"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        self.api_key = config.get("api_key")
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}

    def search(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{self.BASE_URL}/paper/search",
            params={
                "query": query,
                "limit": filters.get("max_results", 100),
                "fields": "paperId,title,abstract,authors,year,citationCount,url"
            },
            headers=self.headers
        )

        return response.json()["data"]

    def get_citations(self, paper_id: str) -> List[str]:
        response = requests.get(
            f"{self.BASE_URL}/paper/{paper_id}/citations",
            params={"fields": "paperId,title"},
            headers=self.headers
        )

        return [
            citation["citingPaper"]["paperId"]
            for citation in response.json()["data"]
        ]

    def get_references(self, paper_id: str) -> List[str]:
        response = requests.get(
            f"{self.BASE_URL}/paper/{paper_id}/references",
            params={"fields": "paperId,title"},
            headers=self.headers
        )

        return [
            ref["citedPaper"]["paperId"]
            for ref in response.json()["data"]
        ]
```

### Plugin Registry

```python
# src/core/plugin_registry.py

from typing import Dict, Type
from pathlib import Path
import importlib.util

class PluginRegistry:
    """
    Central registry for all plugins.

    Auto-discovers plugins from plugins/ directory.
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Auto-discover plugins from plugins/ directory."""

        plugins_dir = Path(__file__).parent.parent.parent / "plugins"

        for plugin_file in plugins_dir.rglob("*_plugin.py"):
            # Import module
            spec = importlib.util.spec_from_file_location(
                plugin_file.stem, plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find Plugin classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) and
                    issubclass(attr, Plugin) and
                    attr is not Plugin
                ):
                    plugin = attr()
                    self.plugins[plugin.name] = plugin
                    print(f"✅ Loaded plugin: {plugin.name} v{plugin.version}")

    def get_plugin(self, name: str) -> Plugin:
        """Get plugin by name."""
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all available plugins."""
        return list(self.plugins.keys())
```

### Research Database Integrations

#### Supported Databases

```yaml
# config/research-databases.yaml
databases:
  arxiv:
    enabled: true
    base_url: "http://export.arxiv.org/api/query"
    rate_limit: 3_requests_per_second

  pubmed:
    enabled: true
    base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    api_key: "${PUBMED_API_KEY}"

  semantic_scholar:
    enabled: true
    base_url: "https://api.semanticscholar.org/graph/v1"
    api_key: "${SEMANTIC_SCHOLAR_API_KEY}"

  crossref:
    enabled: true
    base_url: "https://api.crossref.org/works"

  unpaywall:
    enabled: true
    base_url: "https://api.unpaywall.org/v2"
    email: "${UNPAYWALL_EMAIL}"

search_strategy:
  parallel: true
  deduplicate: true
  merge_results: true
  max_results_per_db: 100
```

---

## 🤖 COMPLETE AUTOMATION STACK

### Automation Layers

```
Layer 4: Business Intelligence
  ↑ (insights, dashboards, alerts)
Layer 3: Process Automation (Zapier, n8n)
  ↑ (workflows, integrations)
Layer 2: AI Agents (Autonomous)
  ↑ (code maintenance, monitoring)
Layer 1: Infrastructure (CI/CD, Kubernetes)
  ↑ (deployments, scaling)
Layer 0: Foundation (GitHub, Cloud)
```

### Complete Automation Workflows

#### Workflow 1: Automated Onboarding

```yaml
# automation/workflows/onboarding.yaml
name: Automated User Onboarding

trigger:
  type: webhook
  endpoint: /api/webhooks/user-signup

steps:
  - name: Create User Account
    agent: executor
    action: create_database_record
    inputs:
      table: users
      data: ${trigger.payload}

  - name: Send Welcome Email
    agent: synthesizer
    action: generate_email
    inputs:
      template: welcome
      personalization: ${user.name}
      send_via: resend

  - name: Setup User Workspace
    agent: executor
    action: provision_resources
    inputs:
      user_id: ${user.id}
      resources: [dashboard, api_key, storage]

  - name: Schedule Follow-Up
    agent: planner
    action: schedule_task
    inputs:
      task: send_onboarding_checklist
      delay: 24_hours

  - name: Track Conversion
    agent: router
    action: log_event
    inputs:
      event: user_onboarded
      analytics: [google, amplitude]
```

#### Workflow 2: Code Quality Automation

```yaml
# automation/workflows/code-quality.yaml
name: Automated Code Quality Maintenance

trigger:
  type: schedule
  cron: "0 0 * * *"  # Daily at midnight

steps:
  - name: Run Code Analysis
    agent: executor
    action: run_command
    inputs:
      command: python scripts/analyze.py --json

  - name: Identify Issues
    agent: critic
    action: analyze_report
    inputs:
      report: ${analysis.output}
      threshold: critical

  - name: Auto-Fix Simple Issues
    agent: executor
    action: apply_fixes
    inputs:
      issues: ${critic.fixable_issues}
      create_pr: true

  - name: Alert Team
    condition: ${critic.critical_issues > 0}
    agent: router
    action: send_slack
    inputs:
      channel: #engineering
      message: "🚨 ${critic.critical_issues} critical issues found"

  - name: Update Metrics
    agent: memory-keeper
    action: store_metrics
    inputs:
      date: ${today}
      metrics: ${analysis.summary}
```

#### Workflow 3: Research Paper Automation

```yaml
# automation/workflows/research-pipeline.yaml
name: Automated Research Paper Pipeline

trigger:
  type: schedule
  cron: "0 9 * * 1"  # Monday 9 AM

steps:
  - name: Fetch New Papers
    agent: researcher
    action: search_databases
    inputs:
      databases: [arxiv, pubmed, semantic_scholar]
      query: ${research_interests}
      since_date: ${last_week}

  - name: Filter Relevant Papers
    agent: critic
    action: evaluate_relevance
    inputs:
      papers: ${researcher.papers}
      criteria: ${relevance_criteria}
      threshold: 0.7

  - name: Generate Summaries
    agent: synthesizer
    action: summarize_papers
    inputs:
      papers: ${critic.relevant_papers}
      style: executive_summary
      max_length: 300_words

  - name: Create Newsletter
    agent: insight-synthesizer
    action: generate_newsletter
    inputs:
      summaries: ${synthesizer.summaries}
      format: html
      template: weekly_research_digest

  - name: Send to Subscribers
    agent: executor
    action: send_email
    inputs:
      newsletter: ${insight_synthesizer.output}
      list: research_subscribers
      via: resend

  - name: Archive Papers
    agent: memory-keeper
    action: store_papers
    inputs:
      papers: ${researcher.papers}
      database: vector_db
      index: research_archive
```

---

## 📚 USAGE EXAMPLES

### Example 1: Generate Idea from Scratch

```python
from ultrathink import UltraThinkIdeaGenerator

# Initialize generator
generator = UltraThinkIdeaGenerator()

# Generate 10 crazy ideas
ideas = generator.generate_ideas(
    content="""
    Problem: Scientific research is slow and siloed.
    Researchers work in isolation, duplicate efforts, and
    can't find collaborators easily.
    """,
    mode="crazy",
    count=10
)

# Each idea comes with complete spec
for idea in ideas:
    print(f"\n{'='*50}")
    print(f"IDEA: {idea.title}")
    print(f"Score: {idea.score}/100")
    print(f"Novelty: {idea.novelty}/100")
    print(f"Feasibility: {idea.feasibility}/100")
    print(f"\nDescription:")
    print(idea.description)
    print(f"\nTech Stack:")
    for tech in idea.spec.tech_stack:
        print(f"  - {tech}")
    print(f"\nTimeline: {idea.spec.timeline}")
    print(f"Cost Estimate: ${idea.spec.cost_estimate:,}")
```

### Example 2: Validate Existing Idea

```python
from ultrathink import UltraThinkIdeaGenerator

generator = UltraThinkIdeaGenerator()

# Existing idea
my_idea = """
AI-powered research assistant that automatically reads papers,
extracts insights, and suggests new research directions.
"""

# Run through validation
validation = generator.validator.test([my_idea])[0]

print(f"Survival Score: {validation.survival_score}/100")
print(f"\nCritiques:")
for critique in validation.critiques:
    print(f"  - {critique}")

print(f"\nRecommendations:")
for rec in validation.recommendations:
    print(f"  ✓ {rec}")
```

### Example 3: Deploy Idea as Project

```python
from ultrathink import UltraThinkIdeaGenerator
from pathlib import Path

generator = UltraThinkIdeaGenerator()

# Generate idea
ideas = generator.generate_ideas(
    content="Make scientific collaboration more fun",
    mode="fun",
    count=1
)

best_idea = ideas[0]

# Generate complete repository
output_dir = Path("/mnt/c/Users/mesha/Desktop/GitHub")
repo_path = generator.deploy_as_project(
    idea=best_idea,
    output_dir=output_dir,
    init_git=True,
    create_github_repo=True
)

print(f"✅ Project created at: {repo_path}")
print(f"📦 Repository: https://github.com/your-org/{best_idea.repo_name}")
print(f"\nNext steps:")
print(f"1. cd {repo_path}")
print(f"2. ./scripts/setup.sh")
print(f"3. just dev")
```

---

## 🎯 CONCLUSION

This **UltraThink Idea Generator Master System** provides:

1. **15 Thinking Frameworks** for multi-perspective analysis
2. **17 Specialized AI Agents** for autonomous operation
3. **Complete Golden Template** for production-ready deployment
4. **Development Workflow** with git, branches, CI/CD
5. **MCP Integration** for AI assistant access
6. **Plugin Ecosystem** for research databases
7. **Full Automation Stack** from infrastructure to business intelligence
8. **Self-Refuting Detection** to avoid logical contradictions
9. **Comprehensive Documentation** following SSOT principles

### Key Innovations

- **Algorithmic Creativity**: Systematic generation of novel ideas
- **Multi-Agent Orchestration**: 17 specialized agents working in concert
- **Production-Ready Output**: Every idea becomes deployable code
- **Validation-First**: Ideas tested before development
- **Fail-Fast Philosophy**: Kill bad ideas early, double down on winners

### Next Actions

1. **Deploy System**: Run `python golden.py create-template ...`
2. **Generate Ideas**: Use UltraThinkIdeaGenerator on your content
3. **Validate Ideas**: Run through adversarial validation
4. **Build Winner**: Deploy highest-scoring idea as project
5. **Iterate**: Use meta-learner to improve system over time

---

**Version**: 2.0.0
**Last Updated**: 2025-11-15
**Status**: ✅ Production-Ready
**License**: MIT
**Author**: Synthesized from 20+ sources via UltraThink Algorithm
