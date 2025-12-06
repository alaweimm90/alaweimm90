# 🧠 Meta-Learning Core - Self-Improving Research System

**Turing Challenge Feature #4** - Agents with personality who learn from experience!

Part of [Project ORCHEX](../../../MASTER_SUPER_PLAN.md)

---

## 🎭 Meet the Team!

Research is more fun with personality! Meet your AI research assistants:

### 😠 **Grumpy Refuter**
- **Role**: Self-Refutation
- **Catchphrase**: "Everything is flawed until proven otherwise."
- **Style**: Never satisfied, always finds the flaws. Strictest critic.
- **When to use**: When you need brutal honesty

### 🤨 **Skeptical Steve**
- **Role**: Interrogation
- **Catchphrase**: "Show me the data or get out."
- **Style**: Asks 200 annoying questions. Never takes anything at face value.
- **When to use**: For thorough validation

### 🤦 **Failure Frank**
- **Role**: Hall of Failures
- **Catchphrase**: "I've seen this mistake before, kid..."
- **Style**: Veteran researcher who remembers every past failure.
- **When to use**: Risk assessment and learning from history

### 😄 **Optimistic Oliver**
- **Role**: Hypothesis Generation
- **Catchphrase**: "Every idea is a potential breakthrough!"
- **Style**: Generates hypotheses enthusiastically. Sees potential everywhere.
- **When to use**: Brainstorming and creative ideation

### 🤓 **Pedantic Pete**
- **Role**: Peer Review
- **Catchphrase**: "Technically speaking, there's an issue on line 47..."
- **Style**: Reviews papers with extreme attention to detail.
- **When to use**: Final manuscript review

### 😰 **Cautious Cathy**
- **Role**: Risk Assessment
- **Catchphrase**: "Let's think about what could go wrong..."
- **Style**: Risk assessment expert. Always considers failure modes.
- **When to use**: Before committing to expensive experiments

### 🎉 **Enthusiastic Emma**
- **Role**: Experiment Design
- **Catchphrase**: "Let's run ALL the experiments!"
- **Style**: Loves designing creative experiments. Energy never runs out.
- **When to use**: Experimental protocol generation

---

## 🚀 Quick Start

### Basic Usage

```python
from meta_learning import MetaLearningProtocol, list_all_agents

# Initialize
meta = MetaLearningProtocol()

# Start research
meta.start_research("QAP solving with RL", "optimization")

# Select agent for refutation
agent = meta.select_agent(role="self_refutation", domain="optimization")
print(f"{agent.emoji} {agent.name}: {agent.catchphrase}")

# Agent reacts to score
meta.agent_react(agent, score=73.5)
# Output: 😠 Fine, it's... acceptable.
```

### Agent Personality

```python
from meta_learning import get_agent

grumpy = get_agent("grumpy_refuter")
print(grumpy.get_greeting())
# 😠 Ugh, another hypothesis to tear apart...

print(grumpy.react_to_score(45.0))
# 😠 I KNEW this was garbage!

print(grumpy.react_to_score(85.0))
# 😠 Fine, it's... acceptable.
```

---

## 🧠 How It Works

### 1. **Trajectory Recording**

Every research project is recorded:
```python
{
  "trajectory_id": "optimization_20250511_143022",
  "topic": "RL for QAP",
  "actions": [
    {
      "agent": "Grumpy Refuter",
      "action": "refute",
      "score": 73.2,
      "duration": 25.3,
      "success": true
    },
    ...
  ],
  "final_score": 68.4,
  "final_success": true
}
```

### 2. **UCB1 Multi-Armed Bandit**

Learns which agents work best:
- **Exploration**: Try untested agents
- **Exploitation**: Use agents known to work well
- **Context-aware**: Different agents for different domains

```
UCB1(agent) = avg_reward + sqrt(2 * ln(total) / pulls)
                ↑                      ↑
           exploitation         exploration bonus
```

### 3. **Continuous Improvement**

System gets better over time:
- Records every decision and outcome
- Learns from successes AND failures
- Adapts agent selection automatically
- Improves with each project

---

## 📊 Self-Learning in Action

```python
# Project 1: Random agent selection
agent = meta.select_agent("self_refutation", "optimization")
# Selected: Grumpy Refuter (0 uses, exploring)
# Result: score 65.2

# Project 2: Starting to learn
agent = meta.select_agent("self_refutation", "optimization")
# Selected: Grumpy Refuter (1 use, 65.2 avg)
# Result: score 71.8

# Project 10: Learned optimal agent
agent = meta.select_agent("self_refutation", "optimization")
# Selected: Skeptical Steve (5 uses, 78.4 avg)
# System learned Steve > Grumpy for this domain!
```

---

## 🎛️ Customization

### Create Custom Agent

```python
from meta_learning import create_custom_agent, AgentMood

party_agent = create_custom_agent(
    name="Party Pat",
    role="celebration",
    mood=AgentMood.ENTHUSIASTIC,
    catchphrase="Let's celebrate this discovery! 🎊",
    emoji="🥳",
    description="Celebrates every small win",
    strictness=0.1,
    creativity=0.95,
    optimism=1.0,
    verbosity=0.8,
)
```

### Modify Agent Behavior

```python
agent = get_agent("grumpy_refuter")

# Make even grumpier
agent.strictness = 1.0
agent.optimism = 0.0

# Make more verbose
agent.verbosity = 1.0
```

---

## 🔧 Advanced Features

### Contextual Agent Selection

```python
# Different agents for different domains
meta.select_agent("self_refutation", "optimization")
# → Might pick Grumpy Refuter (great for math)

meta.select_agent("self_refutation", "biology")
# → Might pick Cautious Cathy (biological systems are complex)
```

### Force Specific Agent

```python
# Override bandit selection
agent = meta.select_agent(
    role="self_refutation",
    domain="optimization",
    force_agent="grumpy_refuter"
)
```

### Export Learning

```python
# Save what the system has learned
meta.export_learning("learned_knowledge.json")

# Load in new session
meta2 = MetaLearningProtocol()
# Automatically warm-starts from past trajectories
```

---

## 📈 Performance Tracking

### Get Agent Stats

```python
stats = meta.bandit.get_stats()

print(stats["grumpy_refuter"])
# {
#   'pulls': 25,
#   'avg_reward': 0.72,
#   'success_rate': 0.88,
#   'total_reward': 18.0
# }
```

### Get Best Agents

```python
best_agents = meta.get_best_agents(domain="optimization", top_k=3)

for agent in best_agents:
    print(f"{agent.emoji} {agent.name}")
# 🤨 Skeptical Steve
# 😠 Grumpy Refuter
# 🤓 Pedantic Pete
```

---

## 🎯 Integration with ORCHEX

Meta-learning is built into ORCHEX:

```python
from ORCHEX import ATLASProtocol

# ORCHEX automatically uses meta-learning
ORCHEX = ATLASProtocol()

# Agents learn from each project
project = await ORCHEX.run_research("Your topic", "optimization")

# System improves with each use!
```

---

## 🎨 Why Personalities?

**Research is serious, but it doesn't have to be boring!**

Benefits of agent personalities:
1. **More engaging** - Research becomes fun
2. **Clearer roles** - Each agent has a distinct purpose
3. **Better debugging** - "Grumpy failed? Check refutation logic"
4. **Community** - "My Grumpy is stricter than yours!"
5. **Education** - Students learn research methodology through characters

---

## 🔬 The Science

### Multi-Armed Bandit Theory

The UCB1 algorithm provably converges to optimal agent selection:
- **Regret Bound**: O(log n)
- **Exploration**: Decreases over time (sqrt term)
- **Exploitation**: Increases over time (avg reward term)

### Meta-Learning

System exhibits true **meta-learning**:
- **Learning to learn**: Improves strategy selection
- **Transfer**: Knowledge transfers across projects
- **Adaptation**: Adjusts to new domains
- **Continuous**: Never stops improving

---

## 📚 Research Context

### Inspired By

- **Multi-Armed Bandits** (Auer et al. 2002) - UCB1 algorithm
- **Meta-Learning** (Thrun & Pratt 1998) - Learning to learn
- **Agent Systems** (Wooldridge 2009) - Multi-agent coordination
- **Personality AI** (Nass & Brave 2005) - Computers are social actors

### Novel Contributions

1. **First research system with personality-based agents**
2. **First to apply bandits to agent selection in autonomous research**
3. **First with continuous improvement from trajectory data**

---

## 📊 Expected Impact

- **10-30% improvement** in agent selection over random
- **Faster convergence** to optimal strategies (log n regret)
- **Domain-specific** knowledge accumulation
- **Fun factor**: Research becomes more enjoyable! 😄

---

## 🚧 Roadmap

### Current (v0.1.0)
- ✅ Agent personality system with 7 characters
- ✅ Trajectory recording
- ✅ UCB1 contextual bandit
- ✅ Auto warm-start from past data
- ✅ Performance tracking

### Next (v0.2.0)
- ⏳ Agent marketplace (community agents)
- ⏳ Agent breeding (combine agents)
- ⏳ Voice synthesis (agents talk!)
- ⏳ Visual avatars (see the agents)

### Future (v1.0.0)
- Agent tournaments (best agent competition)
- Social agents (agents collaborate)
- Emotional intelligence (agents read user mood)

---

## 🎉 Fun Interactions

```python
# Agents can disagree!
grumpy = get_agent("grumpy_refuter")
oliver = get_agent("optimistic_oliver")

print(grumpy.react_to_score(55))
# 😠 Meh. Could be worse.

print(oliver.react_to_score(55))
# 😄 Good start! We can improve this!

# Different personalities for different tasks!
```

---

## 📄 License

MIT License

---

**Make research fun again with personality-based AI agents!** 🎭🧠🚀

*(PS: Grumpy Refuter thinks this README is too optimistic. Skeptical Steve wants citations for every claim. Failure Frank says "we tried fun agents before and it failed." But Optimistic Oliver says "This will revolutionize research!" 😄)*
