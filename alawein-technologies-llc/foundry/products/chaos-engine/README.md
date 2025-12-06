# ORCHEX Chaos Engine

**Cross-domain hypothesis generator that creates breakthrough ideas by colliding unrelated fields.**

## 🎯 What is This?

Chaos Engine:
- Randomly collides 100+ scientific domains (physics × poetry, biology × economics)
- Finds deep structural analogies between fields
- Generates testable hypotheses from unlikely combinations
- Filters for feasibility (1% genius, 99% noise)
- Gamified: "Weekly Chaos Challenge," leaderboards, viral sharing

## 💡 Examples

**Input:** "Quantum Mechanics" × "Social Media"
**Output:** "What if information can exist in superposition? Users could post content that shows different versions to different viewers based on quantum observation - the act of viewing determines which reality manifests."

**Input:** "Protein Folding" × "Urban Planning"
**Output:** "Cities fold like proteins - low-energy states (efficient transport) vs. misfolded states (traffic). Apply annealing algorithms from computational biology to traffic optimization."

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Docker (optional)

### Installation

```bash
# Clone
git clone https://github.com/your-org/ORCHEX-chaos-engine.git
cd ORCHEX-chaos-engine

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY

# Frontend setup
cd ../frontend
npm install
cp .env.example .env.local

# Start everything
docker-compose up
```

Visit http://localhost:3000

## 📂 Project Structure

```
chaos-engine/
├── frontend/                    # Next.js + React
│   ├── components/
│   │   ├── DomainSelector.tsx   # Choose 2 domains
│   │   ├── HypothesisCard.tsx   # Display generated hypothesis
│   │   ├── ChaosRoulette.tsx    # Random collision button
│   │   └── LeaderboardWidget.tsx
│   ├── pages/
│   │   ├── index.tsx            # Landing page
│   │   ├── collide.tsx          # Main collision interface
│   │   └── weekly-challenge.tsx # Gamification
│   └── lib/
│       ├── api.ts
│       └── domain-data.ts       # 100+ domains
│
├── backend/                     # FastAPI + Python
│   ├── api/
│   │   ├── collisions.py        # POST /api/collisions
│   │   ├── domains.py           # GET /api/domains
│   │   └── voting.py            # POST /api/vote (genius vs nonsense)
│   ├── collision-engine/        # Core logic
│   │   ├── domain_embeddings.py # Encode domains as vectors
│   │   ├── collision_generator.py
│   │   ├── analogy_finder.py    # Cross-domain pattern matching
│   │   └── feasibility_scorer.py
│   ├── ml-models/
│   │   ├── hypothesis_generator.py  # GPT-4 wrapper
│   │   └── genius_detector.py       # Filter nonsense
│   ├── database/
│   │   ├── domains.json         # 100+ domain definitions
│   │   └── models.py
│   └── main.py
│
├── docs/
│   ├── DOMAIN_DATABASE.md       # How domains are structured
│   ├── COLLISION_ALGORITHM.md   # Technical details
│   └── ANALOGY_THEORY.md        # Theoretical foundation
│
└── tests/
```

## 🧠 How It Works

### 1. Domain Database (100+ Domains)

Each domain has:
```json
{
  "name": "Quantum Mechanics",
  "category": "Physics",
  "key_concepts": [
    {"name": "Superposition", "description": "..."},
    {"name": "Entanglement", "description": "..."}
  ],
  "principles": ["Uncertainty principle", "Wave-particle duality"],
  "methodologies": ["Schrödinger equation", "Path integrals"],
  "famous_examples": ["Double-slit experiment"]
}
```

### 2. Collision Types

**Random Collision:** Pick 2 domains at random
**Guided Collision:** Maximize domain distance (most different fields)
**Analogical Collision:** Find structural similarities

### 3. Hypothesis Generation

```python
# Pseudocode
def generate_hypothesis(domain_a, domain_b):
    # Find analogical concepts
    analogies = find_deep_analogies(domain_a, domain_b)

    # Generate hypothesis using GPT-4
    prompt = f"""
    Find a deep structural analogy between {domain_a} and {domain_b}.

    Analogies found:
    {analogies}

    Generate a testable hypothesis that applies insights from
    {domain_a} to solve problems in {domain_b}.

    Be creative but rigorous.
    """

    hypothesis = gpt4(prompt)

    # Score feasibility
    feasibility = score_feasibility(hypothesis)

    # Only return if feasibility > 60%
    if feasibility > 0.6:
        return hypothesis
```

### 4. Feasibility Scoring

Uses ML to filter:
- **Testability:** Can this be experimentally verified?
- **Resource Requirements:** Do we have the technology?
- **Existing Research:** Has this been tried?
- **Logical Coherence:** Does it make sense?

Score 0-100. Only show hypotheses scoring >60.

## 🎮 Gamification Features

### Weekly Challenges
"Connect Poetry with Neuroscience" - Best hypothesis wins!

### Leaderboard
- Most creative collisions
- Most upvoted hypotheses
- Highest feasibility scores

### Accidental Nobel Archive
Best ideas that could actually work

### Social Sharing
"I just discovered that traffic patterns follow protein folding rules! 🤯 #ChaosEngine"

## 🔧 Configuration

### Environment Variables

**Backend:**
```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost/chaos_db
REDIS_URL=redis://localhost:6379
```

**Frontend:**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🧪 Testing

```bash
pytest
pytest tests/collision-engine/  # Test collision algorithm
pytest tests/analogy-finder/    # Test analogy detection
```

## 💰 Pricing

**Free Product** (monetize later with ads/premium features)
- Unlimited collisions
- Community voting
- Weekly challenges

**Future Premium ($9/mo):**
- Custom domain uploads
- API access
- No ads

## 📊 Success Metrics

**Week 10 Targets:**
- 1,000 registered users
- 10,000+ hypotheses generated
- 100+ hypotheses voted "genius"
- 1 hypothesis inspires actual research paper

## 🗺️ Roadmap

### Week 1-2: Domain Database ✅
- Collect 100 domains
- Define structure
- Create embeddings

### Week 3-4: Collision Engine
- Random collision
- Guided collision
- Analogy finder

### Week 5-6: Hypothesis Generation
- GPT-4 integration
- Feasibility scoring
- Quality filter

### Week 7-8: UI & Gamification
- Domain selector
- Hypothesis feed
- Voting system
- Weekly challenges

### Week 9-10: Launch
- Beta testing
- Product Hunt launch
- Viral campaign

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License

---

**Built with ❤️ by the ORCHEX team**

*"The best discoveries happen at the intersection of unrelated fields."*
