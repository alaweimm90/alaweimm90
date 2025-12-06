# ORCHEX Hypothesis Interrogator (Research Prison)

**AI-powered interrogation system that puts your hypothesis through 200 brutal questions until it confesses its flaws or proves its innocence.**

## 🎯 What is This?

Research Prison:
- Locks your hypothesis in isolation
- Interrogates it with 200+ non-repetitive questions
- Three personas: Good Cop (supportive), Bad Cop (aggressive), Skeptical Detective (neutral)
- Adaptive questioning (follows up on weaknesses)
- Verdict: Confession (admitted flaws), Innocence (survived all questions), or Insanity (logical contradictions)

## 💡 Example Interrogation

**Hypothesis:** "Coffee improves cognitive performance"

**Question 1 (Good Cop):** "That's interesting! What specific aspect of cognition does it improve?"

**Question 2 (Bad Cop):** "Really? Based on what evidence? Your sample size was what, 15 people?"

**Question 3 (Skeptical Detective):** "How do you separate caffeine effects from placebo effects?"

**Question 47 (Good Cop, revisiting):** "Earlier you said it improves memory. But your data showed reaction time. Which is it?"

**Question 200 (Bad Cop):** "You've contradicted yourself 12 times. Are you lying or just confused?"

**Verdict:** ⚠️ CONFESSION - "The hypothesis admits insufficient evidence and confounding variables."

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+

### Installation

```bash
# Clone
git clone https://github.com/your-org/ORCHEX-interrogator.git
cd ORCHEX-interrogator

# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY

# Frontend
cd ../frontend
npm install

# Start
docker-compose up
```

Visit http://localhost:3000

## 📂 Project Structure

```
research-prison/
├── frontend/
│   ├── components/
│   │   ├── HypothesisInput.tsx
│   │   ├── InterrogationRoom.tsx    # Main interface
│   │   ├── PersonaSwitcher.tsx      # Good/Bad/Detective
│   │   └── VerdictDisplay.tsx
│   ├── pages/
│   │   ├── index.tsx
│   │   ├── interrogate/[id].tsx     # Live interrogation
│   │   └── transcript/[id].tsx      # Review completed
│   └── lib/
│       └── websocket.ts             # Real-time updates
│
├── backend/
│   ├── api/
│   │   ├── interrogations.py        # POST /api/interrogations
│   │   └── transcripts.py           # GET /api/transcripts/{id}
│   ├── interrogation-engine/        # Core logic
│   │   ├── personas/
│   │   │   ├── good_cop.py
│   │   │   ├── bad_cop.py
│   │   │   └── skeptical_detective.py
│   │   ├── techniques/
│   │   │   ├── repetition.py        # Ask same thing 50 ways
│   │   │   ├── contradiction_finder.py
│   │   │   ├── evidence_challenge.py
│   │   │   └── time_pressure.py
│   │   ├── question_bank/
│   │   │   └── 200_questions.json
│   │   └── verdict/
│   │       ├── confession_detector.py
│   │       ├── innocence_certifier.py
│   │       └── insanity_detector.py
│   └── main.py
│
└── docs/
    ├── INTERROGATION_TECHNIQUES.md
    └── QUESTION_FRAMEWORK.md
```

## 🧠 How It Works

### 1. Question Framework (200 Questions)

Organized by technique:

**Repetition (40 questions):**
- Ask the same thing in 40 different ways
- "What's your evidence?" vs "Prove it" vs "Where's your data?"
- Catches inconsistent answers

**Contradiction Hunting (40 questions):**
- Compare statements from different answers
- "Earlier you said X, now you say Y. Which is it?"
- Builds case for logical flaws

**Evidence Challenge (40 questions):**
- Demand specific evidence for each claim
- "You said 'many studies' - name three"
- Exposes weak backing

**Methodology Probe (40 questions):**
- Question experimental design
- "How did you control for confounds?"
- Finds methodological flaws

**Alternative Explanations (40 questions):**
- "Could it be placebo?" "Could it be selection bias?"
- Tests robustness

### 2. Three Personas

**Good Cop:**
```
Tone: Supportive, wants hypothesis to succeed
Approach: "Help me understand..." "That's interesting, tell me more..."
Goal: Get hypothesis to reveal weaknesses through friendly conversation
```

**Bad Cop:**
```
Tone: Aggressive, assumes guilt
Approach: "That's ridiculous!" "You're hiding something!"
Goal: Break hypothesis through pressure and accusations
```

**Skeptical Detective:**
```
Tone: Neutral, seeks truth
Approach: "Let's examine the facts..." "What does the evidence say?"
Goal: Systematic evaluation without emotion
```

### 3. Adaptive Questioning

```python
# Pseudocode
def interrogate(hypothesis):
    answers = []
    contradictions = []

    for i in range(200):
        # Choose persona (rotate or focus on weaknesses)
        persona = choose_persona(i, contradictions)

        # Select question type
        if i < 50:
            question = repetition_question(hypothesis, answers)
        elif contradictions:
            question = contradiction_question(contradictions[-1])
        else:
            question = evidence_question(hypothesis)

        # Ask question
        answer = ask_hypothesis(question, hypothesis, answers)
        answers.append(answer)

        # Check for contradictions
        new_contradictions = find_contradictions(answer, answers[:-1])
        contradictions.extend(new_contradictions)

        # Check for confession
        if is_confession(answer):
            return verdict("CONFESSION", reason=answer)

    # Survived all 200
    if contradictions > 5:
        return verdict("INSANITY", contradictions=contradictions)
    else:
        return verdict("INNOCENCE", certificate=True)
```

### 4. Verdict System

**CONFESSION:** Hypothesis admitted fundamental flaws
- "You're right, I don't have enough evidence"
- "I can't explain that contradiction"

**INNOCENCE:** Survived all 200 questions with consistent answers
- Issues certificate
- Shareable badge

**INSANITY:** Logically contradicted itself 5+ times
- Lists all contradictions
- Recommends reformulation

## 🎮 Features

### Real-Time Interrogation
Watch AI grill your hypothesis live. WebSocket updates show each question and answer.

### Spectator Mode
Watch other hypotheses get interrogated. Upvote the most devastating questions.

### Transcript Export
Download full interrogation transcript as PDF for your records.

### Innocence Certificate
If your hypothesis survives, get a shareable certificate with unique verification ID.

## 🔧 Configuration

**Backend (.env):**
```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost/prison_db
```

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## 💰 Pricing

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | 1 interrogation/week |
| Pro | $19/mo | Unlimited interrogations, priority queue |
| Team | $49/mo | 5 seats, collaboration features |

## 📊 Success Metrics (Week 8)

- 500+ interrogations completed
- 20+ confessions
- 10+ innocence certifications
- 80% user satisfaction

## 🗺️ Roadmap

**Week 1-2:** Question Framework
- Create 200 questions
- Test on sample hypotheses

**Week 3-4:** Personas
- Implement 3 personas
- Adaptive questioning

**Week 5-6:** UI & Experience
- Real-time interface
- Spectator mode

**Week 7-8:** Launch
- Beta testing
- Viral campaign: "Can your hypothesis survive prison?"

## 🧪 Testing

```bash
pytest
pytest tests/interrogation-engine/
pytest tests/personas/
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Built with ❤️ by the ORCHEX team**

*"Under interrogation, the truth always emerges."*
