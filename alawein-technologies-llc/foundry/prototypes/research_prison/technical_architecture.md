# RESEARCH PRISON - Technical Architecture

## System Diagram

```
[Research Paper URL/Text] → [Document Parser] → [Interrogation Engine]
                                    ↓
                        [Question Generation System]
                                    ↓
                        [Multi-Agent AI Interrogators]
                                    ↓
                        [Weakness Detection Engine]
                                    ↓
                        [Report Generator] → [Validity Score + Critique]
```

## Database Schema

```sql
-- Main research papers table
CREATE TABLE research_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    authors TEXT[],
    abstract TEXT,
    full_text TEXT,
    url VARCHAR(1000),
    submission_date TIMESTAMP DEFAULT NOW(),
    field VARCHAR(200),
    status VARCHAR(50) DEFAULT 'pending'
);

-- Interrogation sessions
CREATE TABLE interrogation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID REFERENCES research_papers(id),
    session_start TIMESTAMP DEFAULT NOW(),
    session_end TIMESTAMP,
    total_questions INTEGER DEFAULT 0,
    validity_score DECIMAL(3,2),
    status VARCHAR(50) DEFAULT 'in_progress'
);

-- Individual interrogation rounds
CREATE TABLE interrogation_rounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES interrogation_sessions(id),
    round_number INTEGER,
    interrogator_type VARCHAR(100), -- 'methodology', 'statistics', 'logic', etc.
    question TEXT NOT NULL,
    response TEXT,
    weakness_found BOOLEAN DEFAULT FALSE,
    severity_level INTEGER CHECK (severity_level BETWEEN 1 AND 5),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Detected weaknesses and flaws
CREATE TABLE detected_weaknesses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES interrogation_sessions(id),
    round_id UUID REFERENCES interrogation_rounds(id),
    weakness_type VARCHAR(100),
    description TEXT,
    evidence TEXT,
    impact_score DECIMAL(3,2),
    suggested_fix TEXT
);

-- Final reports
CREATE TABLE interrogation_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES interrogation_sessions(id),
    paper_id UUID REFERENCES research_papers(id),
    final_verdict VARCHAR(50), -- 'valid', 'questionable', 'flawed', 'invalid'
    validity_score DECIMAL(3,2),
    major_weaknesses JSONB,
    minor_issues JSONB,
    recommendations TEXT,
    generated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_papers_status ON research_papers(status);
CREATE INDEX idx_sessions_paper ON interrogation_sessions(paper_id);
CREATE INDEX idx_rounds_session ON interrogation_rounds(session_id);
CREATE INDEX idx_weaknesses_session ON detected_weaknesses(session_id);
```

## API Endpoints

```
POST /api/papers/submit - Submit a research paper for interrogation
  Body: { url: string, text?: string, metadata?: object }
  Returns: { paper_id: string, session_id: string }

GET /api/sessions/{session_id} - Get current interrogation status
  Returns: { status: string, progress: number, current_round: object }

POST /api/sessions/{session_id}/interrogate - Start/continue interrogation
  Body: { intensity: 'light'|'moderate'|'aggressive' }
  Returns: { round_results: array, weaknesses_found: array }

GET /api/reports/{session_id} - Get final interrogation report
  Returns: { validity_score: number, verdict: string, weaknesses: array, report: object }

GET /api/papers/{paper_id}/history - Get all interrogation history
  Returns: { sessions: array, reports: array }

POST /api/interrogators/custom - Add custom interrogator agent
  Body: { name: string, focus_area: string, prompts: array }
  Returns: { interrogator_id: string }
```

## Core Algorithm (Pseudocode)

```
FUNCTION interrogate_paper(paper_content, intensity_level):
  # Phase 1: Document Analysis
  parsed_paper = parse_document(paper_content)
  claims = extract_claims(parsed_paper)
  methodology = extract_methodology(parsed_paper)

  # Phase 2: Initialize Interrogator Agents
  interrogators = [
    MethodologyAgent(specialty="research_design"),
    StatisticsAgent(specialty="data_analysis"),
    LogicAgent(specialty="reasoning_flaws"),
    DomainAgent(specialty=parsed_paper.field),
    EthicsAgent(specialty="research_ethics")
  ]

  # Phase 3: Multi-Round Interrogation
  weaknesses = []
  for round in range(intensity_to_rounds(intensity_level)):
    for interrogator in interrogators:
      question = interrogator.generate_question(parsed_paper, previous_answers)
      response = simulate_author_response(question, parsed_paper)

      weakness = interrogator.analyze_response(response)
      if weakness:
        weaknesses.append(weakness)

      # Adaptive questioning based on weaknesses found
      if weakness.severity > 3:
        followup = interrogator.generate_followup(weakness)
        deep_probe_result = deep_probe(followup, parsed_paper)
        weaknesses.extend(deep_probe_result)

  # Phase 4: Cross-Validation
  for weakness in weaknesses:
    verification = verify_weakness_across_agents(weakness, interrogators)
    weakness.confidence = verification.confidence_score

  # Phase 5: Generate Report
  validity_score = calculate_validity_score(weaknesses, parsed_paper)
  report = generate_interrogation_report(weaknesses, validity_score)

  return {
    'validity_score': validity_score,
    'weaknesses': weaknesses,
    'report': report,
    'recommendations': generate_improvements(weaknesses)
  }
```

## Tech Stack Decisions

**Frontend:** Next.js 14 with TypeScript
- Real-time updates via WebSockets for interrogation progress
- Tailwind CSS for rapid UI development

**Backend:** Python/FastAPI
- Async support for concurrent AI interrogations
- Pydantic for data validation

**Database:** PostgreSQL with pgvector
- Store embeddings for semantic paper analysis
- JSONB for flexible weakness storage

**Cache:** Redis
- Cache AI responses to reduce API costs
- Session state for long-running interrogations

**AI Models:**
- GPT-4 for methodology interrogation
- Claude-3 for logical analysis
- Specialized models for statistics checking

**Queue:** Celery + RabbitMQ
- Handle long-running interrogation sessions
- Parallel processing of multiple papers

**Hosting:** AWS
- EC2 for compute
- RDS for PostgreSQL
- ElastiCache for Redis
- S3 for paper storage

**Monitoring:** DataDog
- Track AI API usage and costs
- Monitor interrogation completion rates

**Why these choices:**
- FastAPI for high-performance async AI operations
- PostgreSQL for complex relational data and JSON flexibility
- Redis for real-time session state management
- Multiple AI models for specialized interrogation expertise

## Implementation Roadmap

**Week 1-2:** Foundation Setup
- Set up FastAPI backend with database models
- Implement paper parsing and claim extraction
- Create basic interrogation flow

**Week 3-4:** Core Interrogation Engine
- Build multi-agent interrogator system
- Implement question generation algorithms
- Create weakness detection logic

**Week 5-6:** AI Integration
- Integrate GPT-4 and Claude APIs
- Fine-tune prompts for each interrogator type
- Implement response analysis system

**Week 7-8:** Testing and Deployment
- Test with real research papers
- Optimize for performance and cost
- Deploy to AWS with monitoring
- Create user documentation