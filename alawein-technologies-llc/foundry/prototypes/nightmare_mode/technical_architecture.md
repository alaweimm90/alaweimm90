# NIGHTMARE MODE - Technical Architecture

## System Diagram

```
[User Code/Project] → [Multi-Perspective Analyzer]
                              ↓
                    [10 Specialized AI Critics]
                              ↓
                 [Harsh Critique Generation Engine]
                              ↓
                    [Weakness Amplification]
                              ↓
                [Brutal Feedback Report + Score]
```

## Database Schema

```sql
-- Projects under review
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    github_url VARCHAR(500),
    language VARCHAR(50),
    framework VARCHAR(100),
    submission_date TIMESTAMP DEFAULT NOW(),
    user_id UUID,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Review sessions
CREATE TABLE review_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    session_start TIMESTAMP DEFAULT NOW(),
    session_end TIMESTAMP,
    nightmare_level INTEGER CHECK (nightmare_level BETWEEN 1 AND 10),
    overall_score DECIMAL(3,1) CHECK (overall_score BETWEEN 0 AND 10),
    status VARCHAR(50) DEFAULT 'in_progress'
);

-- Individual critic reviews
CREATE TABLE critic_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES review_sessions(id),
    critic_type VARCHAR(100), -- 'security', 'performance', 'architecture', etc.
    critic_persona VARCHAR(200), -- 'Angry Senior Dev', 'Perfectionist', etc.
    harshness_level INTEGER CHECK (harshness_level BETWEEN 1 AND 10),
    review_text TEXT NOT NULL,
    score DECIMAL(3,1) CHECK (score BETWEEN 0 AND 10),
    key_issues JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Identified problems
CREATE TABLE identified_problems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID REFERENCES critic_reviews(id),
    problem_type VARCHAR(100),
    severity VARCHAR(50), -- 'catastrophic', 'severe', 'major', 'minor'
    description TEXT,
    file_path VARCHAR(500),
    line_numbers INTEGER[],
    suggested_fix TEXT,
    sarcasm_level INTEGER CHECK (sarcasm_level BETWEEN 0 AND 10)
);

-- Code smells and anti-patterns
CREATE TABLE code_smells (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES review_sessions(id),
    smell_type VARCHAR(100),
    description TEXT,
    locations JSONB,
    impact_description TEXT,
    roast_comment TEXT -- The particularly harsh comment about this smell
);

-- Final nightmare reports
CREATE TABLE nightmare_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES review_sessions(id),
    project_id UUID REFERENCES projects(id),
    overall_verdict VARCHAR(500), -- The brutal one-liner summary
    survival_score DECIMAL(3,1), -- How well the code survived the nightmare
    total_issues_found INTEGER,
    catastrophic_issues INTEGER,
    severe_issues INTEGER,
    most_brutal_criticism TEXT,
    harshest_critic VARCHAR(100),
    generated_at TIMESTAMP DEFAULT NOW()
);

-- User reactions and feedback
CREATE TABLE user_reactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID REFERENCES nightmare_reports(id),
    user_id UUID,
    reaction VARCHAR(50), -- 'cried', 'laughed', 'quit_coding', 'motivated', etc.
    found_helpful BOOLEAN,
    actually_fixed_issues BOOLEAN,
    comment TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_sessions_project ON review_sessions(project_id);
CREATE INDEX idx_critics_session ON critic_reviews(session_id);
CREATE INDEX idx_problems_severity ON identified_problems(severity);
```

## API Endpoints

```
POST /api/projects/submit - Submit code for nightmare review
  Body: { github_url?: string, code_files?: array, language: string }
  Returns: { project_id: string, session_id: string }

POST /api/sessions/{session_id}/unleash-nightmare - Start the nightmare review
  Body: { nightmare_level: 1-10, focus_areas?: array }
  Returns: { status: string, estimated_trauma_time: number }

GET /api/sessions/{session_id}/status - Get current nightmare progress
  Returns: { critics_deployed: number, tears_shed: number, progress: number }

GET /api/sessions/{session_id}/live-roast - Stream live criticism as it happens
  Returns: EventStream of brutal comments

GET /api/reports/{session_id} - Get the final nightmare report
  Returns: { verdict: string, survival_score: number, brutal_feedback: array }

POST /api/critics/summon - Add a custom critic personality
  Body: { name: string, specialty: string, harshness: number, personality: string }
  Returns: { critic_id: string }

GET /api/leaderboard/survivors - Show who survived the nightmare
  Returns: { top_survivors: array, average_survival_rate: number }
```

## Core Algorithm (Pseudocode)

```
FUNCTION nightmare_review(code_project, nightmare_level):
  # Phase 1: Deploy the Critics
  critics = [
    AngryVeteranDev(years_of_rage=20),
    PerfectionistArchitect(tolerance=0),
    SecurityParanoid(trust_level=-100),
    PerformanceObsessed(microseconds_matter=true),
    CleanCodeZealot(uncle_bob_mode=true),
    DocumentationNazi(comments_per_line=2),
    TestCoverageFanatic(minimum=100),
    AccessibilityAdvocate(zero_tolerance=true),
    ScalabilityPessimist(users=1_billion),
    OpenSourceElitist(standards=impossibly_high)
  ]

  # Adjust harshness based on nightmare level
  for critic in critics:
    critic.harshness = nightmare_level
    critic.mercy = 0
    critic.sarcasm = nightmare_level * 2

  # Phase 2: Analyze from Every Angle
  all_reviews = []
  for critic in critics:
    review = critic.review_mercilessly(code_project)

    # Make it worse
    review = amplify_negativity(review, nightmare_level)
    review = add_sarcastic_comparisons(review)
    review = question_life_choices(review)

    all_reviews.append(review)

  # Phase 3: Find Every Possible Flaw
  flaws = []
  flaws.extend(find_code_smells(code_project, sensitivity=MAXIMUM))
  flaws.extend(find_anti_patterns(code_project, pedantic=true))
  flaws.extend(find_naming_issues(code_project, nitpick_level=EXTREME))
  flaws.extend(find_formatting_crimes(code_project, tolerance=ZERO))
  flaws.extend(imagine_edge_cases(code_project, paranoia=MAXIMUM))

  # Phase 4: Generate Brutal Comparisons
  comparisons = []
  comparisons.append(compare_to_industry_best(code_project))
  comparisons.append(compare_to_theoretical_perfect(code_project))
  comparisons.append(compare_to_my_grandmother_code(code_project))

  # Phase 5: Calculate Survival Score
  total_criticism_points = calculate_trauma(all_reviews, flaws)
  survival_score = max(0, 10 - (total_criticism_points / 100))

  # Phase 6: Generate The Ultimate Roast
  final_verdict = generate_soul_crushing_summary(all_reviews, flaws)
  most_brutal = select_most_devastating_comment(all_reviews)

  return {
    'survival_score': survival_score,
    'verdict': final_verdict,
    'most_brutal': most_brutal,
    'all_reviews': all_reviews,
    'total_issues': len(flaws),
    'will_to_code_remaining': calculate_remaining_motivation(survival_score)
  }
```

## Tech Stack Decisions

**Frontend:** Next.js with Framer Motion
- Dramatic animations for criticism reveal
- Real-time streaming of harsh feedback
- Dark theme (obviously)

**Backend:** Python/FastAPI
- Async streaming for live roasting
- Multiple AI model orchestration

**Database:** PostgreSQL
- Store the tears and trauma
- Track improvement over time

**Cache:** Redis
- Cache particularly brutal comments for reuse
- Store session state during review

**AI Models:**
- GPT-4 for creative insults
- Claude for thorough analysis
- CodeBERT for code understanding
- Custom fine-tuned roast model

**Queue:** Celery
- Queue multiple critics in parallel
- Manage long review sessions

**Hosting:** AWS
- Lambda for serverless critic functions
- DynamoDB for quick roast storage

**Analytics:** Amplitude
- Track user emotional responses
- Measure actual code improvements

**Why these choices:**
- FastAPI for handling multiple AI critics concurrently
- PostgreSQL for complex relationship between reviews and issues
- Redis for real-time brutal feedback streaming
- Multiple AI models for comprehensive destruction

## Implementation Roadmap

**Week 1-2:** Critic System Foundation
- Build base critic classes and personalities
- Implement code analysis framework
- Create harshness adjustment system

**Week 3-4:** Brutal Review Engine
- Integrate multiple AI models
- Build sarcasm and roast generators
- Implement real-time streaming

**Week 5-6:** Psychological Damage System
- Create trauma calculation algorithm
- Build motivation tracking
- Add recovery suggestions

**Week 7-8:** Launch and Therapy
- Deploy to production
- Create emotional support documentation
- Set up counseling hotline integration