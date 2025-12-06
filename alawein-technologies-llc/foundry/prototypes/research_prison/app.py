"""
RESEARCH PRISON - AI Interrogation System
A working prototype that interrogates research papers to find weaknesses
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import uuid
import json
from enum import Enum

app = FastAPI(title="Research Prison", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class IntensityLevel(str, Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class PaperSubmission(BaseModel):
    title: str
    abstract: str
    full_text: str
    authors: List[str] = []
    field: str = "general"
    url: Optional[str] = None

class InterrogationRequest(BaseModel):
    intensity: IntensityLevel = IntensityLevel.MODERATE

class Weakness(BaseModel):
    type: str
    description: str
    severity: int = Field(ge=1, le=5)
    evidence: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None

class InterrogationRound(BaseModel):
    round_number: int
    interrogator_type: str
    question: str
    response: str
    weakness_found: bool
    weaknesses: List[Weakness] = []

class InterrogationReport(BaseModel):
    session_id: str
    paper_id: str
    validity_score: float = Field(ge=0, le=1)
    verdict: str
    total_rounds: int
    weaknesses: List[Weakness]
    major_issues: List[str]
    minor_issues: List[str]
    recommendations: List[str]
    timestamp: datetime

# In-memory storage (replace with database in production)
papers_db = {}
sessions_db = {}
reports_db = {}

# Interrogator Agents
class InterrogatorAgent:
    def __init__(self, agent_type: str, focus_area: str):
        self.agent_type = agent_type
        self.focus_area = focus_area
        self.questions = self._load_questions()

    def _load_questions(self) -> List[str]:
        """Load interrogation questions based on agent type"""
        questions_db = {
            "methodology": [
                "How was the sample size determined and is it sufficient for statistical power?",
                "What controls were implemented to prevent bias?",
                "Were there any confounding variables not addressed?",
                "How reproducible is your experimental setup?",
                "What validation methods were used for your measurements?"
            ],
            "statistics": [
                "What statistical tests were used and why were they appropriate?",
                "How were outliers handled in the data analysis?",
                "Was multiple testing correction applied where necessary?",
                "What assumptions were made for the statistical models?",
                "How was statistical significance determined?"
            ],
            "logic": [
                "Is there circular reasoning in any of your arguments?",
                "Are all conclusions directly supported by the presented data?",
                "Are there alternative explanations for your findings?",
                "How do you address contradictory evidence from other studies?",
                "Are there logical leaps in your reasoning chain?"
            ],
            "ethics": [
                "How was informed consent obtained from participants?",
                "Were there any conflicts of interest?",
                "How was data privacy ensured?",
                "Were vulnerable populations involved?",
                "Was the study approved by an ethics committee?"
            ],
            "data": [
                "How was data quality ensured during collection?",
                "What was the data cleaning process?",
                "How complete is the dataset?",
                "Were there any data integrity issues?",
                "How was missing data handled?"
            ]
        }
        return questions_db.get(self.agent_type, [])

    async def interrogate(self, paper_content: dict, round_num: int) -> InterrogationRound:
        """Perform one round of interrogation"""
        # Select question based on round
        question_idx = round_num % len(self.questions)
        question = self.questions[question_idx] if self.questions else "Explain your methodology."

        # Simulate paper response (in production, use AI model)
        response = await self._generate_response(paper_content, question)

        # Analyze response for weaknesses
        weaknesses = await self._analyze_response(response, paper_content)

        return InterrogationRound(
            round_number=round_num,
            interrogator_type=self.agent_type,
            question=question,
            response=response,
            weakness_found=len(weaknesses) > 0,
            weaknesses=weaknesses
        )

    async def _generate_response(self, paper: dict, question: str) -> str:
        """Simulate author's response to question (replace with AI in production)"""
        await asyncio.sleep(0.1)  # Simulate processing
        return f"Based on the paper's content about '{paper.get('title', '')}', the response to '{question}' would address the specific methodological choices made..."

    async def _analyze_response(self, response: str, paper: dict) -> List[Weakness]:
        """Analyze response for weaknesses (replace with AI in production)"""
        weaknesses = []

        # Simulate weakness detection
        if "sample size" in response.lower() or self.agent_type == "statistics":
            if len(paper.get('full_text', '')) < 1000:  # Simple heuristic
                weaknesses.append(Weakness(
                    type="statistical_power",
                    description="Insufficient sample size for claimed statistical significance",
                    severity=3,
                    evidence="Paper mentions small sample but claims strong effects",
                    suggested_fix="Increase sample size or adjust statistical claims"
                ))

        return weaknesses

# Core Interrogation Engine
class InterrogationEngine:
    def __init__(self):
        self.interrogators = [
            InterrogatorAgent("methodology", "research_design"),
            InterrogatorAgent("statistics", "data_analysis"),
            InterrogatorAgent("logic", "reasoning"),
            InterrogatorAgent("ethics", "research_ethics"),
            InterrogatorAgent("data", "data_quality")
        ]

    async def interrogate_paper(
        self,
        paper_id: str,
        paper_content: dict,
        intensity: IntensityLevel
    ) -> InterrogationReport:
        """Main interrogation process"""
        session_id = str(uuid.uuid4())
        sessions_db[session_id] = {
            "status": "in_progress",
            "paper_id": paper_id,
            "start_time": datetime.now()
        }

        # Determine number of rounds based on intensity
        rounds_map = {
            IntensityLevel.LIGHT: 3,
            IntensityLevel.MODERATE: 5,
            IntensityLevel.AGGRESSIVE: 10
        }
        num_rounds = rounds_map[intensity]

        all_weaknesses = []
        all_rounds = []

        # Perform interrogation rounds
        for round_num in range(num_rounds):
            for interrogator in self.interrogators:
                round_result = await interrogator.interrogate(paper_content, round_num)
                all_rounds.append(round_result)
                all_weaknesses.extend(round_result.weaknesses)

        # Calculate validity score
        validity_score = self._calculate_validity_score(all_weaknesses)

        # Generate verdict
        verdict = self._determine_verdict(validity_score)

        # Categorize issues
        major_issues = [w.description for w in all_weaknesses if w.severity >= 4]
        minor_issues = [w.description for w in all_weaknesses if w.severity < 4]

        # Generate recommendations
        recommendations = self._generate_recommendations(all_weaknesses)

        # Create report
        report = InterrogationReport(
            session_id=session_id,
            paper_id=paper_id,
            validity_score=validity_score,
            verdict=verdict,
            total_rounds=len(all_rounds),
            weaknesses=all_weaknesses,
            major_issues=major_issues,
            minor_issues=minor_issues,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

        # Store report
        reports_db[session_id] = report
        sessions_db[session_id]["status"] = "completed"
        sessions_db[session_id]["end_time"] = datetime.now()

        return report

    def _calculate_validity_score(self, weaknesses: List[Weakness]) -> float:
        """Calculate paper validity score based on weaknesses"""
        if not weaknesses:
            return 1.0

        # Weight weaknesses by severity
        total_impact = sum(w.severity for w in weaknesses)
        max_possible_impact = len(weaknesses) * 5

        # Invert to get validity (more weaknesses = lower validity)
        validity = 1.0 - (total_impact / max_possible_impact) if max_possible_impact > 0 else 1.0
        return max(0.0, min(1.0, validity))

    def _determine_verdict(self, validity_score: float) -> str:
        """Determine verdict based on validity score"""
        if validity_score >= 0.8:
            return "VALID - Minor issues only"
        elif validity_score >= 0.6:
            return "QUESTIONABLE - Significant issues found"
        elif validity_score >= 0.4:
            return "FLAWED - Major methodological problems"
        else:
            return "INVALID - Critical flaws detected"

    def _generate_recommendations(self, weaknesses: List[Weakness]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Group weaknesses by type
        weakness_types = {}
        for w in weaknesses:
            if w.type not in weakness_types:
                weakness_types[w.type] = []
            weakness_types[w.type].append(w)

        # Generate recommendations for each type
        for w_type, w_list in weakness_types.items():
            if w_type == "statistical_power":
                recommendations.append("Increase sample size or reduce claimed effect size")
            elif w_type == "methodology":
                recommendations.append("Improve experimental controls and validation methods")
            elif w_type == "logic":
                recommendations.append("Strengthen logical arguments and address contradictions")

        # Add general recommendations
        if len(weaknesses) > 5:
            recommendations.append("Consider major revision of research design")

        return recommendations

# Initialize engine
engine = InterrogationEngine()

# API Endpoints
@app.post("/api/papers/submit")
async def submit_paper(paper: PaperSubmission) -> Dict[str, str]:
    """Submit a research paper for interrogation"""
    paper_id = str(uuid.uuid4())
    papers_db[paper_id] = paper.dict()
    papers_db[paper_id]["submission_date"] = datetime.now()
    papers_db[paper_id]["status"] = "submitted"

    return {
        "paper_id": paper_id,
        "message": "Paper submitted successfully",
        "status": "ready_for_interrogation"
    }

@app.post("/api/sessions/{paper_id}/interrogate")
async def start_interrogation(
    paper_id: str,
    request: InterrogationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Start interrogation session for a paper"""
    if paper_id not in papers_db:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Start interrogation in background
    session_id = str(uuid.uuid4())
    background_tasks.add_task(
        engine.interrogate_paper,
        paper_id,
        papers_db[paper_id],
        request.intensity
    )

    return {
        "session_id": session_id,
        "paper_id": paper_id,
        "status": "interrogation_started",
        "intensity": request.intensity,
        "message": "Interrogation started. Check status endpoint for progress."
    }

@app.get("/api/sessions/{session_id}")
async def get_session_status(session_id: str) -> Dict[str, Any]:
    """Get current interrogation status"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions_db[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "paper_id": session["paper_id"],
        "start_time": session["start_time"],
        "end_time": session.get("end_time")
    }

@app.get("/api/reports/{session_id}")
async def get_report(session_id: str) -> InterrogationReport:
    """Get final interrogation report"""
    if session_id not in reports_db:
        raise HTTPException(status_code=404, detail="Report not found")

    return reports_db[session_id]

@app.get("/api/papers/{paper_id}/history")
async def get_paper_history(paper_id: str) -> Dict[str, Any]:
    """Get all interrogation history for a paper"""
    if paper_id not in papers_db:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Find all sessions for this paper
    paper_sessions = [
        s for s in sessions_db.values()
        if s["paper_id"] == paper_id
    ]

    # Find all reports for this paper
    paper_reports = [
        r for r in reports_db.values()
        if r.paper_id == paper_id
    ]

    return {
        "paper_id": paper_id,
        "paper_details": papers_db[paper_id],
        "total_sessions": len(paper_sessions),
        "sessions": paper_sessions,
        "reports": paper_reports
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "papers_count": len(papers_db),
        "active_sessions": sum(1 for s in sessions_db.values() if s["status"] == "in_progress")
    }

# Demo endpoint for testing
@app.post("/api/demo/interrogate")
async def demo_interrogation() -> Dict[str, Any]:
    """Demo endpoint with sample paper"""
    sample_paper = PaperSubmission(
        title="A Novel Approach to Quantum Computing Using Tea Leaves",
        abstract="We demonstrate that tea leaves can be used as quantum bits...",
        full_text="In this groundbreaking study, we show that ordinary tea leaves exhibit quantum properties when steeped at exactly 98.6°F. Our sample size of 3 tea bags showed consistent results.",
        authors=["Dr. Earl Grey", "Prof. Oolong"],
        field="Quantum Physics"
    )

    # Submit paper
    paper_response = await submit_paper(sample_paper)
    paper_id = paper_response["paper_id"]

    # Run interrogation
    paper_content = papers_db[paper_id]
    report = await engine.interrogate_paper(
        paper_id,
        paper_content,
        IntensityLevel.AGGRESSIVE
    )

    return {
        "demo": True,
        "paper_id": paper_id,
        "report": report.dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)