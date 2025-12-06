"""
NIGHTMARE MODE - Multi-Angle AI Code Reviewer
The most brutally honest code review system ever created
"""

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, AsyncGenerator
from datetime import datetime
import asyncio
import uuid
import json
import random
from enum import Enum

app = FastAPI(
    title="Nightmare Mode",
    version="1.0.0",
    description="Prepare to have your code destroyed"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class NightmareLevel(int, Enum):
    GENTLE = 1  # Still brutal
    HARSH = 3
    BRUTAL = 5
    SAVAGE = 7
    APOCALYPTIC = 10

class ProjectSubmission(BaseModel):
    name: str
    description: Optional[str] = None
    github_url: Optional[str] = None
    code_snippet: Optional[str] = None
    language: str = "python"
    framework: Optional[str] = None

class NightmareRequest(BaseModel):
    nightmare_level: int = Field(ge=1, le=10, default=5)
    focus_areas: List[str] = []
    extra_brutal: bool = False

class CriticReview(BaseModel):
    critic_type: str
    critic_persona: str
    harshness_level: int
    score: float
    review_text: str
    key_issues: List[str]
    most_brutal_comment: str

class Problem(BaseModel):
    severity: str  # 'catastrophic', 'severe', 'major', 'minor'
    description: str
    location: Optional[str] = None
    fix: Optional[str] = None
    sarcastic_remark: str

class NightmareReport(BaseModel):
    session_id: str
    project_id: str
    overall_verdict: str
    survival_score: float
    total_issues: int
    catastrophic_issues: int
    most_brutal_criticism: str
    harshest_critic: str
    all_reviews: List[CriticReview]
    will_to_code_remaining: float
    timestamp: datetime

# In-memory storage
projects_db = {}
sessions_db = {}
reports_db = {}

# Brutal Critics
class Critic:
    def __init__(self, name: str, specialty: str, personality: str):
        self.name = name
        self.specialty = specialty
        self.personality = personality
        self.brutal_phrases = self._load_brutal_phrases()

    def _load_brutal_phrases(self) -> List[str]:
        """Load devastating phrases based on personality"""
        phrases = {
            "Angry Veteran": [
                "I've seen better code written by interns on their first day",
                "This is what happens when you learn programming from YouTube",
                "Did you write this with your eyes closed?",
                "My grandmother writes cleaner code, and she's been dead for 10 years",
                "This code is a war crime against computer science"
            ],
            "Perfectionist": [
                "This is an insult to the concept of software engineering",
                "Every line of this code is wrong in its own unique way",
                "I've never seen so many anti-patterns in one place",
                "This violates every principle I hold dear",
                "Calling this 'code' is generous"
            ],
            "Security Paranoid": [
                "This is a hacker's paradise",
                "You might as well publish your passwords on Twitter",
                "I counted 47 security vulnerabilities in the first 10 lines",
                "This code is a lawsuit waiting to happen",
                "The NSA called, they want to thank you for the backdoors"
            ],
            "Performance Obsessed": [
                "This code runs slower than a snail on tranquilizers",
                "You've discovered new ways to waste CPU cycles",
                "O(n!) complexity? Really? REALLY?",
                "This would timeout on a quantum computer",
                "My patience runs out faster than this code runs"
            ],
            "Clean Code Zealot": [
                "Uncle Bob is crying somewhere",
                "This code gave me physical pain",
                "Variable names like 'x' and 'temp'? Are we in 1970?",
                "This function is longer than my will to live",
                "I've seen spaghetti with better structure"
            ]
        }
        return phrases.get(self.personality, ["This code is questionable at best"])

    async def review(self, code: str, nightmare_level: int) -> CriticReview:
        """Generate a brutal review"""
        # Simulate analysis delay
        await asyncio.sleep(0.2)

        # Generate score (lower is worse, adjusted by nightmare level)
        base_score = random.uniform(2, 6)
        score = max(0, base_score - (nightmare_level * 0.3))

        # Select brutal comments
        brutal_comment = random.choice(self.brutal_phrases)

        # Generate issues based on specialty
        issues = self._generate_issues(code, nightmare_level)

        # Generate review text
        review_text = self._generate_review_text(code, issues, nightmare_level)

        return CriticReview(
            critic_type=self.specialty,
            critic_persona=self.personality,
            harshness_level=nightmare_level,
            score=round(score, 1),
            review_text=review_text,
            key_issues=issues,
            most_brutal_comment=brutal_comment
        )

    def _generate_issues(self, code: str, nightmare_level: int) -> List[str]:
        """Generate issues based on specialty"""
        issues_db = {
            "architecture": [
                "No clear separation of concerns",
                "Mixing business logic with presentation",
                "God objects everywhere",
                "Zero modularity",
                "Coupling so tight it might explode"
            ],
            "security": [
                "SQL injection vulnerabilities",
                "No input validation whatsoever",
                "Hardcoded credentials",
                "Missing authentication",
                "XSS vulnerabilities galore"
            ],
            "performance": [
                "N+1 query problems",
                "No caching strategy",
                "Synchronous operations that should be async",
                "Memory leaks waiting to happen",
                "Inefficient algorithms throughout"
            ],
            "quality": [
                "No tests. Zero. None.",
                "Comments that lie",
                "Dead code cemetery",
                "Copy-paste programming at its finest",
                "Error handling? Never heard of it"
            ],
            "style": [
                "Inconsistent formatting",
                "Variable names from hell",
                "Functions doing 17 different things",
                "Indentation that defies logic",
                "Line length exceeding my monitor width"
            ]
        }

        specialty_issues = issues_db.get(self.specialty, ["Generally terrible"])
        num_issues = min(nightmare_level, len(specialty_issues))
        return random.sample(specialty_issues, num_issues)

    def _generate_review_text(self, code: str, issues: List[str], nightmare_level: int) -> str:
        """Generate detailed review text"""
        intensity_adjectives = {
            1: "concerning",
            3: "disturbing",
            5: "horrifying",
            7: "soul-crushing",
            10: "apocalyptically bad"
        }

        adj = intensity_adjectives.get(nightmare_level, "terrible")

        review = f"After reviewing this code, I can only describe it as {adj}. "
        review += f"The {len(issues)} major issues I found are just the tip of the iceberg. "
        review += f"Specifically: {', '.join(issues[:3])}. "

        if nightmare_level >= 7:
            review += "I've seen a lot of bad code in my career, but this... this is special. "
            review += "It's like you're actively trying to write the worst possible code. "

        if nightmare_level == 10:
            review += "I need therapy after reading this. You owe me compensation for emotional damage. "

        return review

# Nightmare Engine
class NightmareEngine:
    def __init__(self):
        self.critics = [
            Critic("Bob the Destroyer", "architecture", "Angry Veteran"),
            Critic("Karen from QA", "quality", "Perfectionist"),
            Critic("Security Steve", "security", "Security Paranoid"),
            Critic("Performance Pete", "performance", "Performance Obsessed"),
            Critic("Clean Code Clara", "style", "Clean Code Zealot"),
        ]

    async def unleash_nightmare(
        self,
        project_id: str,
        code: str,
        nightmare_level: int
    ) -> NightmareReport:
        """Unleash the full nightmare review"""
        session_id = str(uuid.uuid4())
        sessions_db[session_id] = {
            "status": "in_nightmare",
            "project_id": project_id,
            "start_time": datetime.now()
        }

        # Deploy all critics
        reviews = []
        for critic in self.critics:
            review = await critic.review(code, nightmare_level)
            reviews.append(review)

            # Make it worse if extra_brutal
            if nightmare_level >= 8:
                review.review_text += " Actually, it's worse than I initially thought."

        # Find the most brutal criticism
        most_brutal = min(reviews, key=lambda r: r.score)

        # Calculate survival score
        avg_score = sum(r.score for r in reviews) / len(reviews)
        survival_score = max(0, avg_score)

        # Generate overall verdict
        verdict = self._generate_verdict(survival_score, nightmare_level)

        # Count issues
        all_issues = []
        for review in reviews:
            all_issues.extend(review.key_issues)

        # Calculate will to code
        will_to_code = max(0, 100 - (nightmare_level * 10) - (10 - survival_score) * 5)

        report = NightmareReport(
            session_id=session_id,
            project_id=project_id,
            overall_verdict=verdict,
            survival_score=round(survival_score, 1),
            total_issues=len(all_issues),
            catastrophic_issues=nightmare_level * 2,
            most_brutal_criticism=most_brutal.most_brutal_comment,
            harshest_critic=most_brutal.critic_persona,
            all_reviews=reviews,
            will_to_code_remaining=will_to_code,
            timestamp=datetime.now()
        )

        reports_db[session_id] = report
        sessions_db[session_id]["status"] = "nightmare_complete"

        return report

    def _generate_verdict(self, survival_score: float, nightmare_level: int) -> str:
        """Generate the final brutal verdict"""
        if survival_score < 2:
            verdicts = [
                "This code is a crime against humanity",
                "I've seen tragedy, but this is something else",
                "Burn it. Burn it all and start over",
                "This is why aliens won't talk to us",
                "I'm calling the code police"
            ]
        elif survival_score < 4:
            verdicts = [
                "This barely qualifies as code",
                "My disappointment is immeasurable",
                "Did you even try?",
                "This is what nightmares are made of",
                "I've lost faith in humanity"
            ]
        elif survival_score < 6:
            verdicts = [
                "Mediocre would be a compliment",
                "I expected nothing and I'm still disappointed",
                "This is aggressively average",
                "You can do better. You MUST do better",
                "Technically it's code, but that's about it"
            ]
        else:
            verdicts = [
                "It's not completely terrible, I guess",
                "I've seen worse, but not often",
                "Congratulations, you've achieved adequacy",
                "This is almost acceptable",
                "You might have a future in coding. Maybe."
            ]

        base_verdict = random.choice(verdicts)

        if nightmare_level >= 8:
            base_verdict += " (That's the nicest thing I can say)"

        return base_verdict

# Initialize engine
engine = NightmareEngine()

# API Endpoints
@app.post("/api/projects/submit")
async def submit_project(project: ProjectSubmission) -> Dict[str, str]:
    """Submit a project for nightmare review"""
    project_id = str(uuid.uuid4())
    projects_db[project_id] = project.dict()
    projects_db[project_id]["submission_time"] = datetime.now()

    return {
        "project_id": project_id,
        "message": "Project submitted. Prepare for pain.",
        "warning": "There's still time to back out..."
    }

@app.post("/api/sessions/{project_id}/unleash-nightmare")
async def start_nightmare(
    project_id: str,
    request: NightmareRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Start the nightmare review"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects_db[project_id]
    code = project.get("code_snippet", "# No code provided")

    # Start nightmare in background for dramatic effect
    session_id = str(uuid.uuid4())

    estimated_trauma = request.nightmare_level * 10  # seconds of pain

    return {
        "session_id": session_id,
        "status": "nightmare_initiated",
        "nightmare_level": request.nightmare_level,
        "estimated_trauma_time": estimated_trauma,
        "final_words": "May God have mercy on your code"
    }

@app.get("/api/sessions/{session_id}/live-roast")
async def live_roast_stream(session_id: str):
    """Stream live brutal comments as they're generated"""
    async def generate_roasts() -> AsyncGenerator[str, None]:
        roasts = [
            "Analyzing your code... oh no...",
            "This is worse than I thought...",
            "I'm finding new ways to be disappointed...",
            "Your code is making my AI cry...",
            "Even my error handling can't handle these errors...",
            "I've seen bad code before, but this...",
            "*Critics are speechless*",
            "The security expert just quit...",
            "Performance analysis complete: It doesn't perform...",
            "Clean code principles found: 0"
        ]

        for roast in roasts:
            yield f"data: {json.dumps({'comment': roast, 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate_roasts(), media_type="text/event-stream")

@app.websocket("/ws/nightmare/{session_id}")
async def websocket_nightmare(websocket: WebSocket, session_id: str):
    """WebSocket for real-time brutal feedback"""
    await websocket.accept()

    try:
        critics_comments = [
            {"critic": "Angry Veteran", "comment": "What is this garbage?"},
            {"critic": "Perfectionist", "comment": "Every line hurts me"},
            {"critic": "Security Steve", "comment": "This is a security nightmare"},
            {"critic": "Performance Pete", "comment": "O(n^n)? Seriously?"},
            {"critic": "Clean Code Clara", "comment": "I need a shower after reading this"}
        ]

        for comment in critics_comments:
            await websocket.send_json(comment)
            await asyncio.sleep(2)

        await websocket.send_json({"status": "complete", "message": "The nightmare is over"})

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@app.get("/api/reports/{session_id}")
async def get_nightmare_report(session_id: str) -> Dict[str, Any]:
    """Get the final nightmare report"""
    # Generate report on the fly for demo
    project_id = str(uuid.uuid4())
    code = "def calculate(x): return x + 1"  # Sample code

    report = await engine.unleash_nightmare(project_id, code, 7)

    return report.dict()

@app.get("/api/leaderboard/survivors")
async def get_survivors() -> Dict[str, Any]:
    """Show the brave souls who survived"""
    survivors = [
        {"name": "Anonymous", "survival_score": 3.2, "comment": "Never again"},
        {"name": "BraveDev", "survival_score": 4.1, "comment": "I've seen things"},
        {"name": "CodingMasochist", "survival_score": 2.8, "comment": "More please"},
    ]

    return {
        "top_survivors": survivors,
        "average_survival_rate": 3.4,
        "total_nightmares": 1337,
        "careers_ended": 42
    }

@app.post("/api/demo/nightmare")
async def demo_nightmare() -> Dict[str, Any]:
    """Demo endpoint to experience the nightmare"""
    sample_code = """
def calculate_sum(list_of_numbers):
    total = 0
    for i in range(len(list_of_numbers)):
        total = total + list_of_numbers[i]
    return total

x = [1,2,3,4,5]
print(calculate_sum(x))
"""

    project = ProjectSubmission(
        name="Demo Project",
        description="A simple sum calculator",
        code_snippet=sample_code,
        language="python"
    )

    # Submit project
    project_response = await submit_project(project)
    project_id = project_response["project_id"]

    # Run nightmare
    report = await engine.unleash_nightmare(
        project_id,
        sample_code,
        nightmare_level=8
    )

    return {
        "demo": True,
        "message": "You asked for this",
        "report": report.dict()
    }

@app.get("/api/motivation")
async def get_motivation() -> Dict[str, str]:
    """Emergency motivation after the nightmare"""
    motivations = [
        "Everyone writes bad code sometimes",
        "At least it compiles... right?",
        "Tomorrow is another day to write better code",
        "The critics are harsh because they care (they don't)",
        "You're brave for submitting your code"
    ]

    return {
        "motivation": random.choice(motivations),
        "therapy_hotline": "1-800-BAD-CODE"
    }

if __name__ == "__main__":
    import uvicorn
    print("⚠️  WARNING: Nightmare Mode is starting...")
    print("💀 Prepare for brutal honesty about your code")
    uvicorn.run(app, host="0.0.0.0", port=8001)