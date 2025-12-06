"""
CHAOS ENGINE - Domain Collision Idea Generator
Where incompatible domains collide to create breakthrough innovations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import asyncio
import uuid
import random
import json
from enum import Enum

app = FastAPI(
    title="Chaos Engine",
    version="1.0.0",
    description="Colliding incompatible domains since 2024"
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
class ChaosLevel(int, Enum):
    MILD = 1
    MODERATE = 3
    WILD = 5
    EXTREME = 7
    INSANE = 10

class Domain(BaseModel):
    name: str
    category: str
    key_concepts: List[str]
    typical_problems: List[str]
    core_technologies: List[str]
    market_size: Optional[int] = None

class CollisionRequest(BaseModel):
    chaos_level: int = Field(ge=1, le=10, default=5)
    domain1_id: Optional[str] = None
    domain2_id: Optional[str] = None
    custom_domain1: Optional[str] = None
    custom_domain2: Optional[str] = None

class GeneratedIdea(BaseModel):
    id: str
    title: str
    tagline: str
    description: str
    key_features: List[str]
    target_market: str
    absurdity_score: float
    viability_score: float
    innovation_score: float
    from_domains: Tuple[str, str]

class BusinessPlan(BaseModel):
    executive_summary: str
    problem_statement: str
    solution: str
    market_opportunity: str
    revenue_model: str
    competitive_advantage: str
    implementation_steps: List[str]
    funding_required: str
    projected_revenue_year1: str
    projected_revenue_year5: str
    risk_factors: List[str]

class PitchSlide(BaseModel):
    slide_number: int
    title: str
    content: str
    visual_suggestion: str

class PitchDeck(BaseModel):
    idea_id: str
    hook: str
    slides: List[PitchSlide]
    call_to_action: str

class CollisionResult(BaseModel):
    session_id: str
    idea: GeneratedIdea
    business_plan: BusinessPlan
    pitch_deck: PitchDeck
    chaos_level: int
    timestamp: datetime

# Predefined domains database
DOMAINS_DB = {
    "blockchain": Domain(
        name="Blockchain",
        category="Technology",
        key_concepts=["decentralization", "smart contracts", "cryptocurrency", "NFTs", "consensus"],
        typical_problems=["trust", "transparency", "intermediaries", "ownership"],
        core_technologies=["Ethereum", "Bitcoin", "Web3", "DeFi"],
        market_size=1000000000
    ),
    "meditation": Domain(
        name="Meditation",
        category="Wellness",
        key_concepts=["mindfulness", "breathing", "zen", "consciousness", "inner peace"],
        typical_problems=["stress", "anxiety", "focus", "mental health"],
        core_technologies=["apps", "VR", "biofeedback", "sound therapy"],
        market_size=2000000000
    ),
    "fast_food": Domain(
        name="Fast Food",
        category="Food Service",
        key_concepts=["speed", "convenience", "franchising", "drive-through", "value meals"],
        typical_problems=["wait times", "quality", "health", "customization"],
        core_technologies=["POS systems", "kitchen automation", "delivery apps"],
        market_size=800000000000
    ),
    "quantum_computing": Domain(
        name="Quantum Computing",
        category="Advanced Technology",
        key_concepts=["qubits", "superposition", "entanglement", "quantum supremacy"],
        typical_problems=["complexity", "optimization", "cryptography", "simulation"],
        core_technologies=["IBM Q", "Google Sycamore", "quantum algorithms"],
        market_size=65000000000
    ),
    "dating": Domain(
        name="Dating",
        category="Social",
        key_concepts=["matching", "romance", "compatibility", "profiles", "swipping"],
        typical_problems=["loneliness", "authenticity", "ghosting", "commitment"],
        core_technologies=["AI matching", "video dating", "personality tests"],
        market_size=8000000000
    ),
    "funeral_services": Domain(
        name="Funeral Services",
        category="Death Care",
        key_concepts=["memorial", "burial", "cremation", "grief", "celebration of life"],
        typical_problems=["cost", "planning", "personalization", "environmental impact"],
        core_technologies=["online memorials", "green burial", "livestreaming"],
        market_size=20000000000
    ),
    "extreme_sports": Domain(
        name="Extreme Sports",
        category="Recreation",
        key_concepts=["adrenaline", "risk", "adventure", "skill", "competition"],
        typical_problems=["safety", "accessibility", "training", "equipment cost"],
        core_technologies=["action cameras", "safety gear", "VR training"],
        market_size=10000000000
    ),
    "tax_preparation": Domain(
        name="Tax Preparation",
        category="Financial Services",
        key_concepts=["deductions", "filing", "compliance", "refunds", "audits"],
        typical_problems=["complexity", "deadlines", "accuracy", "maximizing returns"],
        core_technologies=["tax software", "e-filing", "AI assistants"],
        market_size=11000000000
    )
}

# In-memory storage
sessions_db = {}
ideas_db = {}
business_plans_db = {}
pitch_decks_db = {}

# Chaos Engine Core
class ChaosEngine:
    def __init__(self):
        self.collision_templates = self._load_collision_templates()
        self.business_buzzwords = self._load_business_buzzwords()

    def _load_collision_templates(self) -> List[str]:
        """Templates for combining domains"""
        return [
            "{d1} for {d2} enthusiasts",
            "The {d1} of {d2}",
            "{d2}-powered {d1} revolution",
            "Uber for {d1} meets {d2}",
            "{d1} but make it {d2}",
            "What if {d1} and {d2} had a baby?",
            "{d1} disrupted by {d2} thinking",
            "Bringing {d2} innovation to {d1}",
            "{d1} reimagined through {d2} lens",
            "The intersection of {d1} and {d2}"
        ]

    def _load_business_buzzwords(self) -> List[str]:
        """Business jargon for making ideas sound legit"""
        return [
            "synergistic", "disruptive", "paradigm-shifting", "revolutionary",
            "game-changing", "innovative", "transformative", "cutting-edge",
            "next-generation", "AI-powered", "blockchain-enabled", "quantum-enhanced",
            "sustainable", "scalable", "viral", "exponential", "democratizing"
        ]

    async def collide_domains(
        self,
        domain1: Domain,
        domain2: Domain,
        chaos_level: int
    ) -> CollisionResult:
        """Main collision algorithm"""
        session_id = str(uuid.uuid4())

        # Generate the core idea
        idea = await self._generate_idea(domain1, domain2, chaos_level)

        # Create business plan
        business_plan = await self._generate_business_plan(idea, domain1, domain2)

        # Generate pitch deck
        pitch_deck = await self._generate_pitch_deck(idea, business_plan)

        result = CollisionResult(
            session_id=session_id,
            idea=idea,
            business_plan=business_plan,
            pitch_deck=pitch_deck,
            chaos_level=chaos_level,
            timestamp=datetime.now()
        )

        # Store in memory
        sessions_db[session_id] = result
        ideas_db[idea.id] = idea
        business_plans_db[idea.id] = business_plan
        pitch_decks_db[idea.id] = pitch_deck

        return result

    async def _generate_idea(
        self,
        domain1: Domain,
        domain2: Domain,
        chaos_level: int
    ) -> GeneratedIdea:
        """Generate a crazy idea from domain collision"""
        idea_id = str(uuid.uuid4())

        # Select random concepts from each domain
        concept1 = random.choice(domain1.key_concepts)
        concept2 = random.choice(domain2.key_concepts)
        problem1 = random.choice(domain1.typical_problems)
        tech2 = random.choice(domain2.core_technologies)

        # Generate title based on chaos level
        if chaos_level <= 3:
            title = f"{domain1.name} meets {domain2.name}"
        elif chaos_level <= 6:
            title = f"{concept1.title()} {concept2.title()}"
        else:
            template = random.choice(self.collision_templates)
            title = template.format(d1=domain1.name, d2=domain2.name)

        # Generate tagline
        buzzword = random.choice(self.business_buzzwords)
        tagline = f"The {buzzword} solution for {problem1} using {concept2}"

        # Generate description
        description = self._generate_description(
            domain1, domain2, concept1, concept2, problem1, tech2, chaos_level
        )

        # Generate features
        features = self._generate_features(domain1, domain2, chaos_level)

        # Generate target market
        target_market = self._generate_target_market(domain1, domain2, chaos_level)

        # Calculate scores
        absurdity = min(10, chaos_level + random.uniform(-1, 2))
        viability = max(1, 10 - chaos_level + random.uniform(-2, 2))
        innovation = min(10, chaos_level * 1.2 + random.uniform(-1, 1))

        return GeneratedIdea(
            id=idea_id,
            title=title,
            tagline=tagline,
            description=description,
            key_features=features,
            target_market=target_market,
            absurdity_score=round(absurdity, 1),
            viability_score=round(viability, 1),
            innovation_score=round(innovation, 1),
            from_domains=(domain1.name, domain2.name)
        )

    def _generate_description(
        self,
        d1: Domain,
        d2: Domain,
        concept1: str,
        concept2: str,
        problem1: str,
        tech2: str,
        chaos_level: int
    ) -> str:
        """Generate detailed idea description"""
        if chaos_level <= 3:
            return f"A practical solution that combines {d1.name} with {d2.name} to solve {problem1}. " \
                   f"By leveraging {concept2} and {tech2}, we create a unique approach to {concept1}."
        elif chaos_level <= 6:
            return f"Imagine if {concept1} from {d1.name} was completely reimagined using {concept2} from {d2.name}. " \
                   f"This revolutionary platform addresses {problem1} in ways never before possible, " \
                   f"utilizing {tech2} to create an entirely new paradigm."
        else:
            return f"In a world where {d1.name} and {d2.name} collide, we present the unthinkable: " \
                   f"{concept1} powered by {concept2}. This isn't just solving {problem1}, " \
                   f"it's creating problems we didn't know we had and solving them with {tech2}. " \
                   f"It's {random.choice(self.business_buzzwords)}, {random.choice(self.business_buzzwords)}, " \
                   f"and completely {random.choice(['unnecessary', 'essential', 'inevitable', 'absurd'])}."

    def _generate_features(self, d1: Domain, d2: Domain, chaos_level: int) -> List[str]:
        """Generate key features of the idea"""
        features = []

        # Mix concepts from both domains
        for i in range(min(5, chaos_level)):
            if i % 2 == 0 and d1.key_concepts:
                concept = random.choice(d1.key_concepts)
                tech = random.choice(d2.core_technologies) if d2.core_technologies else "AI"
                features.append(f"{tech}-powered {concept}")
            else:
                concept = random.choice(d2.key_concepts) if d2.key_concepts else "innovation"
                problem = random.choice(d1.typical_problems) if d1.typical_problems else "inefficiency"
                features.append(f"{concept.title()} approach to {problem}")

        # Add chaos-level dependent features
        if chaos_level >= 5:
            features.append("Quantum entanglement for enhanced user experience")
        if chaos_level >= 7:
            features.append("Blockchain verification of every interaction")
        if chaos_level >= 9:
            features.append("AI that questions its own existence while serving you")

        return features[:5]  # Limit to 5 features

    def _generate_target_market(self, d1: Domain, d2: Domain, chaos_level: int) -> str:
        """Generate target market description"""
        if chaos_level <= 3:
            return f"Professionals in {d1.name} interested in {d2.name}"
        elif chaos_level <= 6:
            return f"{d1.name} enthusiasts who secretly love {d2.name}"
        else:
            personas = [
                f"Time-traveling {d1.name} experts from the year 2050",
                f"People who think {d1.name} and {d2.name} are the same thing",
                f"Aliens studying human {d1.name} through {d2.name}",
                f"Anyone brave enough to try"
            ]
            return random.choice(personas)

    async def _generate_business_plan(
        self,
        idea: GeneratedIdea,
        d1: Domain,
        d2: Domain
    ) -> BusinessPlan:
        """Generate a business plan that makes the absurd seem logical"""
        market_size = (d1.market_size or 1000000) + (d2.market_size or 1000000)

        # Revenue models based on both domains
        revenue_models = [
            f"Subscription model for {d1.name} professionals",
            f"Transaction fees on {d2.name} interactions",
            f"Freemium with premium {idea.key_features[0] if idea.key_features else 'features'}",
            f"B2B enterprise licensing for large {d1.name} organizations",
            f"Data monetization from {d2.name} insights"
        ]

        # Risk factors
        risks = [
            "Market may not be ready for this level of innovation",
            f"Regulatory challenges in {d1.name} industry",
            f"User education required for {d2.name} concepts",
            "Potential resistance from traditionalists",
            "Reality might not support our vision"
        ]

        # Implementation steps
        steps = [
            f"Phase 1: Build MVP combining core {d1.name} and {d2.name} features",
            f"Phase 2: Beta test with early adopters who understand both domains",
            f"Phase 3: Iterate based on feedback (ignore if negative)",
            f"Phase 4: Scale to capture {random.randint(1, 10)}% of market",
            f"Phase 5: IPO or acquisition by confused tech giant"
        ]

        return BusinessPlan(
            executive_summary=f"{idea.title} revolutionizes {d1.name} through {d2.name} innovation. "
                            f"With a TAM of ${market_size:,}, we're positioned to capture significant market share.",
            problem_statement=f"Current {d1.name} solutions fail to leverage {d2.name} principles, "
                            f"leaving billions in value uncaptured.",
            solution=idea.description,
            market_opportunity=f"The intersection of {d1.name} (${d1.market_size or 1000000:,} market) "
                             f"and {d2.name} (${d2.market_size or 1000000:,} market) "
                             f"creates a ${market_size * 2:,} opportunity.",
            revenue_model=" + ".join(random.sample(revenue_models, min(3, len(revenue_models)))),
            competitive_advantage=f"First mover advantage in the {d1.name}-{d2.name} convergence space. "
                                 f"Our {random.choice(self.business_buzzwords)} approach is inimitable.",
            implementation_steps=steps,
            funding_required=f"${random.randint(1, 50)}M Series A for product development and market education",
            projected_revenue_year1=f"${random.randint(100000, 1000000):,}",
            projected_revenue_year5=f"${random.randint(10000000, 1000000000):,}",
            risk_factors=risks[:3]
        )

    async def _generate_pitch_deck(
        self,
        idea: GeneratedIdea,
        business_plan: BusinessPlan
    ) -> PitchDeck:
        """Generate a pitch deck for investors"""
        slides = [
            PitchSlide(
                slide_number=1,
                title="Title Slide",
                content=f"{idea.title}\n{idea.tagline}",
                visual_suggestion="Collision of two worlds visualized"
            ),
            PitchSlide(
                slide_number=2,
                title="The Problem",
                content=business_plan.problem_statement,
                visual_suggestion="Graph showing massive inefficiency"
            ),
            PitchSlide(
                slide_number=3,
                title="Our Solution",
                content=idea.description,
                visual_suggestion="Elegant diagram of our approach"
            ),
            PitchSlide(
                slide_number=4,
                title="Market Opportunity",
                content=business_plan.market_opportunity,
                visual_suggestion="TAM/SAM/SOM pyramid"
            ),
            PitchSlide(
                slide_number=5,
                title="Product Demo",
                content=f"Key Features:\n" + "\n".join(f"• {f}" for f in idea.key_features),
                visual_suggestion="Screenshots that don't exist yet"
            ),
            PitchSlide(
                slide_number=6,
                title="Business Model",
                content=business_plan.revenue_model,
                visual_suggestion="Revenue flow diagram"
            ),
            PitchSlide(
                slide_number=7,
                title="Go-to-Market Strategy",
                content=f"Target Market: {idea.target_market}\nStrategy: Viral growth through confusion",
                visual_suggestion="Hockey stick growth chart"
            ),
            PitchSlide(
                slide_number=8,
                title="Competitive Advantage",
                content=business_plan.competitive_advantage,
                visual_suggestion="Us vs. Nobody (we're alone here)"
            ),
            PitchSlide(
                slide_number=9,
                title="Financial Projections",
                content=f"Year 1: {business_plan.projected_revenue_year1}\n"
                        f"Year 5: {business_plan.projected_revenue_year5}",
                visual_suggestion="Exponential growth curve"
            ),
            PitchSlide(
                slide_number=10,
                title="The Ask",
                content=business_plan.funding_required + "\nJoin us in creating the future!",
                visual_suggestion="Rocket ship to the moon"
            )
        ]

        hook = f"What if I told you that {idea.from_domains[0]} and {idea.from_domains[1]} " \
               f"are about to converge in a {random.choice(self.business_buzzwords)} way?"

        return PitchDeck(
            idea_id=idea.id,
            hook=hook,
            slides=slides,
            call_to_action="Invest now or regret forever when this becomes a unicorn"
        )

# Initialize engine
chaos_engine = ChaosEngine()

# API Endpoints
@app.post("/api/collisions/generate")
async def generate_collision(request: CollisionRequest) -> CollisionResult:
    """Generate a domain collision idea"""
    # Select domains
    if request.custom_domain1 and request.custom_domain2:
        # Create custom domains
        domain1 = Domain(
            name=request.custom_domain1,
            category="Custom",
            key_concepts=["innovation", "disruption", "transformation"],
            typical_problems=["inefficiency", "cost", "complexity"],
            core_technologies=["AI", "automation", "optimization"],
            market_size=1000000000
        )
        domain2 = Domain(
            name=request.custom_domain2,
            category="Custom",
            key_concepts=["revolution", "paradigm shift", "breakthrough"],
            typical_problems=["tradition", "resistance", "limitations"],
            core_technologies=["next-gen", "cutting-edge", "emerging"],
            market_size=1000000000
        )
    else:
        # Use random domains
        domain_names = list(DOMAINS_DB.keys())
        d1_name = request.domain1_id or random.choice(domain_names)
        d2_name = request.domain2_id or random.choice([d for d in domain_names if d != d1_name])
        domain1 = DOMAINS_DB[d1_name]
        domain2 = DOMAINS_DB[d2_name]

    # Generate collision
    result = await chaos_engine.collide_domains(domain1, domain2, request.chaos_level)

    return result

@app.get("/api/domains")
async def list_domains() -> Dict[str, Domain]:
    """List all available domains"""
    return DOMAINS_DB

@app.get("/api/domains/random-pair")
async def get_random_pair() -> Dict[str, Domain]:
    """Get a random pair of domains"""
    domain_names = list(DOMAINS_DB.keys())
    d1_name = random.choice(domain_names)
    d2_name = random.choice([d for d in domain_names if d != d1_name])

    return {
        "domain1": DOMAINS_DB[d1_name],
        "domain2": DOMAINS_DB[d2_name]
    }

@app.post("/api/collisions/{session_id}/amplify")
async def amplify_chaos(session_id: str, amplification: Dict[str, float]) -> CollisionResult:
    """Make an idea even more chaotic"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    original = sessions_db[session_id]
    amplification_factor = amplification.get("factor", 1.5)

    # Increase chaos
    new_chaos = min(10, int(original.chaos_level * amplification_factor))

    # Re-generate with more chaos
    domain_names = list(DOMAINS_DB.keys())
    d1 = DOMAINS_DB[random.choice(domain_names)]
    d2 = DOMAINS_DB[random.choice(domain_names)]

    amplified = await chaos_engine.collide_domains(d1, d2, new_chaos)
    amplified.idea.title = f"EXTREME: {original.idea.title}"
    amplified.idea.absurdity_score = min(10, original.idea.absurdity_score * amplification_factor)

    return amplified

@app.get("/api/ideas/{idea_id}")
async def get_idea_details(idea_id: str) -> Dict[str, Any]:
    """Get full idea details"""
    if idea_id not in ideas_db:
        raise HTTPException(status_code=404, detail="Idea not found")

    return {
        "idea": ideas_db[idea_id],
        "business_plan": business_plans_db.get(idea_id),
        "pitch_deck": pitch_decks_db.get(idea_id)
    }

@app.post("/api/ideas/{idea_id}/vote")
async def vote_on_idea(idea_id: str, vote: Dict[str, Any]) -> Dict[str, Any]:
    """Vote on an idea"""
    if idea_id not in ideas_db:
        raise HTTPException(status_code=404, detail="Idea not found")

    # Simulate vote tracking
    vote_type = vote.get("vote_type", "genius")
    would_invest = vote.get("would_invest", False)

    return {
        "message": f"Vote recorded: {vote_type}",
        "total_votes": random.randint(10, 1000),
        "investment_interest": random.randint(0, 100) if would_invest else 0,
        "chaos_rating": "Maximum chaos achieved" if vote_type == "genius" else "Needs more chaos"
    }

@app.get("/api/hall-of-chaos")
async def get_hall_of_chaos() -> Dict[str, Any]:
    """Get the most successful chaotic ideas"""
    hall_of_fame = [
        {
            "title": "Blockchain Meditation",
            "description": "Mine cryptocurrency with your thoughts",
            "votes": 42069,
            "funding_raised": "$4.2M",
            "status": "Somehow operational"
        },
        {
            "title": "Quantum Dating",
            "description": "Find your soulmate in parallel universes",
            "votes": 13337,
            "funding_raised": "$2.1M",
            "status": "In beta across dimensions"
        },
        {
            "title": "Funeral Food Truck",
            "description": "Drive-through grief counseling with tacos",
            "votes": 9001,
            "funding_raised": "$500K",
            "status": "Rolling out nationwide"
        }
    ]

    return {
        "hall_of_fame": hall_of_fame,
        "total_ideas_generated": 1337420,
        "ideas_funded": 69,
        "unicorns_created": 0,
        "reality_distortions": 42
    }

@app.post("/api/demo/chaos")
async def demo_chaos() -> CollisionResult:
    """Demo endpoint for instant chaos"""
    # Pick two random incompatible domains
    domains = ["blockchain", "meditation", "fast_food", "funeral_services", "extreme_sports"]
    d1_name = random.choice(domains)
    d2_name = random.choice([d for d in domains if d != d1_name])

    domain1 = DOMAINS_DB[d1_name]
    domain2 = DOMAINS_DB[d2_name]

    # Maximum chaos
    result = await chaos_engine.collide_domains(domain1, domain2, chaos_level=8)

    return result

@app.get("/api/inspiration")
async def get_inspiration() -> Dict[str, str]:
    """Get inspirational quotes about chaos"""
    quotes = [
        "The best ideas come from the worst combinations",
        "Chaos is just innovation wearing a disguise",
        "If it makes sense, you're not thinking hard enough",
        "Every unicorn started as a ridiculous idea",
        "The future belongs to those who embrace the absurd",
        "Normal is just a setting on a washing machine"
    ]

    return {
        "quote": random.choice(quotes),
        "chaos_level": random.randint(1, 10),
        "inspiration_type": random.choice(["motivational", "confusing", "profound", "nonsensical"])
    }

if __name__ == "__main__":
    import uvicorn
    print("🌀 CHAOS ENGINE INITIALIZED")
    print("💥 Preparing to collide incompatible domains...")
    print("🚀 Innovation through absurdity starts now!")
    uvicorn.run(app, host="0.0.0.0", port=8002)