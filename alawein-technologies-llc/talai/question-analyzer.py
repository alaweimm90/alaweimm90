#!/usr/bin/env python3
"""
Question Analysis & Research Planning System
Analyzes generated questions and creates targeted research/development plans
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter, defaultdict
import re

class QuestionAnalyzer:
    def __init__(self):
        self.questions_file = Path(".meta/project-questions.json")
        self.questions_data = self._load_questions()

    def _load_questions(self) -> Dict:
        """Load generated questions"""
        if self.questions_file.exists():
            with open(self.questions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def analyze_question_patterns(self) -> Dict:
        """Analyze patterns across all questions"""
        all_questions = []
        question_categories = defaultdict(list)
        domain_questions = defaultdict(list)

        for org_name, org_data in self.questions_data.items():
            for project_name, project_data in org_data.items():
                questions = project_data["questions"]
                domain = project_data["profile"]["domain"]

                all_questions.extend(questions)
                domain_questions[domain].extend(questions)

                # Categorize questions
                for question in questions:
                    category = self._categorize_question(question)
                    question_categories[category].append(question)

        # Analyze patterns
        word_frequency = self._analyze_word_frequency(all_questions)
        question_types = self._analyze_question_types(all_questions)
        complexity_analysis = self._analyze_complexity(all_questions)

        return {
            "total_questions": len(all_questions),
            "categories": {cat: len(qs) for cat, qs in question_categories.items()},
            "domains": {domain: len(qs) for domain, qs in domain_questions.items()},
            "word_frequency": dict(word_frequency.most_common(50)),
            "question_types": question_types,
            "complexity": complexity_analysis,
            "patterns": self._identify_patterns(all_questions)
        }

    def _categorize_question(self, question: str) -> str:
        """Categorize a question based on its content"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["how", "method", "approach", "strategy"]):
            return "methodology"
        elif any(word in question_lower for word in ["what", "which", "define", "identify"]):
            return "definition"
        elif any(word in question_lower for word in ["why", "reason", "cause", "purpose"]):
            return "rationale"
        elif any(word in question_lower for word in ["when", "timeline", "schedule", "deadline"]):
            return "timing"
        elif any(word in question_lower for word in ["where", "location", "place", "environment"]):
            return "context"
        elif any(word in question_lower for word in ["who", "stakeholder", "user", "customer"]):
            return "stakeholder"
        elif any(word in question_lower for word in ["measure", "metric", "evaluate", "assess"]):
            return "measurement"
        elif any(word in question_lower for word in ["risk", "challenge", "problem", "issue"]):
            return "risk_management"
        elif any(word in question_lower for word in ["optimize", "improve", "enhance", "better"]):
            return "optimization"
        else:
            return "general"

    def _analyze_word_frequency(self, questions: List[str]) -> Counter:
        """Analyze word frequency across all questions"""
        words = []
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "must", "shall", "we", "you", "they", "it", "this", "that"
        }

        for question in questions:
            # Extract words, convert to lowercase, remove punctuation
            question_words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
            words.extend([word for word in question_words if word not in stop_words and len(word) > 2])

        return Counter(words)

    def _analyze_question_types(self, questions: List[str]) -> Dict:
        """Analyze types of questions (open-ended, yes/no, etc.)"""
        types = {
            "open_ended": 0,
            "yes_no": 0,
            "multiple_choice": 0,
            "quantitative": 0,
            "comparative": 0
        }

        for question in questions:
            question_lower = question.lower()

            if question_lower.startswith(("how", "what", "why", "when", "where", "which")):
                types["open_ended"] += 1
            elif any(word in question_lower for word in ["is", "are", "can", "will", "should", "does"]):
                types["yes_no"] += 1
            elif any(word in question_lower for word in ["how many", "how much", "what percentage"]):
                types["quantitative"] += 1
            elif any(word in question_lower for word in ["better", "worse", "compare", "versus", "vs"]):
                types["comparative"] += 1
            else:
                types["multiple_choice"] += 1

        return types

    def _analyze_complexity(self, questions: List[str]) -> Dict:
        """Analyze question complexity"""
        complexity_scores = []

        for question in questions:
            # Simple complexity scoring based on length and structure
            word_count = len(question.split())
            clause_count = question.count(",") + question.count(";") + 1
            complexity = word_count * 0.1 + clause_count * 0.5
            complexity_scores.append(complexity)

        avg_complexity = sum(complexity_scores) / len(complexity_scores)

        return {
            "average_complexity": avg_complexity,
            "simple_questions": len([s for s in complexity_scores if s < 2]),
            "moderate_questions": len([s for s in complexity_scores if 2 <= s < 4]),
            "complex_questions": len([s for s in complexity_scores if s >= 4])
        }

    def _identify_patterns(self, questions: List[str]) -> Dict:
        """Identify common patterns in questions"""
        patterns = {
            "research_focused": len([q for q in questions if any(word in q.lower() for word in ["research", "study", "analyze", "investigate"])]),
            "implementation_focused": len([q for q in questions if any(word in q.lower() for word in ["implement", "build", "create", "develop"])]),
            "evaluation_focused": len([q for q in questions if any(word in q.lower() for word in ["evaluate", "measure", "assess", "test"])]),
            "strategy_focused": len([q for q in questions if any(word in q.lower() for word in ["strategy", "plan", "approach", "method"])]),
            "stakeholder_focused": len([q for q in questions if any(word in q.lower() for word in ["user", "customer", "stakeholder", "client"])]),
            "technical_focused": len([q for q in questions if any(word in q.lower() for word in ["technical", "architecture", "system", "technology"])]),
            "business_focused": len([q for q in questions if any(word in q.lower() for word in ["business", "market", "revenue", "profit"])]),
            "risk_focused": len([q for q in questions if any(word in q.lower() for word in ["risk", "challenge", "problem", "issue"])])
        }

        return patterns

    def generate_research_plan(self, org_name: str, project_name: str) -> Dict:
        """Generate a targeted research plan for a specific project"""
        if org_name not in self.questions_data or project_name not in self.questions_data[org_name]:
            return {"error": "Project not found"}

        project_data = self.questions_data[org_name][project_name]
        questions = project_data["questions"]
        profile = project_data["profile"]

        # Categorize questions for research planning
        categorized_questions = defaultdict(list)
        for question in questions:
            category = self._categorize_question(question)
            categorized_questions[category].append(question)

        # Generate research phases
        research_phases = self._generate_research_phases(categorized_questions, profile)

        # Prioritize questions
        priority_questions = self._prioritize_questions(questions, profile)

        # Generate methodology recommendations
        methodology = self._recommend_methodology(profile, categorized_questions)

        return {
            "project": {
                "name": project_name,
                "organization": org_name,
                "domain": profile["domain"],
                "focus_areas": profile["focus_areas"]
            },
            "research_phases": research_phases,
            "priority_questions": priority_questions,
            "methodology": methodology,
            "timeline": self._generate_timeline(research_phases),
            "resources": self._recommend_resources(profile),
            "success_metrics": self._define_success_metrics(profile)
        }

    def _generate_research_phases(self, categorized_questions: Dict, profile: Dict) -> List[Dict]:
        """Generate research phases based on question categories"""
        phases = []

        # Phase 1: Discovery & Definition
        if "definition" in categorized_questions or "rationale" in categorized_questions:
            phases.append({
                "phase": 1,
                "name": "Discovery & Definition",
                "description": "Define scope, objectives, and foundational understanding",
                "questions": (categorized_questions.get("definition", []) +
                            categorized_questions.get("rationale", []))[:20],
                "duration": "2-4 weeks",
                "deliverables": ["Project charter", "Requirements document", "Stakeholder analysis"]
            })

        # Phase 2: Research & Analysis
        if "methodology" in categorized_questions:
            phases.append({
                "phase": 2,
                "name": "Research & Analysis",
                "description": "Conduct research and analyze findings",
                "questions": categorized_questions.get("methodology", [])[:25],
                "duration": "4-8 weeks",
                "deliverables": ["Research findings", "Analysis report", "Recommendations"]
            })

        # Phase 3: Implementation Planning
        if "context" in categorized_questions or "timing" in categorized_questions:
            phases.append({
                "phase": 3,
                "name": "Implementation Planning",
                "description": "Plan implementation strategy and timeline",
                "questions": (categorized_questions.get("context", []) +
                            categorized_questions.get("timing", []))[:20],
                "duration": "2-3 weeks",
                "deliverables": ["Implementation plan", "Timeline", "Resource allocation"]
            })

        # Phase 4: Execution & Monitoring
        if "measurement" in categorized_questions or "optimization" in categorized_questions:
            phases.append({
                "phase": 4,
                "name": "Execution & Monitoring",
                "description": "Execute plan and monitor progress",
                "questions": (categorized_questions.get("measurement", []) +
                            categorized_questions.get("optimization", []))[:25],
                "duration": "6-12 weeks",
                "deliverables": ["Progress reports", "Performance metrics", "Optimization recommendations"]
            })

        # Phase 5: Evaluation & Improvement
        if "risk_management" in categorized_questions:
            phases.append({
                "phase": 5,
                "name": "Evaluation & Improvement",
                "description": "Evaluate results and plan improvements",
                "questions": categorized_questions.get("risk_management", [])[:15],
                "duration": "2-4 weeks",
                "deliverables": ["Evaluation report", "Lessons learned", "Improvement plan"]
            })

        return phases

    def _prioritize_questions(self, questions: List[str], profile: Dict) -> Dict:
        """Prioritize questions based on project profile"""
        high_priority = []
        medium_priority = []
        low_priority = []

        domain = profile["domain"]
        focus_areas = profile["focus_areas"]

        for question in questions:
            question_lower = question.lower()
            priority_score = 0

            # Domain-specific prioritization
            if domain == "research":
                if any(word in question_lower for word in ["methodology", "data", "analysis", "validation"]):
                    priority_score += 3
            elif domain == "business":
                if any(word in question_lower for word in ["market", "customer", "revenue", "growth"]):
                    priority_score += 3
            elif domain == "technology":
                if any(word in question_lower for word in ["architecture", "performance", "security", "scalability"]):
                    priority_score += 3

            # Focus area prioritization
            for focus_area in focus_areas:
                if focus_area.replace("_", " ") in question_lower:
                    priority_score += 2

            # General prioritization
            if any(word in question_lower for word in ["critical", "essential", "key", "primary"]):
                priority_score += 2
            elif any(word in question_lower for word in ["optimize", "improve", "enhance"]):
                priority_score += 1

            # Categorize by priority
            if priority_score >= 4:
                high_priority.append(question)
            elif priority_score >= 2:
                medium_priority.append(question)
            else:
                low_priority.append(question)

        return {
            "high_priority": high_priority[:50],
            "medium_priority": medium_priority[:100],
            "low_priority": low_priority[:100]
        }

    def _recommend_methodology(self, profile: Dict, categorized_questions: Dict) -> Dict:
        """Recommend research methodology based on project profile"""
        domain = profile["domain"]

        if domain == "research":
            return {
                "primary_method": "Scientific Research",
                "approaches": ["Literature review", "Experimental design", "Data analysis", "Peer review"],
                "tools": ["Statistical software", "Research databases", "Lab equipment", "Collaboration platforms"],
                "validation": ["Reproducibility", "Peer review", "Statistical significance", "Ethical approval"]
            }
        elif domain == "business":
            return {
                "primary_method": "Market Research",
                "approaches": ["Market analysis", "Customer interviews", "Competitive analysis", "Financial modeling"],
                "tools": ["Survey platforms", "Analytics tools", "CRM systems", "Financial software"],
                "validation": ["Market validation", "Customer feedback", "Financial metrics", "A/B testing"]
            }
        elif domain == "technology":
            return {
                "primary_method": "Agile Development",
                "approaches": ["Iterative development", "User testing", "Performance testing", "Code review"],
                "tools": ["Development frameworks", "Testing tools", "CI/CD pipelines", "Monitoring systems"],
                "validation": ["Unit testing", "Integration testing", "Performance benchmarks", "Security audits"]
            }
        else:
            return {
                "primary_method": "Exploratory Learning",
                "approaches": ["Self-directed learning", "Practical application", "Community engagement", "Mentorship"],
                "tools": ["Learning platforms", "Documentation tools", "Practice environments", "Community forums"],
                "validation": ["Skill assessment", "Project completion", "Peer feedback", "Portfolio development"]
            }

    def _generate_timeline(self, research_phases: List[Dict]) -> Dict:
        """Generate project timeline"""
        total_duration = sum(
            int(phase["duration"].split("-")[1].split()[0])
            for phase in research_phases
        )

        return {
            "total_duration_weeks": total_duration,
            "phases": [
                {
                    "phase": phase["phase"],
                    "name": phase["name"],
                    "duration": phase["duration"],
                    "start_week": sum(
                        int(research_phases[i]["duration"].split("-")[1].split()[0])
                        for i in range(phase["phase"] - 1)
                    ) + 1
                }
                for phase in research_phases
            ]
        }

    def _recommend_resources(self, profile: Dict) -> List[str]:
        """Recommend resources based on project profile"""
        domain = profile["domain"]

        if domain == "research":
            return [
                "Academic databases (PubMed, IEEE, ACM)",
                "Statistical software (R, Python, SPSS)",
                "Research collaboration tools",
                "Laboratory equipment",
                "Funding sources"
            ]
        elif domain == "business":
            return [
                "Market research platforms",
                "Customer analytics tools",
                "Financial modeling software",
                "Business intelligence platforms",
                "Industry reports and databases"
            ]
        elif domain == "technology":
            return [
                "Development environments",
                "Cloud computing platforms",
                "Testing frameworks",
                "Monitoring and analytics tools",
                "Documentation platforms"
            ]
        else:
            return [
                "Online learning platforms",
                "Practice environments",
                "Community forums",
                "Mentorship programs",
                "Portfolio platforms"
            ]

    def _define_success_metrics(self, profile: Dict) -> List[str]:
        """Define success metrics based on project profile"""
        domain = profile["domain"]

        if domain == "research":
            return [
                "Publications in peer-reviewed journals",
                "Citation impact",
                "Reproducibility of results",
                "Grant funding secured",
                "Collaboration networks established"
            ]
        elif domain == "business":
            return [
                "Revenue growth",
                "Customer acquisition",
                "Market share",
                "Customer satisfaction",
                "Return on investment"
            ]
        elif domain == "technology":
            return [
                "System performance",
                "User adoption",
                "Code quality metrics",
                "Security compliance",
                "Maintenance efficiency"
            ]
        else:
            return [
                "Skill development progress",
                "Project completion rate",
                "Community engagement",
                "Portfolio quality",
                "Career advancement"
            ]

def main():
    """Main entry point"""
    analyzer = QuestionAnalyzer()

    print("[QUESTION-ANALYZER] Analyzing question patterns...")

    # Analyze overall patterns
    patterns = analyzer.analyze_question_patterns()

    print(f"\n[ANALYSIS] Question Analysis Results:")
    print(f"  Total Questions: {patterns['total_questions']}")
    print(f"  Categories: {len(patterns['categories'])}")
    print(f"  Domains: {len(patterns['domains'])}")

    print(f"\n[CATEGORIES] Question Distribution:")
    for category, count in sorted(patterns['categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")

    print(f"\n[DOMAINS] Questions by Domain:")
    for domain, count in patterns['domains'].items():
        print(f"  {domain}: {count}")

    print(f"\n[TOP-WORDS] Most Frequent Words:")
    for word, count in list(patterns['word_frequency'].items())[:10]:
        print(f"  {word}: {count}")

    # Save analysis results
    analysis_file = Path(".meta/question-analysis.json")
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    print(f"\n[OUTPUT] Analysis saved to: {analysis_file}")

    # Generate sample research plan
    if analyzer.questions_data:
        org_name = list(analyzer.questions_data.keys())[0]
        project_name = list(analyzer.questions_data[org_name].keys())[0]

        research_plan = analyzer.generate_research_plan(org_name, project_name)

        plan_file = Path(f".meta/research-plan-{org_name}-{project_name}.json")
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(research_plan, f, indent=2, ensure_ascii=False)

        print(f"\n[SAMPLE] Research plan generated for {org_name}/{project_name}")
        print(f"  Phases: {len(research_plan.get('research_phases', []))}")
        print(f"  Timeline: {research_plan.get('timeline', {}).get('total_duration_weeks', 0)} weeks")
        print(f"  Plan saved to: {plan_file}")

if __name__ == "__main__":
    main()
