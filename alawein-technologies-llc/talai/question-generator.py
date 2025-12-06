#!/usr/bin/env python3
"""
Comprehensive Question Generator
Generates 500+ targeted questions for each project based on domain and type
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ProjectProfile:
    name: str
    organization: str
    domain: str
    type: str
    focus_areas: List[str]
    stakeholders: List[str]

class QuestionGenerator:
    def __init__(self):
        self.question_templates = self._load_question_templates()

    def _load_question_templates(self) -> Dict:
        """Load question templates by domain and category"""
        return {
            "research": {
                "methodology": [
                    "What research methodology will be most effective for {focus_area}?",
                    "How can we validate the hypothesis for {focus_area}?",
                    "What are the key variables to control in {focus_area} experiments?",
                    "How do we ensure reproducibility in {focus_area} research?",
                    "What statistical methods are appropriate for {focus_area} data?",
                    "How can we minimize bias in {focus_area} studies?",
                    "What sample size is needed for statistically significant results in {focus_area}?",
                    "How do we handle confounding variables in {focus_area} research?",
                    "What are the ethical considerations for {focus_area} research?",
                    "How can we ensure data quality in {focus_area} experiments?"
                ],
                "literature": [
                    "What are the seminal papers in {focus_area}?",
                    "Who are the leading researchers in {focus_area}?",
                    "What are the current gaps in {focus_area} literature?",
                    "How has {focus_area} evolved over the past decade?",
                    "What are the most cited works in {focus_area}?",
                    "What theoretical frameworks exist for {focus_area}?",
                    "What are the emerging trends in {focus_area} research?",
                    "How do different schools of thought approach {focus_area}?",
                    "What are the key debates in {focus_area}?",
                    "What interdisciplinary connections exist with {focus_area}?"
                ],
                "data": [
                    "What data sources are available for {focus_area}?",
                    "How can we collect high-quality data for {focus_area}?",
                    "What are the data privacy considerations for {focus_area}?",
                    "How do we handle missing data in {focus_area} datasets?",
                    "What data preprocessing is needed for {focus_area}?",
                    "How can we ensure data integrity in {focus_area}?",
                    "What are the storage requirements for {focus_area} data?",
                    "How do we version control {focus_area} datasets?",
                    "What metadata standards apply to {focus_area}?",
                    "How can we make {focus_area} data FAIR (Findable, Accessible, Interoperable, Reusable)?"
                ]
            },
            "business": {
                "market": [
                    "What is the total addressable market for {focus_area}?",
                    "Who are the key competitors in {focus_area}?",
                    "What are the market trends affecting {focus_area}?",
                    "How price-sensitive are customers in {focus_area}?",
                    "What are the barriers to entry in {focus_area}?",
                    "How is the {focus_area} market segmented?",
                    "What are the growth projections for {focus_area}?",
                    "How do regulatory changes impact {focus_area}?",
                    "What are the seasonal patterns in {focus_area}?",
                    "How does economic climate affect {focus_area} demand?"
                ],
                "customer": [
                    "Who is the ideal customer for {focus_area}?",
                    "What pain points does {focus_area} solve?",
                    "How do customers currently solve {focus_area} problems?",
                    "What is the customer journey for {focus_area}?",
                    "How do customers discover {focus_area} solutions?",
                    "What influences customer decisions in {focus_area}?",
                    "How do we measure customer satisfaction in {focus_area}?",
                    "What are the customer retention strategies for {focus_area}?",
                    "How do customer needs vary by segment in {focus_area}?",
                    "What are the switching costs for {focus_area} customers?"
                ],
                "revenue": [
                    "What revenue models work best for {focus_area}?",
                    "How do we price {focus_area} offerings?",
                    "What are the unit economics for {focus_area}?",
                    "How can we increase customer lifetime value in {focus_area}?",
                    "What are the key revenue drivers for {focus_area}?",
                    "How do we forecast revenue for {focus_area}?",
                    "What are the monetization opportunities in {focus_area}?",
                    "How do we optimize pricing for {focus_area}?",
                    "What are the revenue recognition rules for {focus_area}?",
                    "How do we measure revenue quality in {focus_area}?"
                ]
            },
            "technology": {
                "architecture": [
                    "What architecture patterns best suit {focus_area}?",
                    "How do we ensure scalability in {focus_area}?",
                    "What are the performance requirements for {focus_area}?",
                    "How do we handle fault tolerance in {focus_area}?",
                    "What security considerations apply to {focus_area}?",
                    "How do we design for maintainability in {focus_area}?",
                    "What are the integration requirements for {focus_area}?",
                    "How do we ensure data consistency in {focus_area}?",
                    "What monitoring and observability do we need for {focus_area}?",
                    "How do we plan for disaster recovery in {focus_area}?"
                ],
                "implementation": [
                    "What technology stack is optimal for {focus_area}?",
                    "How do we ensure code quality in {focus_area}?",
                    "What testing strategies work best for {focus_area}?",
                    "How do we manage dependencies in {focus_area}?",
                    "What are the deployment strategies for {focus_area}?",
                    "How do we handle configuration management in {focus_area}?",
                    "What are the performance optimization techniques for {focus_area}?",
                    "How do we ensure backward compatibility in {focus_area}?",
                    "What are the debugging strategies for {focus_area}?",
                    "How do we manage technical debt in {focus_area}?"
                ]
            },
            "personal": {
                "goals": [
                    "What are the learning objectives for {focus_area}?",
                    "How does {focus_area} align with career goals?",
                    "What skills need development in {focus_area}?",
                    "How can {focus_area} contribute to personal growth?",
                    "What are the success metrics for {focus_area}?",
                    "How does {focus_area} fit into long-term plans?",
                    "What challenges might arise in {focus_area}?",
                    "How can {focus_area} build expertise?",
                    "What networking opportunities exist in {focus_area}?",
                    "How can {focus_area} enhance professional reputation?"
                ],
                "learning": [
                    "What resources are available for learning {focus_area}?",
                    "How can {focus_area} knowledge be applied practically?",
                    "What are the best practices in {focus_area}?",
                    "How can {focus_area} skills be demonstrated?",
                    "What communities exist around {focus_area}?",
                    "How can {focus_area} learning be structured?",
                    "What are the common pitfalls in {focus_area}?",
                    "How can {focus_area} progress be tracked?",
                    "What mentorship opportunities exist in {focus_area}?",
                    "How can {focus_area} knowledge be shared?"
                ]
            }
        }

    def generate_questions_for_project(self, project: ProjectProfile) -> List[str]:
        """Generate 500+ questions for a specific project"""
        questions = []

        # Base questions for all projects
        questions.extend(self._generate_base_questions(project))

        # Domain-specific questions
        if project.domain in self.question_templates:
            questions.extend(self._generate_domain_questions(project))

        # Focus area specific questions
        for focus_area in project.focus_areas:
            questions.extend(self._generate_focus_questions(project, focus_area))

        # Stakeholder questions
        for stakeholder in project.stakeholders:
            questions.extend(self._generate_stakeholder_questions(project, stakeholder))

        # Ensure we have at least 500 questions
        while len(questions) < 500:
            questions.extend(self._generate_additional_questions(project, len(questions)))

        return questions[:500]  # Return exactly 500 questions

    def _generate_base_questions(self, project: ProjectProfile) -> List[str]:
        """Generate base questions applicable to all projects"""
        return [
            f"What is the primary purpose of {project.name}?",
            f"Who is the target audience for {project.name}?",
            f"What problem does {project.name} solve?",
            f"How does {project.name} create value?",
            f"What are the success criteria for {project.name}?",
            f"What resources are needed for {project.name}?",
            f"What are the risks associated with {project.name}?",
            f"How will {project.name} be maintained?",
            f"What is the timeline for {project.name}?",
            f"How will {project.name} be evaluated?",
            f"What are the dependencies for {project.name}?",
            f"How does {project.name} fit into the broader ecosystem?",
            f"What are the quality standards for {project.name}?",
            f"How will {project.name} handle scalability?",
            f"What documentation is needed for {project.name}?",
            f"How will {project.name} ensure accessibility?",
            f"What are the performance requirements for {project.name}?",
            f"How will {project.name} handle security?",
            f"What are the compliance requirements for {project.name}?",
            f"How will {project.name} be tested?"
        ]

    def _generate_domain_questions(self, project: ProjectProfile) -> List[str]:
        """Generate domain-specific questions"""
        questions = []
        domain_templates = self.question_templates.get(project.domain, {})

        for category, templates in domain_templates.items():
            for template in templates:
                # Generate variations for each focus area
                for focus_area in project.focus_areas:
                    questions.append(template.format(focus_area=focus_area))

        return questions

    def _generate_focus_questions(self, project: ProjectProfile, focus_area: str) -> List[str]:
        """Generate questions specific to a focus area"""
        return [
            f"How does {focus_area} impact {project.name}?",
            f"What are the best practices for {focus_area} in {project.name}?",
            f"How can {focus_area} be optimized in {project.name}?",
            f"What are the challenges with {focus_area} in {project.name}?",
            f"How do we measure success in {focus_area} for {project.name}?",
            f"What tools are available for {focus_area} in {project.name}?",
            f"How does {focus_area} integrate with other components in {project.name}?",
            f"What are the future trends in {focus_area} for {project.name}?",
            f"How can we innovate in {focus_area} for {project.name}?",
            f"What are the cost implications of {focus_area} in {project.name}?"
        ]

    def _generate_stakeholder_questions(self, project: ProjectProfile, stakeholder: str) -> List[str]:
        """Generate stakeholder-specific questions"""
        return [
            f"What are {stakeholder}'s expectations for {project.name}?",
            f"How does {project.name} benefit {stakeholder}?",
            f"What concerns might {stakeholder} have about {project.name}?",
            f"How can we engage {stakeholder} in {project.name}?",
            f"What feedback has {stakeholder} provided on {project.name}?",
            f"How do we communicate with {stakeholder} about {project.name}?",
            f"What role does {stakeholder} play in {project.name}?",
            f"How can we address {stakeholder}'s needs in {project.name}?",
            f"What are {stakeholder}'s success metrics for {project.name}?",
            f"How can we improve {stakeholder} satisfaction with {project.name}?"
        ]

    def _generate_additional_questions(self, project: ProjectProfile, current_count: int) -> List[str]:
        """Generate additional questions to reach 500"""
        additional_categories = [
            "innovation", "sustainability", "ethics", "collaboration",
            "communication", "training", "support", "feedback",
            "optimization", "automation", "integration", "migration"
        ]

        questions = []
        for category in additional_categories:
            questions.extend([
                f"How can we improve {category} in {project.name}?",
                f"What are the {category} requirements for {project.name}?",
                f"How do we measure {category} success in {project.name}?",
                f"What tools support {category} in {project.name}?",
                f"What are the {category} best practices for {project.name}?"
            ])

        return questions

    def analyze_organization_projects(self, org_path: Path) -> List[ProjectProfile]:
        """Analyze projects in an organization"""
        projects = []

        if not org_path.exists():
            return projects

        org_name = org_path.name

        # Determine organization domain
        domain = self._determine_domain(org_name)

        # Find projects in organization
        for item in org_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if self._is_project(item):
                    project = ProjectProfile(
                        name=item.name,
                        organization=org_name,
                        domain=domain,
                        type=self._determine_project_type(item),
                        focus_areas=self._extract_focus_areas(item, domain),
                        stakeholders=self._identify_stakeholders(item, domain)
                    )
                    projects.append(project)

        return projects

    def _determine_domain(self, org_name: str) -> str:
        """Determine organization domain"""
        name_lower = org_name.lower()

        if "science" in name_lower or "physics" in name_lower:
            return "research"
        elif "business" in name_lower:
            return "business"
        elif "tools" in name_lower:
            return "technology"
        elif "testing" in name_lower:
            return "personal"
        else:
            return "technology"

    def _determine_project_type(self, project_path: Path) -> str:
        """Determine project type"""
        if (project_path / "pyproject.toml").exists():
            return "python_package"
        elif (project_path / "package.json").exists():
            return "javascript_project"
        elif (project_path / "Cargo.toml").exists():
            return "rust_project"
        elif (project_path / "go.mod").exists():
            return "go_project"
        else:
            return "general_project"

    def _extract_focus_areas(self, project_path: Path, domain: str) -> List[str]:
        """Extract focus areas from project"""
        focus_areas = []

        # Domain-specific focus areas
        if domain == "research":
            focus_areas = ["data_analysis", "experimentation", "modeling", "validation", "publication"]
        elif domain == "business":
            focus_areas = ["market_analysis", "customer_acquisition", "revenue_optimization", "operations", "growth"]
        elif domain == "technology":
            focus_areas = ["architecture", "performance", "security", "scalability", "maintainability"]
        else:
            focus_areas = ["learning", "skill_development", "experimentation", "documentation", "sharing"]

        return focus_areas

    def _identify_stakeholders(self, project_path: Path, domain: str) -> List[str]:
        """Identify project stakeholders"""
        if domain == "research":
            return ["researchers", "reviewers", "collaborators", "funding_agencies", "academic_community"]
        elif domain == "business":
            return ["customers", "investors", "employees", "partners", "regulators"]
        elif domain == "technology":
            return ["developers", "users", "maintainers", "contributors", "administrators"]
        else:
            return ["learners", "mentors", "peers", "community", "employers"]

    def _is_project(self, path: Path) -> bool:
        """Check if directory is a project"""
        indicators = [
            "README.md", "pyproject.toml", "package.json",
            "Cargo.toml", "go.mod", "Makefile"
        ]
        return any((path / indicator).exists() for indicator in indicators)

    def generate_all_questions(self) -> Dict:
        """Generate questions for all projects in all organizations"""
        all_questions = {}
        orgs_path = Path("ORGANIZATIONS")

        if not orgs_path.exists():
            return all_questions

        for org_dir in orgs_path.iterdir():
            if org_dir.is_dir():
                projects = self.analyze_organization_projects(org_dir)
                org_questions = {}

                for project in projects:
                    questions = self.generate_questions_for_project(project)
                    org_questions[project.name] = {
                        "profile": {
                            "name": project.name,
                            "organization": project.organization,
                            "domain": project.domain,
                            "type": project.type,
                            "focus_areas": project.focus_areas,
                            "stakeholders": project.stakeholders
                        },
                        "questions": questions,
                        "question_count": len(questions)
                    }

                if org_questions:
                    all_questions[org_dir.name] = org_questions

        return all_questions

def main():
    """Main entry point"""
    generator = QuestionGenerator()

    print("[QUESTION-GENERATOR] Generating comprehensive questions for all projects...")

    all_questions = generator.generate_all_questions()

    # Save to file
    output_file = Path(".meta/project-questions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    # Generate summary
    total_projects = sum(len(org_data) for org_data in all_questions.values())
    total_questions = sum(
        project_data["question_count"]
        for org_data in all_questions.values()
        for project_data in org_data.values()
    )

    print(f"\n[SUMMARY] Generated questions for:")
    print(f"  Organizations: {len(all_questions)}")
    print(f"  Projects: {total_projects}")
    print(f"  Total Questions: {total_questions}")
    print(f"  Average per Project: {total_questions/total_projects:.0f}")

    print(f"\n[OUTPUT] Questions saved to: {output_file}")

    # Show breakdown by organization
    print(f"\n[BREAKDOWN] Questions by Organization:")
    for org_name, org_data in all_questions.items():
        org_total = sum(project_data["question_count"] for project_data in org_data.values())
        print(f"  {org_name}: {len(org_data)} projects, {org_total} questions")

if __name__ == "__main__":
    main()
