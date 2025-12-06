"""GrantWriter - AI-Powered Grant Proposal Assistant

Generate competitive grant proposals with AI assistance.
"""

import json
import os
import sys
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Literal
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class BudgetItem:
    """Budget line item"""
    category: str  # personnel, equipment, supplies, travel, other
    description: str
    cost: float
    justification: str
    year: int = 1


@dataclass
class TimelinePhase:
    """Project timeline phase"""
    phase_name: str
    start_month: int
    end_month: int
    deliverables: List[str]
    milestones: List[str]


@dataclass
class Personnel:
    """Project personnel"""
    name: str
    role: str  # PI, Co-I, postdoc, grad student, etc.
    effort_percent: float
    qualifications: str
    responsibilities: str


@dataclass
class GrantProposal:
    """Complete grant proposal"""
    proposal_id: int
    title: str
    agency: str  # NSF, NIH, DOE, etc.
    program: str
    pi_name: str
    institution: str

    # Core proposal sections
    abstract: str
    significance: str
    innovation: str
    approach: str
    broader_impacts: str

    # Project details
    objectives: List[str]
    hypotheses: List[str]
    personnel: List[Personnel]
    timeline: List[TimelinePhase]
    budget: List[BudgetItem]

    # Metadata
    created_date: str
    total_cost: float = 0.0
    duration_years: int = 3

    def __post_init__(self):
        """Calculate total cost"""
        self.total_cost = sum(item.cost for item in self.budget)


class GrantWriter:
    """AI-Powered Grant Proposal Writer"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.proposals_file = self.data_dir / "proposals.json"
        self.templates_file = self.data_dir / "templates.json"

        self.proposals: Dict[int, GrantProposal] = {}
        self.next_proposal_id = 1

        self._load_data()

    def _load_data(self):
        """Load proposals from disk"""
        if self.proposals_file.exists():
            with open(self.proposals_file, 'r') as f:
                data = json.load(f)
                for prop_data in data:
                    # Reconstruct Personnel objects
                    personnel = [Personnel(**p) for p in prop_data['personnel']]
                    prop_data['personnel'] = personnel

                    # Reconstruct TimelinePhase objects
                    timeline = [TimelinePhase(**t) for t in prop_data['timeline']]
                    prop_data['timeline'] = timeline

                    # Reconstruct BudgetItem objects
                    budget = [BudgetItem(**b) for b in prop_data['budget']]
                    prop_data['budget'] = budget

                    proposal = GrantProposal(**prop_data)
                    self.proposals[proposal.proposal_id] = proposal
                    self.next_proposal_id = max(self.next_proposal_id, proposal.proposal_id + 1)

    def _save_data(self):
        """Save proposals to disk"""
        data = []
        for proposal in self.proposals.values():
            prop_dict = asdict(proposal)
            data.append(prop_dict)

        with open(self.proposals_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_proposal(
        self,
        title: str,
        agency: str,
        program: str,
        pi_name: str,
        institution: str,
        research_area: str,
        objectives: List[str],
        duration_years: int = 3
    ) -> GrantProposal:
        """Create a new grant proposal with AI-generated sections"""

        # Generate abstract
        abstract = self._generate_abstract(title, research_area, objectives)

        # Generate significance statement
        significance = self._generate_significance(research_area, objectives)

        # Generate innovation section
        innovation = self._generate_innovation(research_area, title)

        # Generate approach section
        approach = self._generate_approach(objectives, research_area)

        # Generate broader impacts (especially for NSF)
        broader_impacts = self._generate_broader_impacts(research_area, agency)

        # Generate hypotheses from objectives
        hypotheses = self._generate_hypotheses(objectives, research_area)

        # Create initial personnel list
        personnel = self._suggest_personnel(objectives, pi_name)

        # Create timeline
        timeline = self._generate_timeline(objectives, duration_years)

        # Create budget
        budget = self._generate_budget(personnel, objectives, duration_years)

        proposal = GrantProposal(
            proposal_id=self.next_proposal_id,
            title=title,
            agency=agency,
            program=program,
            pi_name=pi_name,
            institution=institution,
            abstract=abstract,
            significance=significance,
            innovation=innovation,
            approach=approach,
            broader_impacts=broader_impacts,
            objectives=objectives,
            hypotheses=hypotheses,
            personnel=personnel,
            timeline=timeline,
            budget=budget,
            created_date=datetime.now().isoformat(),
            duration_years=duration_years
        )

        self.proposals[self.next_proposal_id] = proposal
        self.next_proposal_id += 1
        self._save_data()

        return proposal

    def _generate_abstract(self, title: str, research_area: str, objectives: List[str]) -> str:
        """Generate compelling abstract"""
        abstract = f"This proposal aims to advance the field of {research_area} through {title.lower()}. "

        # Add intellectual merit
        abstract += f"The intellectual merit of this work lies in addressing fundamental questions about {research_area}. "

        # Add objectives
        abstract += "Specific objectives include: "
        abstract += "; ".join(f"({i+1}) {obj.lower()}" for i, obj in enumerate(objectives[:3]))
        abstract += ". "

        # Add approach
        abstract += "We will employ innovative methodologies combining computational and experimental approaches "
        abstract += f"to achieve these objectives. "

        # Add broader impacts
        abstract += "The broader impacts include training graduate students and postdocs, advancing diversity in STEM, "
        abstract += "and disseminating findings through publications, conferences, and open-source software. "

        # Add expected outcomes
        abstract += f"Expected outcomes include significant advances in understanding {research_area}, "
        abstract += "publication of high-impact research articles, and development of novel tools and methodologies "
        abstract += "that will benefit the broader scientific community."

        return abstract

    def _generate_significance(self, research_area: str, objectives: List[str]) -> str:
        """Generate significance statement"""
        significance = "SIGNIFICANCE\n\n"

        # Problem statement
        significance += f"Understanding {research_area} is a central challenge in modern science. "
        significance += "Despite significant progress, critical gaps remain that limit our ability to "
        significance += "make accurate predictions and develop practical applications.\n\n"

        # Why it matters
        significance += "IMPORTANCE AND IMPACT\n\n"
        significance += f"This research is significant because {research_area} underpins numerous "
        significance += "scientific and practical applications. Advances in this area will:\n\n"
        significance += "• Enable new technologies with broad societal impact\n"
        significance += "• Deepen fundamental scientific understanding\n"
        significance += "• Train the next generation of researchers\n"
        significance += "• Foster interdisciplinary collaboration\n\n"

        # Current limitations
        significance += "CURRENT LIMITATIONS\n\n"
        significance += "Current approaches face several limitations:\n\n"
        significance += "• Limited scalability to complex real-world systems\n"
        significance += "• Insufficient integration of computational and experimental methods\n"
        significance += "• Lack of comprehensive frameworks for prediction and validation\n\n"

        # How this work addresses limitations
        significance += "HOW THIS WORK ADDRESSES LIMITATIONS\n\n"
        significance += "This proposal directly addresses these limitations through:\n\n"
        for i, obj in enumerate(objectives, 1):
            significance += f"{i}. {obj}\n"
        significance += "\n"

        # Expected impact
        significance += "EXPECTED IMPACT\n\n"
        significance += "Success in these objectives will:\n\n"
        significance += "• Transform how we understand and predict complex phenomena\n"
        significance += "• Provide validated tools and datasets to the community\n"
        significance += "• Generate high-impact publications in top-tier journals\n"
        significance += "• Establish new paradigms for interdisciplinary research\n"

        return significance

    def _generate_innovation(self, research_area: str, title: str) -> str:
        """Generate innovation section"""
        innovation = "INNOVATION\n\n"

        innovation += "This proposal is innovative in several key aspects:\n\n"

        innovation += "CONCEPTUAL INNOVATION\n\n"
        innovation += f"We introduce a novel conceptual framework for {research_area} that challenges "
        innovation += "conventional assumptions. This framework integrates insights from multiple disciplines "
        innovation += "to provide a more comprehensive understanding than existing approaches.\n\n"

        innovation += "METHODOLOGICAL INNOVATION\n\n"
        innovation += "Our methodological innovations include:\n\n"
        innovation += "• Novel computational methods that achieve unprecedented accuracy and efficiency\n"
        innovation += "• Integration of machine learning with traditional analytical approaches\n"
        innovation += "• Development of new experimental techniques with higher sensitivity\n"
        innovation += "• Creation of open-source software tools for community use\n\n"

        innovation += "TECHNOLOGICAL INNOVATION\n\n"
        innovation += "We will develop new technologies including:\n\n"
        innovation += "• Advanced algorithms optimized for modern computing architectures\n"
        innovation += "• Automated workflows for high-throughput analysis\n"
        innovation += "• Interactive visualization tools for data exploration\n"
        innovation += "• Reproducible research pipelines following best practices\n\n"

        innovation += "TRANSFORMATIVE POTENTIAL\n\n"
        innovation += "The transformative potential of this work lies in its ability to:\n\n"
        innovation += "• Shift paradigms in how we approach fundamental questions\n"
        innovation += "• Enable research directions previously considered infeasible\n"
        innovation += "• Accelerate discovery through computational and experimental synergy\n"
        innovation += "• Establish new standards for rigor and reproducibility\n"

        return innovation

    def _generate_approach(self, objectives: List[str], research_area: str) -> str:
        """Generate research approach"""
        approach = "RESEARCH APPROACH\n\n"

        approach += "OVERVIEW\n\n"
        approach += "Our research approach combines computational modeling, experimental validation, "
        approach += "and theoretical analysis to achieve the proposed objectives. The approach is organized "
        approach += "into integrated research aims, each with specific hypotheses, methods, and expected outcomes.\n\n"

        for i, objective in enumerate(objectives, 1):
            approach += f"AIM {i}: {objective.upper()}\n\n"

            approach += f"Hypothesis {i}: We hypothesize that {objective.lower()} will reveal "
            approach += f"fundamental principles governing {research_area}.\n\n"

            approach += "Methods:\n\n"
            approach += "• Computational modeling using state-of-the-art algorithms\n"
            approach += "• Experimental validation with high-resolution techniques\n"
            approach += "• Statistical analysis and hypothesis testing\n"
            approach += "• Integration of multi-scale data\n\n"

            approach += "Expected Outcomes:\n\n"
            approach += "• Validated computational models\n"
            approach += "• High-quality experimental datasets\n"
            approach += "• Publications in top-tier journals\n"
            approach += "• Open-source software tools\n\n"

            approach += "Potential Challenges and Mitigation:\n\n"
            approach += "• Challenge: Computational complexity\n"
            approach += "  Mitigation: Develop efficient algorithms and leverage HPC resources\n"
            approach += "• Challenge: Experimental variability\n"
            approach += "  Mitigation: Use rigorous controls and statistical power analysis\n\n"

        approach += "INTEGRATION AND SYNERGY\n\n"
        approach += "The research aims are highly integrated, with insights from one aim informing the others. "
        approach += "This synergy will accelerate progress and enable emergent discoveries.\n\n"

        approach += "RIGOR AND REPRODUCIBILITY\n\n"
        approach += "We will ensure rigor through:\n\n"
        approach += "• Pre-registration of hypotheses and analysis plans\n"
        approach += "• Blinded experimental designs where appropriate\n"
        approach += "• Independent replication of key findings\n"
        approach += "• Comprehensive documentation and code/data sharing\n"

        return approach

    def _generate_broader_impacts(self, research_area: str, agency: str) -> str:
        """Generate broader impacts section"""
        impacts = "BROADER IMPACTS\n\n"

        impacts += "This project will generate significant broader impacts across multiple dimensions:\n\n"

        impacts += "EDUCATION AND TRAINING\n\n"
        impacts += "• Train graduate students and postdocs in cutting-edge research methods\n"
        impacts += "• Provide undergraduate research opportunities, especially for underrepresented groups\n"
        impacts += "• Develop course materials integrating research findings\n"
        impacts += "• Mentor students on career development and professional skills\n\n"

        impacts += "DIVERSITY AND INCLUSION\n\n"
        impacts += "• Actively recruit researchers from underrepresented groups\n"
        impacts += "• Partner with minority-serving institutions\n"
        impacts += "• Host workshops to broaden participation\n"
        impacts += "• Create inclusive lab culture with explicit anti-bias training\n\n"

        impacts += "DISSEMINATION AND OUTREACH\n\n"
        impacts += "• Publish findings in open-access journals\n"
        impacts += "• Release all software as open-source with comprehensive documentation\n"
        impacts += "• Share datasets through public repositories\n"
        impacts += "• Present at conferences and webinars\n"
        impacts += "• Engage with K-12 educators through science outreach programs\n\n"

        impacts += "SOCIETAL IMPACT\n\n"
        impacts += f"• Advance {research_area} with applications to pressing societal challenges\n"
        impacts += "• Foster collaboration between academia and industry\n"
        impacts += "• Inform evidence-based policy decisions\n"
        impacts += "• Contribute to economic competitiveness through innovation\n\n"

        impacts += "INFRASTRUCTURE DEVELOPMENT\n\n"
        impacts += "• Develop reusable software tools for the research community\n"
        impacts += "• Establish data standards and best practices\n"
        impacts += "• Build collaborative networks across institutions\n"
        impacts += "• Create educational resources available to all researchers\n"

        if agency.upper() == "NSF":
            impacts += "\nThis project aligns with NSF's core values of promoting discovery, "
            impacts += "advancing education, and fostering innovation."

        return impacts

    def _generate_hypotheses(self, objectives: List[str], research_area: str) -> List[str]:
        """Generate testable hypotheses from objectives"""
        hypotheses = []

        hypothesis_templates = [
            f"We hypothesize that {{obj}} will reveal fundamental mechanisms in {research_area}",
            "Our central hypothesis is that {obj} through novel computational approaches",
            "We hypothesize that {obj} will demonstrate previously unknown relationships",
            "Testing the hypothesis that {obj} will validate our theoretical framework"
        ]

        for i, obj in enumerate(objectives):
            template = hypothesis_templates[i % len(hypothesis_templates)]
            hypothesis = template.format(obj=obj.lower())
            hypotheses.append(hypothesis)

        return hypotheses

    def _suggest_personnel(self, objectives: List[str], pi_name: str) -> List[Personnel]:
        """Suggest project personnel based on objectives"""
        personnel = []

        # Principal Investigator
        pi = Personnel(
            name=pi_name,
            role="Principal Investigator",
            effort_percent=20.0,
            qualifications="PhD with expertise in the research area, extensive publication record, proven track record securing external funding",
            responsibilities="Overall project leadership, supervision of research team, manuscript preparation, grant management"
        )
        personnel.append(pi)

        # Co-Investigator (if multiple complex objectives)
        if len(objectives) >= 3:
            co_i = Personnel(
                name="Co-Investigator (TBD)",
                role="Co-Investigator",
                effort_percent=10.0,
                qualifications="Complementary expertise in computational methods or experimental techniques",
                responsibilities="Lead specific research aims, provide methodological expertise, mentor trainees"
            )
            personnel.append(co_i)

        # Postdoctoral researcher
        postdoc = Personnel(
            name="Postdoctoral Researcher (TBD)",
            role="Postdoctoral Researcher",
            effort_percent=100.0,
            qualifications="Recent PhD in relevant field with strong computational and/or experimental skills",
            responsibilities="Day-to-day research execution, data analysis, manuscript preparation, mentoring graduate students"
        )
        personnel.append(postdoc)

        # Graduate students (1-2 depending on objectives)
        num_grads = min(len(objectives), 2)
        for i in range(num_grads):
            grad = Personnel(
                name=f"Graduate Student {i+1} (TBD)",
                role="Graduate Student",
                effort_percent=50.0,
                qualifications="Strong academic background, research experience, commitment to research career",
                responsibilities=f"Research on Aim {i+1}, data collection and analysis, assist with publications"
            )
            personnel.append(grad)

        return personnel

    def _generate_timeline(self, objectives: List[str], duration_years: int) -> List[TimelinePhase]:
        """Generate project timeline"""
        timeline = []
        months_total = duration_years * 12
        months_per_objective = months_total // len(objectives)

        # Year 1: Setup and initial objectives
        phase1 = TimelinePhase(
            phase_name="Year 1: Project Setup and Initial Research",
            start_month=1,
            end_month=12,
            deliverables=[
                "IRB/IACUC approvals obtained",
                "Lab space and equipment setup complete",
                "Personnel hired and trained",
                f"{objectives[0]} initiated",
                "Preliminary data collected"
            ],
            milestones=[
                "Month 3: Complete hiring and training",
                "Month 6: Complete preliminary experiments",
                "Month 9: First results obtained",
                "Month 12: Submit first manuscript"
            ]
        )
        timeline.append(phase1)

        # Middle years: Core research
        for year in range(2, duration_years):
            obj_idx = min(year - 1, len(objectives) - 1)
            phase = TimelinePhase(
                phase_name=f"Year {year}: Core Research Activities",
                start_month=(year - 1) * 12 + 1,
                end_month=year * 12,
                deliverables=[
                    f"{objectives[obj_idx]} completed",
                    "High-quality datasets generated",
                    "Publications submitted",
                    "Conference presentations delivered",
                    "Software tools released"
                ],
                milestones=[
                    f"Month {(year-1)*12 + 3}: Complete major experiments",
                    f"Month {(year-1)*12 + 6}: Data analysis complete",
                    f"Month {(year-1)*12 + 9}: Manuscript submission",
                    f"Month {year*12}: Annual progress report"
                ]
            )
            timeline.append(phase)

        # Final year: Completion and dissemination
        phase_final = TimelinePhase(
            phase_name=f"Year {duration_years}: Completion and Dissemination",
            start_month=(duration_years - 1) * 12 + 1,
            end_month=duration_years * 12,
            deliverables=[
                "All research objectives completed",
                "Final manuscripts submitted/published",
                "Software documented and released",
                "Datasets deposited in public repositories",
                "Final project report completed"
            ],
            milestones=[
                f"Month {(duration_years-1)*12 + 3}: Final experiments complete",
                f"Month {(duration_years-1)*12 + 6}: All analysis finished",
                f"Month {(duration_years-1)*12 + 9}: Final manuscripts submitted",
                f"Month {duration_years*12}: Project closeout"
            ]
        )
        timeline.append(phase_final)

        return timeline

    def _generate_budget(self, personnel: List[Personnel], objectives: List[str], duration_years: int) -> List[BudgetItem]:
        """Generate detailed budget"""
        budget = []

        # Personnel costs
        for person in personnel:
            if "Principal Investigator" in person.role or "Co-Investigator" in person.role:
                # Faculty salary
                annual_salary = 120000 * (person.effort_percent / 100.0)
                for year in range(1, duration_years + 1):
                    # 3% annual increase
                    year_salary = annual_salary * (1.03 ** (year - 1))
                    item = BudgetItem(
                        category="personnel",
                        description=f"{person.role} salary ({person.effort_percent}% effort)",
                        cost=year_salary,
                        justification=f"{person.role} will {person.responsibilities}",
                        year=year
                    )
                    budget.append(item)
            elif "Postdoc" in person.role:
                annual_salary = 65000
                for year in range(1, duration_years + 1):
                    year_salary = annual_salary * (1.03 ** (year - 1))
                    item = BudgetItem(
                        category="personnel",
                        description="Postdoctoral Researcher salary",
                        cost=year_salary,
                        justification="Postdoc will conduct day-to-day research and mentor students",
                        year=year
                    )
                    budget.append(item)
            elif "Graduate Student" in person.role:
                annual_stipend = 35000
                for year in range(1, duration_years + 1):
                    year_stipend = annual_stipend * (1.03 ** (year - 1))
                    item = BudgetItem(
                        category="personnel",
                        description=f"{person.name} stipend",
                        cost=year_stipend,
                        justification=f"Graduate student will work on {person.responsibilities}",
                        year=year
                    )
                    budget.append(item)

        # Fringe benefits (30% of salaries)
        for year in range(1, duration_years + 1):
            year_salaries = sum(item.cost for item in budget if item.year == year and item.category == "personnel")
            fringe = BudgetItem(
                category="personnel",
                description="Fringe benefits (30%)",
                cost=year_salaries * 0.30,
                justification="Standard university fringe benefit rate",
                year=year
            )
            budget.append(fringe)

        # Equipment (Year 1 only)
        equipment = [
            BudgetItem(
                category="equipment",
                description="High-performance computing cluster",
                cost=50000,
                justification="Required for computational modeling and large-scale data analysis",
                year=1
            ),
            BudgetItem(
                category="equipment",
                description="Laboratory equipment",
                cost=75000,
                justification="Specialized instrumentation for experimental validation",
                year=1
            )
        ]
        budget.extend(equipment)

        # Supplies (each year)
        for year in range(1, duration_years + 1):
            supplies = BudgetItem(
                category="supplies",
                description="Laboratory supplies and reagents",
                cost=15000,
                justification="Consumables for experimental work",
                year=year
            )
            budget.append(supplies)

        # Travel (each year)
        for year in range(1, duration_years + 1):
            travel = BudgetItem(
                category="travel",
                description="Conference travel and collaboration",
                cost=8000,
                justification="Present findings at 2 major conferences per year, visit collaborators",
                year=year
            )
            budget.append(travel)

        # Publication costs
        for year in range(1, duration_years + 1):
            pub_cost = BudgetItem(
                category="other",
                description="Publication and open-access fees",
                cost=5000,
                justification="Open-access publication fees for 2-3 papers per year",
                year=year
            )
            budget.append(pub_cost)

        # Indirect costs (50% of total direct costs, per year)
        for year in range(1, duration_years + 1):
            direct_costs = sum(item.cost for item in budget if item.year == year)
            indirect = BudgetItem(
                category="other",
                description="Indirect costs (F&A)",
                cost=direct_costs * 0.50,
                justification="University indirect cost rate of 50%",
                year=year
            )
            budget.append(indirect)

        return budget

    def get_proposal(self, proposal_id: int) -> Optional[GrantProposal]:
        """Get a specific proposal"""
        return self.proposals.get(proposal_id)

    def list_proposals(self) -> List[GrantProposal]:
        """List all proposals"""
        return list(self.proposals.values())

    def export_proposal(self, proposal_id: int, format: str = "text") -> str:
        """Export proposal to text format"""
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            return ""

        output = "=" * 80 + "\n"
        output += f"{proposal.title}\n"
        output += "=" * 80 + "\n\n"

        output += f"Agency: {proposal.agency}\n"
        output += f"Program: {proposal.program}\n"
        output += f"PI: {proposal.pi_name}\n"
        output += f"Institution: {proposal.institution}\n"
        output += f"Duration: {proposal.duration_years} years\n"
        output += f"Total Budget: ${proposal.total_cost:,.2f}\n"
        output += f"Created: {proposal.created_date}\n\n"

        output += "=" * 80 + "\n"
        output += "ABSTRACT\n"
        output += "=" * 80 + "\n\n"
        output += proposal.abstract + "\n\n"

        output += "=" * 80 + "\n"
        output += "OBJECTIVES\n"
        output += "=" * 80 + "\n\n"
        for i, obj in enumerate(proposal.objectives, 1):
            output += f"{i}. {obj}\n"
        output += "\n"

        output += "=" * 80 + "\n"
        output += proposal.significance + "\n\n"

        output += "=" * 80 + "\n"
        output += proposal.innovation + "\n\n"

        output += "=" * 80 + "\n"
        output += proposal.approach + "\n\n"

        output += "=" * 80 + "\n"
        output += proposal.broader_impacts + "\n\n"

        output += "=" * 80 + "\n"
        output += "PERSONNEL\n"
        output += "=" * 80 + "\n\n"
        for person in proposal.personnel:
            output += f"{person.role}: {person.name}\n"
            output += f"  Effort: {person.effort_percent}%\n"
            output += f"  Qualifications: {person.qualifications}\n"
            output += f"  Responsibilities: {person.responsibilities}\n\n"

        output += "=" * 80 + "\n"
        output += "TIMELINE\n"
        output += "=" * 80 + "\n\n"
        for phase in proposal.timeline:
            output += f"{phase.phase_name}\n"
            output += f"  Months {phase.start_month}-{phase.end_month}\n"
            output += "  Deliverables:\n"
            for deliverable in phase.deliverables:
                output += f"    • {deliverable}\n"
            output += "  Milestones:\n"
            for milestone in phase.milestones:
                output += f"    • {milestone}\n"
            output += "\n"

        output += "=" * 80 + "\n"
        output += "BUDGET SUMMARY\n"
        output += "=" * 80 + "\n\n"

        # Budget by year
        for year in range(1, proposal.duration_years + 1):
            year_items = [item for item in proposal.budget if item.year == year]
            year_total = sum(item.cost for item in year_items)
            output += f"Year {year}: ${year_total:,.2f}\n"

            # By category
            categories = {}
            for item in year_items:
                if item.category not in categories:
                    categories[item.category] = 0
                categories[item.category] += item.cost

            for category, amount in categories.items():
                output += f"  {category.capitalize()}: ${amount:,.2f}\n"
            output += "\n"

        output += f"TOTAL PROJECT COST: ${proposal.total_cost:,.2f}\n"

        return output

    def generate_budget_justification(self, proposal_id: int) -> str:
        """Generate detailed budget justification"""
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            return ""

        justification = "BUDGET JUSTIFICATION\n\n"

        # Group by category
        by_category = {}
        for item in proposal.budget:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)

        for category, items in by_category.items():
            justification += f"{category.upper()}\n"
            justification += "-" * 70 + "\n\n"

            # Group by description
            by_desc = {}
            for item in items:
                if item.description not in by_desc:
                    by_desc[item.description] = []
                by_desc[item.description].append(item)

            for desc, desc_items in by_desc.items():
                total = sum(item.cost for item in desc_items)
                justification += f"{desc}: ${total:,.2f}\n"
                justification += f"  {desc_items[0].justification}\n"

                # Year breakdown if multi-year
                if len(desc_items) > 1:
                    justification += "  Year breakdown:\n"
                    for item in desc_items:
                        justification += f"    Year {item.year}: ${item.cost:,.2f}\n"
                justification += "\n"

            justification += "\n"

        return justification


def main():
    parser = argparse.ArgumentParser(
        description="GrantWriter - AI-Powered Grant Proposal Assistant"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create proposal
    create_parser = subparsers.add_parser('create', help='Create new grant proposal')
    create_parser.add_argument('--title', required=True, help='Proposal title')
    create_parser.add_argument('--agency', required=True, help='Funding agency (NSF, NIH, etc.)')
    create_parser.add_argument('--program', required=True, help='Program name')
    create_parser.add_argument('--pi', required=True, help='PI name')
    create_parser.add_argument('--institution', required=True, help='Institution name')
    create_parser.add_argument('--research-area', required=True, help='Research area')
    create_parser.add_argument('--objectives', required=True, nargs='+', help='Research objectives')
    create_parser.add_argument('--duration', type=int, default=3, help='Duration in years')

    # List proposals
    list_parser = subparsers.add_parser('list', help='List all proposals')

    # Export proposal
    export_parser = subparsers.add_parser('export', help='Export proposal')
    export_parser.add_argument('--proposal-id', type=int, required=True)
    export_parser.add_argument('--output', help='Output file (default: stdout)')

    # Budget justification
    budget_parser = subparsers.add_parser('budget', help='Generate budget justification')
    budget_parser.add_argument('--proposal-id', type=int, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    writer = GrantWriter()

    if args.command == 'create':
        proposal = writer.create_proposal(
            title=args.title,
            agency=args.agency,
            program=args.program,
            pi_name=args.pi,
            institution=args.institution,
            research_area=args.research_area,
            objectives=args.objectives,
            duration_years=args.duration
        )

        print(f"\nProposal created successfully!")
        print(f"Proposal ID: {proposal.proposal_id}")
        print(f"Title: {proposal.title}")
        print(f"Total Budget: ${proposal.total_cost:,.2f}")
        print(f"Duration: {proposal.duration_years} years")
        print(f"\nUse 'export --proposal-id {proposal.proposal_id}' to view full proposal")

    elif args.command == 'list':
        proposals = writer.list_proposals()

        print(f"\n{'='*70}")
        print(f"GRANT PROPOSALS ({len(proposals)} total)")
        print(f"{'='*70}\n")

        for prop in proposals:
            print(f"ID {prop.proposal_id}: {prop.title}")
            print(f"  Agency: {prop.agency} ({prop.program})")
            print(f"  PI: {prop.pi_name}")
            print(f"  Budget: ${prop.total_cost:,.2f} over {prop.duration_years} years")
            print(f"  Created: {prop.created_date}")
            print()

    elif args.command == 'export':
        output = writer.export_proposal(args.proposal_id)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Proposal exported to {args.output}")
        else:
            print(output)

    elif args.command == 'budget':
        justification = writer.generate_budget_justification(args.proposal_id)
        print(justification)


if __name__ == "__main__":
    main()
