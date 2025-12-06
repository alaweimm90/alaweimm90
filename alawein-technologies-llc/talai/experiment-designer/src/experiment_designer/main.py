#!/usr/bin/env python3
"""
ExperimentDesigner - Automated Protocol Generator

Input hypothesis → Complete experiment protocol.
Includes power analysis, controls, equipment, timeline, cost estimation.
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta

@dataclass
class Hypothesis:
    """Research hypothesis to test"""
    hypothesis_id: int
    statement: str
    domain: str
    variables: Dict[str, str]  # independent/dependent/control
    expected_effect_size: str  # "small" | "medium" | "large"
    mechanism: str
    prior_evidence: str
    created_at: str

@dataclass
class PowerAnalysis:
    """Statistical power analysis for experiment"""
    effect_size: float
    alpha: float  # Significance level
    power: float  # 1 - beta
    required_sample_size: int
    groups: int
    observations_per_group: int
    statistical_test: str
    assumptions: List[str]
    sensitivity_analysis: Dict[str, int]  # effect_size -> sample_size

@dataclass
class ControlVariable:
    """Control or confound to manage"""
    name: str
    type: str  # "randomize" | "block" | "measure" | "hold_constant"
    justification: str
    method: str

@dataclass
class Equipment:
    """Required equipment/materials"""
    name: str
    quantity: int
    unit: str
    estimated_cost: float
    vendor: Optional[str] = None
    specifications: Optional[str] = None

@dataclass
class Step:
    """Experimental procedure step"""
    step_number: int
    action: str
    duration: str
    equipment_needed: List[str]
    safety_notes: List[str]
    quality_checks: List[str]
    expected_output: str

@dataclass
class Timeline:
    """Project timeline"""
    phase: str
    description: str
    duration_days: int
    dependencies: List[str]
    milestones: List[str]

@dataclass
class ExperimentProtocol:
    """Complete experiment protocol"""
    protocol_id: int
    hypothesis_id: int

    # Core design
    design_type: str  # "randomized_controlled" | "factorial" | "crossover" | "observational"
    blind_level: str  # "none" | "single" | "double" | "triple"
    randomization_method: str

    # Power and sample
    power_analysis: PowerAnalysis

    # Variables
    independent_vars: List[str]
    dependent_vars: List[str]
    control_vars: List[ControlVariable]

    # Procedure
    steps: List[Step]

    # Resources
    equipment_list: List[Equipment]
    personnel_needed: Dict[str, int]  # role -> count

    # Timeline
    timeline: List[Timeline]
    total_duration_days: int

    # Budget
    total_cost: float
    cost_breakdown: Dict[str, float]

    # Analysis plan
    primary_outcome: str
    secondary_outcomes: List[str]
    statistical_methods: List[str]
    data_collection_plan: str

    # Quality
    quality_assurance: List[str]
    risks: List[Dict[str, str]]  # risk, mitigation
    ethics_considerations: List[str]

    # Metadata
    confidence_score: float  # How confident in this protocol (0-1)
    limitations: List[str]
    alternatives_considered: List[str]
    created_at: str

class ExperimentDesigner:
    """Generate experiment protocols from hypotheses"""

    def __init__(self, data_file: str = "designer.json"):
        self.data_file = Path(data_file)
        self.hypotheses: Dict[int, Hypothesis] = {}
        self.protocols: Dict[int, ExperimentProtocol] = {}
        self._load_data()

    def _load_data(self):
        """Load data from JSON"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.hypotheses = {
                    int(k): Hypothesis(**v)
                    for k, v in data.get('hypotheses', {}).items()
                }
                # Protocols need nested object reconstruction
                for k, v in data.get('protocols', {}).items():
                    v['power_analysis'] = PowerAnalysis(**v['power_analysis'])
                    v['control_vars'] = [ControlVariable(**cv) for cv in v['control_vars']]
                    v['steps'] = [Step(**s) for s in v['steps']]
                    v['equipment_list'] = [Equipment(**e) for e in v['equipment_list']]
                    v['timeline'] = [Timeline(**t) for t in v['timeline']]
                    self.protocols[int(k)] = ExperimentProtocol(**v)

    def _save_data(self):
        """Save data to JSON"""
        data = {
            'hypotheses': {k: asdict(v) for k, v in self.hypotheses.items()},
            'protocols': {k: asdict(v) for k, v in self.protocols.items()}
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def submit_hypothesis(
        self,
        statement: str,
        domain: str,
        independent_var: str,
        dependent_var: str,
        control_vars: List[str],
        expected_effect_size: str,
        mechanism: str,
        prior_evidence: str
    ) -> Hypothesis:
        """Submit a hypothesis for protocol design"""
        hypothesis_id = len(self.hypotheses) + 1

        variables = {
            'independent': independent_var,
            'dependent': dependent_var,
            'control': ', '.join(control_vars)
        }

        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=statement,
            domain=domain,
            variables=variables,
            expected_effect_size=expected_effect_size,
            mechanism=mechanism,
            prior_evidence=prior_evidence,
            created_at=datetime.now().isoformat()
        )

        self.hypotheses[hypothesis_id] = hypothesis
        self._save_data()

        return hypothesis

    def design_protocol(self, hypothesis_id: int) -> ExperimentProtocol:
        """Generate complete experiment protocol"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        hypothesis = self.hypotheses[hypothesis_id]
        protocol_id = len(self.protocols) + 1

        # Determine design type based on domain
        design_type = self._determine_design_type(hypothesis)
        blind_level = self._determine_blinding(hypothesis)
        randomization = self._determine_randomization(design_type)

        # Power analysis
        power_analysis = self._calculate_power(hypothesis)

        # Variables
        ind_vars = [v.strip() for v in hypothesis.variables['independent'].split(',')]
        dep_vars = [v.strip() for v in hypothesis.variables['dependent'].split(',')]
        control_vars = self._identify_controls(hypothesis)

        # Generate procedure
        steps = self._generate_steps(hypothesis, design_type)

        # Equipment and materials
        equipment = self._generate_equipment_list(hypothesis, power_analysis)

        # Personnel
        personnel = self._estimate_personnel(hypothesis, power_analysis)

        # Timeline
        timeline = self._generate_timeline(hypothesis, power_analysis, steps)
        total_duration = sum(t.duration_days for t in timeline)

        # Budget
        equipment_cost = sum(e.estimated_cost for e in equipment)
        personnel_cost = self._estimate_personnel_cost(personnel, total_duration)
        overhead = (equipment_cost + personnel_cost) * 0.3
        total_cost = equipment_cost + personnel_cost + overhead

        cost_breakdown = {
            'equipment': equipment_cost,
            'personnel': personnel_cost,
            'overhead': overhead
        }

        # Analysis plan
        primary_outcome, secondary_outcomes = self._plan_outcomes(hypothesis)
        statistical_methods = self._select_statistics(hypothesis, design_type)
        data_plan = self._plan_data_collection(hypothesis, power_analysis)

        # Quality assurance
        qa_measures = self._generate_qa_measures(hypothesis, design_type)
        risks = self._identify_risks(hypothesis)
        ethics = self._identify_ethics_issues(hypothesis)

        # Confidence and limitations
        confidence = self._estimate_confidence(hypothesis, power_analysis)
        limitations = self._identify_limitations(hypothesis, design_type)
        alternatives = self._generate_alternatives(hypothesis)

        protocol = ExperimentProtocol(
            protocol_id=protocol_id,
            hypothesis_id=hypothesis_id,
            design_type=design_type,
            blind_level=blind_level,
            randomization_method=randomization,
            power_analysis=power_analysis,
            independent_vars=ind_vars,
            dependent_vars=dep_vars,
            control_vars=control_vars,
            steps=steps,
            equipment_list=equipment,
            personnel_needed=personnel,
            timeline=timeline,
            total_duration_days=total_duration,
            total_cost=total_cost,
            cost_breakdown=cost_breakdown,
            primary_outcome=primary_outcome,
            secondary_outcomes=secondary_outcomes,
            statistical_methods=statistical_methods,
            data_collection_plan=data_plan,
            quality_assurance=qa_measures,
            risks=risks,
            ethics_considerations=ethics,
            confidence_score=confidence,
            limitations=limitations,
            alternatives_considered=alternatives,
            created_at=datetime.now().isoformat()
        )

        self.protocols[protocol_id] = protocol
        self._save_data()

        return protocol

    def _determine_design_type(self, hypothesis: Hypothesis) -> str:
        """Determine appropriate experimental design"""
        domain_designs = {
            'medicine': 'randomized_controlled',
            'psychology': 'factorial',
            'biology': 'randomized_controlled',
            'physics': 'factorial',
            'social_science': 'observational'
        }
        return domain_designs.get(hypothesis.domain, 'randomized_controlled')

    def _determine_blinding(self, hypothesis: Hypothesis) -> str:
        """Determine blinding level"""
        if hypothesis.domain in ['medicine', 'psychology']:
            return 'double'
        elif hypothesis.domain in ['biology', 'social_science']:
            return 'single'
        else:
            return 'none'

    def _determine_randomization(self, design_type: str) -> str:
        """Determine randomization method"""
        methods = {
            'randomized_controlled': 'Block randomization with stratification',
            'factorial': 'Complete randomization across all factors',
            'crossover': 'Latin square design',
            'observational': 'Propensity score matching'
        }
        return methods.get(design_type, 'Simple randomization')

    def _calculate_power(self, hypothesis: Hypothesis) -> PowerAnalysis:
        """Calculate statistical power and sample size"""
        effect_sizes = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
        effect_size = effect_sizes.get(hypothesis.expected_effect_size, 0.5)

        alpha = 0.05
        power = 0.80

        # Simplified sample size calculation (Cohen's d)
        # n ≈ (Z_α/2 + Z_β)² * 2σ² / d²
        z_alpha = 1.96  # two-tailed
        z_beta = 0.84   # power = 0.80

        n_per_group = int(((z_alpha + z_beta) ** 2) * 2 / (effect_size ** 2)) + 5
        groups = 2  # Assume control vs treatment
        total_n = n_per_group * groups

        # Statistical test
        tests = {
            'medicine': 't-test or ANOVA',
            'psychology': 'Mixed-effects model',
            'biology': 't-test',
            'physics': 'Linear regression',
            'social_science': 'Chi-square or regression'
        }
        stat_test = tests.get(hypothesis.domain, 't-test')

        assumptions = [
            'Normal distribution of residuals',
            'Homogeneity of variance',
            'Independence of observations',
            f'Minimum effect size: {hypothesis.expected_effect_size}'
        ]

        # Sensitivity analysis
        sensitivity = {
            'small (d=0.2)': int(n_per_group * 4),
            'medium (d=0.5)': n_per_group,
            'large (d=0.8)': int(n_per_group * 0.4)
        }

        return PowerAnalysis(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            required_sample_size=total_n,
            groups=groups,
            observations_per_group=n_per_group,
            statistical_test=stat_test,
            assumptions=assumptions,
            sensitivity_analysis=sensitivity
        )

    def _identify_controls(self, hypothesis: Hypothesis) -> List[ControlVariable]:
        """Identify control variables and confounds"""
        control_strs = hypothesis.variables['control'].split(',')
        controls = []

        for i, ctrl in enumerate(control_strs, 1):
            ctrl = ctrl.strip()
            if not ctrl:
                continue

            # Vary control methods
            methods = ['randomize', 'block', 'measure', 'hold_constant']
            method = random.choice(methods)

            justifications = {
                'randomize': f'Randomization eliminates systematic bias from {ctrl}',
                'block': f'Blocking on {ctrl} reduces within-group variance',
                'measure': f'Measuring {ctrl} allows statistical adjustment',
                'hold_constant': f'Holding {ctrl} constant isolates effect of interest'
            }

            controls.append(ControlVariable(
                name=ctrl,
                type=method,
                justification=justifications[method],
                method=f'{method.replace("_", " ").title()} {ctrl}'
            ))

        return controls

    def _generate_steps(self, hypothesis: Hypothesis, design_type: str) -> List[Step]:
        """Generate experimental procedure steps"""
        steps = []

        # Recruitment
        steps.append(Step(
            step_number=1,
            action='Recruit participants/subjects according to inclusion criteria',
            duration='2-4 weeks',
            equipment_needed=['Recruitment materials', 'Screening forms'],
            safety_notes=['Ensure informed consent', 'Verify eligibility'],
            quality_checks=['Document inclusion/exclusion decisions'],
            expected_output='Enrolled cohort meeting sample size requirements'
        ))

        # Baseline
        steps.append(Step(
            step_number=2,
            action=f'Collect baseline measurements of {hypothesis.variables["dependent"]}',
            duration='1-2 weeks',
            equipment_needed=['Measurement instruments', 'Data collection forms'],
            safety_notes=['Follow standard safety protocols'],
            quality_checks=['Calibrate instruments', 'Train data collectors'],
            expected_output='Baseline data for all subjects'
        ))

        # Randomization
        if design_type in ['randomized_controlled', 'factorial']:
            steps.append(Step(
                step_number=3,
                action='Randomize subjects to treatment groups',
                duration='1 day',
                equipment_needed=['Randomization software', 'Sealed envelopes'],
                safety_notes=['Maintain blinding if applicable'],
                quality_checks=['Verify balanced allocation', 'Document randomization'],
                expected_output='Subject assignment to groups'
            ))

        # Intervention
        steps.append(Step(
            step_number=len(steps) + 1,
            action=f'Administer intervention: manipulate {hypothesis.variables["independent"]}',
            duration='Variable (depends on protocol)',
            equipment_needed=['Treatment materials', 'Monitoring equipment'],
            safety_notes=['Monitor for adverse events', 'Follow dosing protocols'],
            quality_checks=['Verify treatment delivery', 'Document compliance'],
            expected_output='Completed intervention phase'
        ))

        # Follow-up
        steps.append(Step(
            step_number=len(steps) + 1,
            action=f'Measure outcomes: {hypothesis.variables["dependent"]}',
            duration='1-4 weeks',
            equipment_needed=['Outcome measurement tools'],
            safety_notes=['Maintain blinding during assessment'],
            quality_checks=['Inter-rater reliability checks', 'Missing data protocols'],
            expected_output='Complete outcome data'
        ))

        # Analysis
        steps.append(Step(
            step_number=len(steps) + 1,
            action='Data analysis and statistical testing',
            duration='2-3 weeks',
            equipment_needed=['Statistical software', 'Computers'],
            safety_notes=['Data security and privacy'],
            quality_checks=['Code review', 'Replication of analyses'],
            expected_output='Statistical results and figures'
        ))

        return steps

    def _generate_equipment_list(self, hypothesis: Hypothesis, power: PowerAnalysis) -> List[Equipment]:
        """Generate equipment and materials list"""
        equipment = []

        domain_equipment = {
            'medicine': [
                Equipment('Medical imaging system', 1, 'unit', 150000, 'GE Healthcare'),
                Equipment('Blood sample collection kit', power.required_sample_size, 'kit', 25 * power.required_sample_size),
                Equipment('Laboratory analyzer', 1, 'unit', 75000, 'Thermo Fisher')
            ],
            'psychology': [
                Equipment('Computer workstations', 10, 'units', 15000, 'Dell'),
                Equipment('Questionnaire packets', power.required_sample_size, 'packets', 5 * power.required_sample_size),
                Equipment('Eye-tracking system', 1, 'unit', 25000, 'Tobii')
            ],
            'biology': [
                Equipment('Microscope (high-resolution)', 1, 'unit', 45000, 'Zeiss'),
                Equipment('Cell culture incubator', 2, 'units', 12000, 'Thermo Fisher'),
                Equipment('Reagents and consumables', 1, 'bulk', 15000)
            ],
            'physics': [
                Equipment('Oscilloscope', 1, 'unit', 8000, 'Tektronix'),
                Equipment('Laser system', 1, 'unit', 35000, 'Coherent'),
                Equipment('Data acquisition system', 1, 'unit', 12000, 'National Instruments')
            ]
        }

        base_equipment = domain_equipment.get(hypothesis.domain, [
            Equipment('Generic measurement device', 1, 'unit', 10000),
            Equipment('Data collection tools', power.required_sample_size, 'units', 50 * power.required_sample_size)
        ])

        # Add common items
        common = [
            Equipment('Computer for data analysis', 1, 'unit', 2000),
            Equipment('Statistical software license', 1, 'license', 500),
            Equipment('Office supplies', 1, 'bulk', 200)
        ]

        return base_equipment + common

    def _estimate_personnel(self, hypothesis: Hypothesis, power: PowerAnalysis) -> Dict[str, int]:
        """Estimate personnel needs"""
        base = {
            'Principal Investigator': 1,
            'Research Coordinator': 1,
            'Data Analysts': 1
        }

        # Scale with sample size
        if power.required_sample_size > 100:
            base['Research Assistants'] = 3
        elif power.required_sample_size > 50:
            base['Research Assistants'] = 2
        else:
            base['Research Assistants'] = 1

        # Domain-specific
        if hypothesis.domain == 'medicine':
            base['Nurses/Clinicians'] = 2
        elif hypothesis.domain == 'biology':
            base['Lab Technicians'] = 2

        return base

    def _estimate_personnel_cost(self, personnel: Dict[str, int], duration_days: int) -> float:
        """Estimate personnel costs"""
        salaries = {
            'Principal Investigator': 150000,
            'Research Coordinator': 65000,
            'Research Assistants': 45000,
            'Data Analysts': 75000,
            'Nurses/Clinicians': 70000,
            'Lab Technicians': 50000
        }

        total = 0
        for role, count in personnel.items():
            annual_salary = salaries.get(role, 50000)
            # Prorate by duration
            cost = (annual_salary / 365) * duration_days * count
            total += cost

        return total

    def _generate_timeline(self, hypothesis: Hypothesis, power: PowerAnalysis, steps: List[Step]) -> List[Timeline]:
        """Generate project timeline"""
        timeline = [
            Timeline(
                phase='Planning & Setup',
                description='Protocol development, IRB approval, equipment procurement',
                duration_days=60,
                dependencies=[],
                milestones=['IRB approval', 'Equipment delivered', 'Staff trained']
            ),
            Timeline(
                phase='Recruitment',
                description=f'Recruit {power.required_sample_size} subjects',
                duration_days=max(30, power.required_sample_size // 2),
                dependencies=['Planning & Setup'],
                milestones=[f'{power.required_sample_size//2} subjects enrolled', 'Full enrollment']
            ),
            Timeline(
                phase='Data Collection',
                description='Execute experimental protocol and collect data',
                duration_days=max(60, power.required_sample_size),
                dependencies=['Recruitment'],
                milestones=['50% complete', 'Data collection complete']
            ),
            Timeline(
                phase='Analysis',
                description='Data cleaning, analysis, and interpretation',
                duration_days=45,
                dependencies=['Data Collection'],
                milestones=['Preliminary results', 'Final results']
            ),
            Timeline(
                phase='Dissemination',
                description='Write manuscript, submit for publication',
                duration_days=90,
                dependencies=['Analysis'],
                milestones=['Draft complete', 'Manuscript submitted', 'Paper accepted']
            )
        ]

        return timeline

    def _plan_outcomes(self, hypothesis: Hypothesis) -> tuple:
        """Plan primary and secondary outcomes"""
        primary = f"Change in {hypothesis.variables['dependent']} from baseline to endpoint"

        secondary = [
            f"Subgroup analyses by {hypothesis.variables['control'].split(',')[0].strip()}",
            "Dose-response relationship",
            "Time to effect",
            "Safety and tolerability"
        ]

        return primary, secondary

    def _select_statistics(self, hypothesis: Hypothesis, design_type: str) -> List[str]:
        """Select statistical methods"""
        methods = []

        if design_type == 'randomized_controlled':
            methods.extend([
                'Intention-to-treat analysis',
                'Independent samples t-test or Mann-Whitney U',
                'Linear mixed models for repeated measures',
                'Subgroup analyses with interaction tests'
            ])
        elif design_type == 'factorial':
            methods.extend([
                'Factorial ANOVA',
                'Main effects and interaction effects',
                'Post-hoc pairwise comparisons with Bonferroni correction'
            ])
        else:
            methods.extend([
                'Regression analysis',
                'Propensity score adjustment',
                'Sensitivity analyses'
            ])

        methods.append(f'False discovery rate correction for {len(self._plan_outcomes(hypothesis)[1])} secondary outcomes')

        return methods

    def _plan_data_collection(self, hypothesis: Hypothesis, power: PowerAnalysis) -> str:
        """Plan data collection strategy"""
        return (
            f"Collect {hypothesis.variables['dependent']} measurements from {power.required_sample_size} subjects "
            f"using validated instruments. Data entry will be performed in duplicate with automated range checks. "
            f"Missing data will be handled using multiple imputation if <10% missing, otherwise complete case analysis. "
            f"All data will be stored in encrypted databases with access logs."
        )

    def _generate_qa_measures(self, hypothesis: Hypothesis, design_type: str) -> List[str]:
        """Generate quality assurance measures"""
        return [
            'Standard operating procedures (SOPs) for all procedures',
            'Training and certification of all staff',
            'Regular calibration of equipment',
            'Double data entry with discrepancy resolution',
            'Periodic audits of data quality',
            'Protocol deviations documented and reviewed',
            'Data and safety monitoring board (if applicable)',
            'Pre-registration of analysis plan'
        ]

    def _identify_risks(self, hypothesis: Hypothesis) -> List[Dict[str, str]]:
        """Identify risks and mitigations"""
        return [
            {
                'risk': 'Recruitment failure / low enrollment',
                'mitigation': 'Multiple recruitment strategies, longer recruitment period, broaden inclusion criteria'
            },
            {
                'risk': 'High dropout / loss to follow-up',
                'mitigation': 'Participant retention incentives, frequent contact, minimize burden'
            },
            {
                'risk': 'Equipment failure',
                'mitigation': 'Backup equipment, maintenance contracts, alternative measurement methods'
            },
            {
                'risk': 'Null result / insufficient power',
                'mitigation': 'Interim power analysis, sample size re-estimation, combine with meta-analysis'
            }
        ]

    def _identify_ethics_issues(self, hypothesis: Hypothesis) -> List[str]:
        """Identify ethics considerations"""
        ethics = [
            'Obtain IRB/ethics committee approval',
            'Informed consent from all participants',
            'Data privacy and confidentiality protections',
            'Adverse event monitoring and reporting'
        ]

        if hypothesis.domain == 'medicine':
            ethics.extend([
                'Clinical trial registration (ClinicalTrials.gov)',
                'Data safety monitoring board oversight',
                'Stopping rules for efficacy or futility'
            ])

        return ethics

    def _estimate_confidence(self, hypothesis: Hypothesis, power: PowerAnalysis) -> float:
        """Estimate confidence in protocol"""
        score = 0.7  # Base confidence

        # Adjust for power
        if power.power >= 0.9:
            score += 0.1
        elif power.power < 0.7:
            score -= 0.1

        # Adjust for effect size
        if hypothesis.expected_effect_size == 'large':
            score += 0.1
        elif hypothesis.expected_effect_size == 'small':
            score -= 0.1

        # Adjust for prior evidence
        if 'strong' in hypothesis.prior_evidence.lower() or 'robust' in hypothesis.prior_evidence.lower():
            score += 0.1

        return min(0.95, max(0.5, score))

    def _identify_limitations(self, hypothesis: Hypothesis, design_type: str) -> List[str]:
        """Identify protocol limitations"""
        limitations = [
            f'Statistical power based on assumed effect size ({hypothesis.expected_effect_size})',
            'Generalizability limited to studied population'
        ]

        if design_type == 'observational':
            limitations.append('Causality cannot be established from observational design')

        if hypothesis.domain == 'medicine':
            limitations.append('Short-term outcomes only; long-term effects unknown')

        return limitations

    def _generate_alternatives(self, hypothesis: Hypothesis) -> List[str]:
        """Generate alternative designs considered"""
        alternatives = [
            'Cross-sectional design (faster but no temporal ordering)',
            'Case-control design (more efficient but prone to bias)',
            'Larger sample with less intensive measurements'
        ]

        if hypothesis.domain == 'medicine':
            alternatives.append('Multi-site design (better generalizability, higher coordination costs)')

        return alternatives

def main():
    parser = argparse.ArgumentParser(description="ExperimentDesigner - Automated Protocol Generator")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Submit hypothesis
    submit_parser = subparsers.add_parser('submit', help='Submit a hypothesis')
    submit_parser.add_argument('--statement', required=True, help='Hypothesis statement')
    submit_parser.add_argument('--domain', required=True, help='Research domain')
    submit_parser.add_argument('--independent', required=True, help='Independent variable(s)')
    submit_parser.add_argument('--dependent', required=True, help='Dependent variable(s)')
    submit_parser.add_argument('--controls', required=True, help='Control variables (comma-separated)')
    submit_parser.add_argument('--effect-size', required=True, choices=['small', 'medium', 'large'])
    submit_parser.add_argument('--mechanism', required=True, help='Proposed mechanism')
    submit_parser.add_argument('--prior-evidence', required=True, help='Prior evidence for hypothesis')

    # Design protocol
    design_parser = subparsers.add_parser('design', help='Design experiment protocol')
    design_parser.add_argument('--hypothesis-id', type=int, required=True, help='Hypothesis ID')

    args = parser.parse_args()
    designer = ExperimentDesigner()

    if args.command == 'submit':
        controls = [c.strip() for c in args.controls.split(',')]
        hypothesis = designer.submit_hypothesis(
            statement=args.statement,
            domain=args.domain,
            independent_var=args.independent,
            dependent_var=args.dependent,
            control_vars=controls,
            expected_effect_size=args.effect_size,
            mechanism=args.mechanism,
            prior_evidence=args.prior_evidence
        )

        print("\nHypothesis submitted successfully!")
        print(f"Hypothesis ID: {hypothesis.hypothesis_id}")
        print(f"Statement: {hypothesis.statement}")
        print(f"Domain: {hypothesis.domain}")
        print(f"\nNext: Run 'python designer.py design --hypothesis-id {hypothesis.hypothesis_id}' to generate protocol")

    elif args.command == 'design':
        protocol = designer.design_protocol(args.hypothesis_id)
        hypothesis = designer.hypotheses[args.hypothesis_id]

        print("\n" + "="*70)
        print("EXPERIMENT PROTOCOL")
        print("="*70)

        print(f"\nHypothesis: {hypothesis.statement}")
        print(f"Domain: {hypothesis.domain}")
        print(f"Expected effect: {hypothesis.expected_effect_size}")

        print("\n🔬 EXPERIMENTAL DESIGN")
        print(f"  Design type: {protocol.design_type}")
        print(f"  Blinding: {protocol.blind_level}")
        print(f"  Randomization: {protocol.randomization_method}")

        print("\n📊 POWER ANALYSIS")
        print(f"  Effect size (Cohen's d): {protocol.power_analysis.effect_size}")
        print(f"  Alpha: {protocol.power_analysis.alpha}")
        print(f"  Power: {protocol.power_analysis.power}")
        print(f"  Required sample size: {protocol.power_analysis.required_sample_size}")
        print(f"  Groups: {protocol.power_analysis.groups} ({protocol.power_analysis.observations_per_group} per group)")
        print(f"  Statistical test: {protocol.power_analysis.statistical_test}")

        print("\n📝 VARIABLES")
        print(f"  Independent: {', '.join(protocol.independent_vars)}")
        print(f"  Dependent: {', '.join(protocol.dependent_vars)}")
        print(f"  Controls:")
        for cv in protocol.control_vars:
            print(f"    • {cv.name}: {cv.type} - {cv.justification}")

        print("\n🔄 PROCEDURE")
        for step in protocol.steps:
            print(f"\n  Step {step.step_number}: {step.action}")
            print(f"    Duration: {step.duration}")
            print(f"    Equipment: {', '.join(step.equipment_needed)}")
            if step.safety_notes:
                print(f"    Safety: {', '.join(step.safety_notes)}")

        print("\n⚙️  EQUIPMENT & MATERIALS")
        for eq in protocol.equipment_list:
            print(f"  • {eq.name}: {eq.quantity} {eq.unit} - ${eq.estimated_cost:,.2f}")

        print("\n👥 PERSONNEL")
        for role, count in protocol.personnel_needed.items():
            print(f"  • {role}: {count}")

        print("\n📅 TIMELINE")
        for phase in protocol.timeline:
            print(f"\n  {phase.phase} ({phase.duration_days} days)")
            print(f"    {phase.description}")
            print(f"    Milestones: {', '.join(phase.milestones)}")

        print(f"\n  Total duration: {protocol.total_duration_days} days ({protocol.total_duration_days/30:.1f} months)")

        print("\n💰 BUDGET")
        for category, amount in protocol.cost_breakdown.items():
            print(f"  {category.title()}: ${amount:,.2f}")
        print(f"  TOTAL: ${protocol.total_cost:,.2f}")

        print("\n📈 ANALYSIS PLAN")
        print(f"  Primary outcome: {protocol.primary_outcome}")
        print(f"  Secondary outcomes:")
        for outcome in protocol.secondary_outcomes:
            print(f"    • {outcome}")
        print(f"  Statistical methods:")
        for method in protocol.statistical_methods:
            print(f"    • {method}")

        print("\n✅ QUALITY ASSURANCE")
        for qa in protocol.quality_assurance:
            print(f"  • {qa}")

        print("\n⚠️  RISKS & MITIGATIONS")
        for risk_item in protocol.risks:
            print(f"  Risk: {risk_item['risk']}")
            print(f"  Mitigation: {risk_item['mitigation']}")
            print()

        print("🔒 ETHICS CONSIDERATIONS")
        for eth in protocol.ethics_considerations:
            print(f"  • {eth}")

        print(f"\n🎯 PROTOCOL CONFIDENCE: {protocol.confidence_score*100:.0f}%")

        print("\n⚡ LIMITATIONS")
        for lim in protocol.limitations:
            print(f"  • {lim}")

        print("\n🔄 ALTERNATIVES CONSIDERED")
        for alt in protocol.alternatives_considered:
            print(f"  • {alt}")

if __name__ == '__main__':
    main()
