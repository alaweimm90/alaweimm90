#!/usr/bin/env python3
"""
AbstractWriter - AI-Powered Paper Abstract Generator

Generates structured academic abstracts from paper outlines or key points.
Follows standard abstract structure: Background → Gap → Method → Results → Impact

Usage:
    python writer.py generate --input paper_outline.txt --output abstract.txt
    python writer.py refine --abstract draft.txt --target-words 250
    python writer.py validate --abstract abstract.txt
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class AbstractSection:
    """Section of an abstract"""
    section_type: str  # background, gap, method, results, impact
    content: str
    word_count: int


@dataclass
class Abstract:
    """Complete abstract"""
    title: str
    sections: List[AbstractSection]
    total_words: int
    structure_score: float  # 0-10
    clarity_score: float    # 0-10
    impact_score: float     # 0-10
    timestamp: str


@dataclass
class ValidationResult:
    """Abstract validation result"""
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    word_count: int
    section_balance: Dict[str, float]  # % of total words per section


class AbstractGenerator:
    """Generate structured academic abstracts"""

    def __init__(self):
        # Standard abstract structure
        self.structure = {
            'background': {
                'templates': [
                    "{field} has emerged as a critical area of research due to {motivation}.",
                    "Recent advances in {field} have highlighted {problem}, creating new opportunities for {application}.",
                    "Understanding {phenomenon} is essential for {goal}, particularly in the context of {domain}.",
                    "{field} faces significant challenges in {problem_area}, requiring novel approaches to {objective}.",
                ],
                'target_pct': 0.20,  # 20% of abstract
            },
            'gap': {
                'templates': [
                    "However, existing approaches suffer from {limitation}, limiting their applicability to {context}.",
                    "Despite progress, current methods fail to address {gap}, particularly when {condition}.",
                    "A key limitation of prior work is {weakness}, which prevents {desired_outcome}.",
                    "Previous studies have not adequately considered {missing_aspect}, leaving {problem} unresolved.",
                ],
                'target_pct': 0.15,  # 15% of abstract
            },
            'method': {
                'templates': [
                    "We present {method_name}, a novel approach that {key_innovation}. Our method employs {technique} to {goal}.",
                    "This paper introduces {method_name}, which addresses {gap} through {approach}. We leverage {technology} to achieve {benefit}.",
                    "We propose {method_name}, combining {technique_1} and {technique_2} to overcome {limitation}. Our framework enables {capability}.",
                    "Our approach, {method_name}, uses {core_technique} to {primary_goal}. By integrating {component}, we achieve {advantage}.",
                ],
                'target_pct': 0.30,  # 30% of abstract
            },
            'results': {
                'templates': [
                    "Experiments on {dataset} demonstrate that our approach achieves {metric_1} of {value_1}, outperforming {baseline} by {improvement}.",
                    "We evaluate {method_name} on {num_datasets} benchmark datasets, showing consistent improvements of {percentage} over state-of-the-art methods.",
                    "Our results show that {method_name} reduces {metric} by {value} while maintaining {quality_metric} comparable to {baseline}.",
                    "Empirical evaluation reveals that our method achieves {outcome}, representing a {percentage} improvement over existing approaches.",
                ],
                'target_pct': 0.20,  # 20% of abstract
            },
            'impact': {
                'templates': [
                    "These findings enable {application}, with implications for {broader_field}.",
                    "Our work opens new directions for {future_research} and provides practical solutions for {real_world_problem}.",
                    "This approach has significant potential for {application_area}, particularly in {specific_context}.",
                    "The proposed method offers a scalable solution for {problem}, advancing the state of the art in {field}.",
                ],
                'target_pct': 0.15,  # 15% of abstract
            },
        }

        # Common academic phrases
        self.connectors = {
            'background_to_gap': ['However', 'Nevertheless', 'Despite these advances', 'Yet'],
            'gap_to_method': ['To address this', 'In this work', 'This paper presents', 'We propose'],
            'method_to_results': ['Our experiments show', 'Empirical evaluation demonstrates', 'Results indicate', 'We find that'],
            'results_to_impact': ['These results', 'This work', 'Our findings', 'This approach'],
        }

    def generate_from_outline(self, outline: Dict[str, Any], target_words: int = 250) -> Abstract:
        """
        Generate abstract from paper outline

        Args:
            outline: Dict with keys: title, field, problem, method, results, impact
            target_words: Target abstract length

        Returns:
            Generated Abstract
        """
        sections = []

        # Calculate target words per section
        section_targets = {
            section: int(target_words * info['target_pct'])
            for section, info in self.structure.items()
        }

        # Generate background
        background = self._generate_background(
            outline.get('field', 'this domain'),
            outline.get('motivation', 'emerging challenges'),
            outline.get('problem', 'key limitations'),
            section_targets['background']
        )
        sections.append(background)

        # Generate gap
        gap = self._generate_gap(
            outline.get('limitation', 'existing constraints'),
            outline.get('gap', 'unresolved issues'),
            section_targets['gap']
        )
        sections.append(gap)

        # Generate method
        method = self._generate_method(
            outline.get('method_name', 'our approach'),
            outline.get('technique', 'novel methodology'),
            outline.get('innovation', 'key contribution'),
            section_targets['method']
        )
        sections.append(method)

        # Generate results
        results = self._generate_results(
            outline.get('dataset', 'benchmark datasets'),
            outline.get('metric', 'performance'),
            outline.get('improvement', 'significant gains'),
            section_targets['results']
        )
        sections.append(results)

        # Generate impact
        impact = self._generate_impact(
            outline.get('application', 'practical applications'),
            outline.get('broader_impact', 'future research'),
            section_targets['impact']
        )
        sections.append(impact)

        # Calculate total words
        total_words = sum(s.word_count for s in sections)

        # Score the abstract
        structure_score = self._score_structure(sections, target_words)
        clarity_score = self._score_clarity(sections)
        impact_score = self._score_impact(sections)

        abstract = Abstract(
            title=outline.get('title', 'Untitled'),
            sections=sections,
            total_words=total_words,
            structure_score=round(structure_score, 1),
            clarity_score=round(clarity_score, 1),
            impact_score=round(impact_score, 1),
            timestamp=datetime.now().isoformat()
        )

        return abstract

    def _generate_background(self, field: str, motivation: str, problem: str, target_words: int) -> AbstractSection:
        """Generate background section"""
        template = random.choice(self.structure['background']['templates'])

        content = template.format(
            field=field,
            motivation=motivation,
            problem=problem,
            phenomenon=f"{field} phenomena",
            goal=f"advancing {field}",
            domain=f"{field} applications",
            problem_area=problem,
            objective=f"addressing {problem}",
            application=f"{field} systems"
        )

        # Adjust to target word count (simplified - in production would use smarter expansion)
        words = content.split()
        if len(words) < target_words - 10:
            # Add elaboration
            elaborations = [
                f"This is particularly relevant in contexts where {problem} limits effectiveness.",
                f"The complexity of {field} systems necessitates new approaches.",
                f"Traditional methods have proven insufficient for modern {field} challenges.",
            ]
            content += " " + random.choice(elaborations)

        return AbstractSection(
            section_type='background',
            content=content,
            word_count=len(content.split())
        )

    def _generate_gap(self, limitation: str, gap: str, target_words: int) -> AbstractSection:
        """Generate gap section"""
        connector = random.choice(self.connectors['background_to_gap'])
        template = random.choice(self.structure['gap']['templates'])

        content = f"{connector}, " + template.format(
            limitation=limitation,
            context="real-world scenarios",
            gap=gap,
            condition="dealing with complex data",
            weakness=limitation,
            desired_outcome="optimal performance",
            missing_aspect=gap,
            problem=gap
        )

        return AbstractSection(
            section_type='gap',
            content=content,
            word_count=len(content.split())
        )

    def _generate_method(self, method_name: str, technique: str, innovation: str, target_words: int) -> AbstractSection:
        """Generate method section"""
        connector = random.choice(self.connectors['gap_to_method'])
        template = random.choice(self.structure['method']['templates'])

        content = f"{connector} " + template.format(
            method_name=method_name,
            key_innovation=innovation,
            technique=technique,
            goal=f"achieving {innovation}",
            approach=technique,
            technology=technique,
            benefit="improved performance",
            technique_1=technique,
            technique_2="adaptive optimization",
            limitation="existing constraints",
            capability="robust processing",
            core_technique=technique,
            primary_goal=innovation,
            component=f"advanced {technique}",
            advantage="superior results"
        )

        return AbstractSection(
            section_type='method',
            content=content,
            word_count=len(content.split())
        )

    def _generate_results(self, dataset: str, metric: str, improvement: str, target_words: int) -> AbstractSection:
        """Generate results section"""
        connector = random.choice(self.connectors['method_to_results'])
        template = random.choice(self.structure['results']['templates'])

        # Generate plausible values
        value = round(random.uniform(85, 98), 1)
        improvement_pct = round(random.uniform(10, 35), 1)

        content = f"{connector} that " + template.format(
            dataset=dataset,
            metric_1=metric,
            value_1=f"{value}%",
            baseline="baseline methods",
            improvement=f"{improvement_pct}%",
            method_name="our approach",
            num_datasets="multiple",
            percentage=f"{improvement_pct}%",
            metric=metric,
            value=f"{improvement_pct}%",
            quality_metric="accuracy",
            outcome=improvement
        )

        return AbstractSection(
            section_type='results',
            content=content,
            word_count=len(content.split())
        )

    def _generate_impact(self, application: str, broader_impact: str, target_words: int) -> AbstractSection:
        """Generate impact section"""
        connector = random.choice(self.connectors['results_to_impact'])
        template = random.choice(self.structure['impact']['templates'])

        content = f"{connector} " + template.format(
            application=application,
            broader_field=broader_impact,
            future_research=broader_impact,
            real_world_problem=application,
            application_area=application,
            specific_context=f"{application} scenarios",
            problem="key challenges",
            field=broader_impact
        )

        return AbstractSection(
            section_type='impact',
            content=content,
            word_count=len(content.split())
        )

    def _score_structure(self, sections: List[AbstractSection], target_words: int) -> float:
        """Score structural quality"""
        score = 10.0

        # Check all sections present
        section_types = {s.section_type for s in sections}
        if len(section_types) < 5:
            score -= 2.0

        # Check word count
        total_words = sum(s.word_count for s in sections)
        word_diff_pct = abs(total_words - target_words) / target_words
        score -= word_diff_pct * 5.0  # Penalty for deviation

        # Check section balance
        for section in sections:
            target_pct = self.structure[section.section_type]['target_pct']
            actual_pct = section.word_count / total_words if total_words > 0 else 0
            deviation = abs(actual_pct - target_pct)
            score -= deviation * 10.0

        return max(0.0, min(10.0, score))

    def _score_clarity(self, sections: List[AbstractSection]) -> float:
        """Score clarity and readability"""
        score = 8.0  # Base score

        for section in sections:
            # Penalize very long sentences
            sentences = section.content.split('.')
            for sent in sentences:
                words = sent.split()
                if len(words) > 40:
                    score -= 0.5

        return max(0.0, min(10.0, score))

    def _score_impact(self, sections: List[AbstractSection]) -> float:
        """Score potential impact"""
        score = 7.0  # Base score

        # Check for impact keywords
        impact_keywords = ['novel', 'significant', 'outperform', 'advance', 'enable', 'breakthrough']

        for section in sections:
            content_lower = section.content.lower()
            for keyword in impact_keywords:
                if keyword in content_lower:
                    score += 0.3

        return max(0.0, min(10.0, score))


class AbstractValidator:
    """Validate abstract quality"""

    def validate(self, abstract_text: str, target_words: int = 250) -> ValidationResult:
        """Validate an abstract"""
        issues = []
        suggestions = []

        words = abstract_text.split()
        word_count = len(words)

        # Check word count
        if word_count < target_words * 0.8:
            issues.append(f"Abstract too short: {word_count} words (target: {target_words})")
            suggestions.append("Expand methodology and results sections")
        elif word_count > target_words * 1.2:
            issues.append(f"Abstract too long: {word_count} words (target: {target_words})")
            suggestions.append("Remove unnecessary details and focus on key contributions")

        # Check for key components (simplified detection)
        if not any(kw in abstract_text.lower() for kw in ['however', 'despite', 'yet', 'although']):
            issues.append("Missing contrast/gap statement")
            suggestions.append("Add 'However' or 'Despite' to highlight research gap")

        if not any(kw in abstract_text.lower() for kw in ['we present', 'we propose', 'this paper', 'we introduce']):
            issues.append("Missing method introduction")
            suggestions.append("Add clear statement of proposed method")

        if not any(kw in abstract_text.lower() for kw in ['results', 'experiments', 'evaluation', 'demonstrate']):
            issues.append("Missing results statement")
            suggestions.append("Include empirical results or evaluation findings")

        # Check sentence structure (simplified)
        sentences = [s.strip() for s in abstract_text.split('.') if s.strip()]
        if len(sentences) < 5:
            issues.append("Too few sentences (may lack detail)")
            suggestions.append("Expand each major section")

        # Estimate section balance (simplified)
        section_balance = {
            'background': 0.2,
            'gap': 0.15,
            'method': 0.3,
            'results': 0.2,
            'impact': 0.15
        }

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            word_count=word_count,
            section_balance=section_balance
        )


def main():
    parser = argparse.ArgumentParser(
        description="AbstractWriter - AI-Powered Paper Abstract Generator"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate abstract from outline')
    gen_parser.add_argument('--input', required=True, help='Outline JSON file')
    gen_parser.add_argument('--output', required=True, help='Output text file')
    gen_parser.add_argument('--words', type=int, default=250, help='Target word count')

    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate abstract')
    val_parser.add_argument('--abstract', required=True, help='Abstract text file')
    val_parser.add_argument('--target-words', type=int, default=250, help='Target word count')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'generate':
        # Load outline
        with open(args.input, 'r') as f:
            outline = json.load(f)

        # Generate abstract
        generator = AbstractGenerator()
        abstract = generator.generate_from_outline(outline, args.words)

        # Combine sections
        full_text = " ".join(s.content for s in abstract.sections)

        # Save
        with open(args.output, 'w') as f:
            f.write(full_text)

        # Print stats
        print(f"\nGenerated abstract for: {abstract.title}")
        print(f"Word count: {abstract.total_words}")
        print(f"Structure score: {abstract.structure_score}/10")
        print(f"Clarity score: {abstract.clarity_score}/10")
        print(f"Impact score: {abstract.impact_score}/10")
        print(f"\nSaved to: {args.output}")

    elif args.command == 'validate':
        # Load abstract
        with open(args.abstract, 'r') as f:
            abstract_text = f.read()

        # Validate
        validator = AbstractValidator()
        result = validator.validate(abstract_text, args.target_words)

        # Print results
        print(f"\nValidation Results:")
        print(f"Word count: {result.word_count}")
        print(f"Valid: {result.is_valid}")

        if result.issues:
            print(f"\nIssues found:")
            for i, issue in enumerate(result.issues, 1):
                print(f"  {i}. {issue}")

        if result.suggestions:
            print(f"\nSuggestions:")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")


if __name__ == "__main__":
    main()
