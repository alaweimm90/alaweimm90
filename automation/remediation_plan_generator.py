#!/usr/bin/env python3
"""
Technical Debt Remediation Plan Generator

Generates detailed, actionable remediation plans with code examples
instead of attempting automated fixes for complex technical debt.
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from technical_debt_remediation import TechnicalDebtItem, DebtType, DebtSeverity


@dataclass
class RemediationTicket:
    """Represents a remediation ticket with detailed instructions."""
    ticket_id: str
    title: str
    description: str
    debt_item: TechnicalDebtItem
    priority: str
    estimated_hours: int
    code_examples: List[Dict[str, str]]
    step_by_steps: List[str]
    verification_steps: List[str]
    related_files: List[str]
    created_at: float


class RemediationPlanGenerator:
    """Generates detailed remediation plans for technical debt items."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.ticket_templates = self._load_ticket_templates()

    def generate_remediation_plan(self, debt_items: List[TechnicalDebtItem]) -> List[RemediationTicket]:
        """Generate detailed remediation tickets for technical debt items."""

        print(f"📋 Generating detailed remediation plans for {len(debt_items)} items...")

        tickets = []

        for item in debt_items:
            ticket = self._create_ticket_for_debt_item(item)
            tickets.append(ticket)

            print(f"   ✅ Generated ticket: {ticket.title}")

        print(f"📊 Created {len(tickets)} remediation tickets")
        return tickets

    def _create_ticket_for_debt_item(self, item: TechnicalDebtItem) -> RemediationTicket:
        """Create a detailed remediation ticket for a debt item."""

        # Get template based on debt type
        template = self.ticket_templates.get(item.type, self._get_generic_template())

        # Customize template with specific item details
        ticket = RemediationTicket(
            ticket_id=f"DEBT-{uuid.uuid4().hex[:8].upper()}",
            title=template['title'].format(description=item.description),
            description=template['description'].format(
                description=item.description,
                file_path=item.file_path,
                line_number=item.line_number or "N/A",
                business_impact=item.business_impact
            ),
            debt_item=item,
            priority=self._map_severity_to_priority(item.severity),
            estimated_hours=item.estimated_effort,
            code_examples=self._generate_code_examples(item),
            step_by_steps=template['steps'],
            verification_steps=template['verification'],
            related_files=self._find_related_files(item),
            created_at=time.time()
        )

        return ticket

    def _load_ticket_templates(self) -> Dict[DebtType, Dict[str, Any]]:
        """Load remediation templates for different debt types."""

        return {
            DebtType.CODE_COMPLEXITY: {
                'title': "Refactor Complex Code: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

This code complexity issue makes the code difficult to understand, maintain, and test.
Complex code is a common source of bugs and makes onboarding new developers challenging.

**Remediation Strategy:**
Break down complex logic into smaller, focused functions with clear responsibilities.
                """.strip(),
                'steps': [
                    "1. Analyze the complex code section to identify distinct responsibilities",
                    "2. Extract logical units into separate, well-named functions",
                    "3. Reduce nesting levels by using early returns or guard clauses",
                    "4. Add clear comments explaining the business logic",
                    "5. Ensure each function has a single, clear purpose",
                    "6. Run existing tests to ensure functionality is preserved"
                ],
                'verification': [
                    "Code complexity metrics should decrease (cyclomatic complexity)",
                    "Function names should clearly describe their purpose",
                    "Nesting levels should be reduced to ≤ 3 levels",
                    "All existing tests should pass",
                    "Code should be more readable and understandable"
                ]
            },

            DebtType.NAMING_CONSISTENCY: {
                'title': "Fix Naming Convention: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

Inconsistent naming makes the codebase harder to navigate and understand.
Following consistent naming conventions improves code readability and maintainability.

**Remediation Strategy:**
Update variable/function names to follow project conventions (snake_case for Python).
                """.strip(),
                'steps': [
                    "1. Identify all instances of the inconsistently named element",
                    "2. Determine the correct naming convention for the project",
                    "3. Choose a descriptive, snake_case name for the element",
                    "4. Update all references to use the new name",
                    "5. Update any related documentation or comments",
                    "6. Run tests to ensure no functionality is broken"
                ],
                'verification': [
                    "All naming follows project conventions (snake_case)",
                    "No broken references or imports",
                    "Tests pass with the new naming",
                    "Code is more consistent with project style"
                ]
            },

            DebtType.DOCUMENTATION_DRIFT: {
                'title': "Add Missing Documentation: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

Missing documentation makes it difficult for developers to understand code purpose
and usage, leading to slower development and potential misuse.

**Remediation Strategy:**
Add comprehensive docstrings and comments to improve code understandability.
                """.strip(),
                'steps': [
                    "1. Identify all undocumented functions and classes",
                    "2. Add docstrings following the project's documentation style",
                    "3. Document parameters, return values, and exceptions",
                    "4. Add inline comments for complex business logic",
                    "5. Update any related API documentation",
                    "6. Ensure examples in docstrings are accurate"
                ],
                'verification': [
                    "All public functions have docstrings",
                    "Docstrings follow project format and style",
                    "Parameters and return values are documented",
                    "Examples in docstrings are functional"
                ]
            },

            DebtType.ARCHITECTURAL_VIOLATIONS: {
                'title': "Fix Architectural Violation: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

Architectural violations indicate deviation from the intended system design,
leading to tight coupling and reduced maintainability.

**Remediation Strategy:**
Refactor code to align with established architectural patterns and principles.
                """.strip(),
                'steps': [
                    "1. Review the intended architecture and design patterns",
                    "2. Identify how the current code violates architectural principles",
                    "3. Design a refactoring approach that restores proper architecture",
                    "4. Extract cohesive functionality into appropriate modules",
                    "5. Reduce coupling between components",
                    "6. Update dependency injection and interfaces as needed"
                ],
                'verification': [
                    "Code follows established architectural patterns",
                    "Modules have clear responsibilities and boundaries",
                    "Coupling between components is reduced",
                    "Dependencies are properly managed"
                ]
            },

            DebtType.SECURITY_ISSUES: {
                'title': "🚨 CRITICAL: Fix Security Issue: {description}",
                'description': """
**🚨 CRITICAL SECURITY ISSUE:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

This security vulnerability poses a significant risk to the application and data.
Immediate remediation is required to prevent potential security breaches.

**Remediation Strategy:**
Remove hardcoded secrets and implement proper security practices.
                """.strip(),
                'steps': [
                    "1. IMMEDIATELY remove any hardcoded secrets from the code",
                    "2. Move secrets to environment variables or secure configuration",
                    "3. Implement proper secret management (e.g., AWS Secrets Manager, HashiCorp Vault)",
                    "4. Add validation to ensure secrets are properly loaded",
                    "5. Rotate any exposed secrets immediately",
                    "6. Add security tests to prevent future regressions"
                ],
                'verification': [
                    "No hardcoded secrets in the codebase",
                    "Secrets are loaded from secure sources",
                    "Security tests pass and prevent regressions",
                    "Access to secrets is properly audited and logged"
                ]
            },

            DebtType.PERFORMANCE_BOTTLENECKS: {
                'title': "Optimize Performance: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

Performance issues can lead to poor user experience and increased infrastructure costs.
Optimizing these bottlenecks improves system efficiency and scalability.

**Remediation Strategy:**
Replace inefficient patterns with more performant alternatives.
                """.strip(),
                'steps': [
                    "1. Profile the code to confirm performance bottleneck",
                    "2. Identify more efficient algorithms or data structures",
                    "3. Replace inefficient patterns with optimized alternatives",
                    "4. Add performance tests to measure improvement",
                    "5. Consider caching or memoization where appropriate",
                    "6. Monitor performance after optimization"
                ],
                'verification': [
                    "Performance metrics show improvement",
                    "Code uses efficient algorithms and patterns",
                    "Performance tests validate improvements",
                    "System scales better under load"
                ]
            },

            DebtType.TEST_COVERAGE_GAPS: {
                'title': "Add Test Coverage: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

Insufficient test coverage increases the risk of bugs and makes refactoring dangerous.
Adding comprehensive tests improves code reliability and developer confidence.

**Remediation Strategy:**
Add unit tests to achieve adequate coverage for critical functionality.
                """.strip(),
                'steps': [
                    "1. Identify critical functions and classes lacking tests",
                    "2. Write unit tests for normal operation scenarios",
                    "3. Add tests for edge cases and error conditions",
                    "4. Ensure test coverage meets project targets (≥ 80%)",
                    "5. Add integration tests for component interactions",
                    "6. Set up continuous integration to run tests automatically"
                ],
                'verification': [
                    "Test coverage meets project targets",
                    "Tests cover normal, edge, and error cases",
                    "All tests pass consistently",
                    "CI/CD pipeline includes automated testing"
                ]
            },

            DebtType.DEPENDENCY_BLOAT: {
                'title': "Reduce Dependency Bloat: {description}",
                'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

Too many dependencies increase the attack surface, slow down builds, and create
maintenance overhead. Reducing dependencies improves security and performance.

**Remediation Strategy:**
Review and remove unnecessary dependencies while maintaining functionality.
                """.strip(),
                'steps': [
                    "1. Audit all dependencies and their usage in the codebase",
                    "2. Identify unused or redundant dependencies",
                    "3. Replace heavy dependencies with lighter alternatives",
                    "4. Remove dependencies that are no longer needed",
                    "5. Update documentation to reflect dependency changes",
                    "6. Test thoroughly after dependency removal"
                ],
                'verification': [
                    "Unused dependencies are removed",
                    "Application functionality is preserved",
                    "Build and deployment times improve",
                    "Security attack surface is reduced"
                ]
            }
        }

    def _get_generic_template(self) -> Dict[str, Any]:
        """Generic template for unknown debt types."""
        return {
            'title': "Address Technical Debt: {description}",
            'description': """
**Technical Debt Item:** {description}
**Location:** {file_path}:{line_number}
**Business Impact:** {business_impact}

This technical debt item should be addressed to improve code quality and maintainability.
            """.strip(),
            'steps': [
                "1. Analyze the technical debt item and its impact",
                "2. Plan an appropriate remediation approach",
                "3. Implement the fix following best practices",
                "4. Test to ensure functionality is preserved",
                "5. Update documentation as needed"
            ],
            'verification': [
                "Technical debt is properly addressed",
                "Code quality improves",
                "Functionality is preserved",
                "Tests pass"
            ]
        }

    def _map_severity_to_priority(self, severity: DebtSeverity) -> str:
        """Map debt severity to ticket priority."""
        mapping = {
            DebtSeverity.CRITICAL: "Critical",
            DebtSeverity.HIGH: "High",
            DebtSeverity.MEDIUM: "Medium",
            DebtSeverity.LOW: "Low"
        }
        return mapping.get(severity, "Medium")

    def _generate_code_examples(self, item: TechnicalDebtItem) -> List[Dict[str, str]]:
        """Generate before/after code examples for the debt item."""

        examples = []

        if item.type == DebtType.CODE_COMPLEXITY:
            examples.append({
                'title': 'Extract Complex Logic',
                'before': '''def process_data(data, options, config, settings):
    if data is not None:
        if options.get('validate', False):
            if config.get('strict', True):
                for item in data:
                    if item.get('active', True):
                        if settings.get('check_permissions', True):
                            if item.get('permission', 'read') == 'write':
                                # Complex nested logic here
                                pass''',
                'after': '''def process_data(data, options, config, settings):
    if not data:
        return []

    if not should_validate_data(options, config):
        return filter_active_items(data, settings)

    return validate_and_process_items(data, settings)

def should_validate_data(options, config):
    return options.get('validate', False) and config.get('strict', True)

def filter_active_items(data, settings):
    return [item for item in data
            if item.get('active', True) and
               has_write_permission(item, settings)]

def has_write_permission(item, settings):
    return (settings.get('check_permissions', True) and
            item.get('permission', 'read') == 'write')'''
            })

        elif item.type == DebtType.NAMING_CONSISTENCY:
            examples.append({
                'title': 'Fix Naming Convention',
                'before': 'def processData(userData, configSettings):',
                'after': 'def process_data(user_data, config_settings):'
            })

        elif item.type == DebtType.DOCUMENTATION_DRIFT:
            examples.append({
                'title': 'Add Documentation',
                'before': '''def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total''',
                'after': '''def calculate_total(items):
    """Calculate the total price for a list of items.

    Args:
        items (list): List of item objects with price and quantity attributes

    Returns:
        float: Total price of all items

    Raises:
        ValueError: If items is not a list or contains invalid items
    """
    if not isinstance(items, list):
        raise ValueError("Items must be a list")

    total = 0.0
    for item in items:
        if not hasattr(item, 'price') or not hasattr(item, 'quantity'):
            raise ValueError("Item must have price and quantity attributes")
        total += item.price * item.quantity
    return total'''
            })

        elif item.type == DebtType.SECURITY_ISSUES:
            examples.append({
                'title': 'Remove Hardcoded Secret',
                'before': '''# Example only – do NOT hardcode real secrets like this
example_api_key = "sk-EXAMPLE-ONLY-NOT-VALID"  # DANGEROUS IN REAL CODE
client = APIClient(example_api_key)''',
                'after': '''import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable required")

client = APIClient(api_key)'''
            })

        return examples

    def _find_related_files(self, item: TechnicalDebtItem) -> List[str]:
        """Find files that might be related to this debt item."""

        related = [item.file_path]

        # Add common related files based on debt type
        if item.type == DebtType.DOCUMENTATION_DRIFT:
            related.extend([
                "README.md",
                "docs/api.md",
                "docs/usage.md"
            ])

        elif item.type == DebtType.TEST_COVERAGE_GAPS:
            test_file = item.file_path.replace('.py', '_test.py')
            related.append(test_file)

        elif item.type == DebtType.DEPENDENCY_BLOAT:
            related.extend([
                "requirements.txt",
                "setup.py",
                "pyproject.toml"
            ])

        return [f for f in related if Path(self.project_root / f).exists()]

    def export_tickets_to_markdown(self, tickets: List[RemediationTicket],
                                 output_file: str = "remediation_plan.md") -> str:
        """Export tickets to a markdown file for easy review."""

        output_path = self.project_root / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Technical Debt Remediation Plan\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tickets: {len(tickets)}\n\n")

            # Group by priority
            tickets_by_priority = {}
            for ticket in tickets:
                if ticket.priority not in tickets_by_priority:
                    tickets_by_priority[ticket.priority] = []
                tickets_by_priority[ticket.priority].append(ticket)

            priority_order = ['Critical', 'High', 'Medium', 'Low']

            for priority in priority_order:
                if priority in tickets_by_priority:
                    f.write(f"## {priority} Priority\n\n")

                    for ticket in tickets_by_priority[priority]:
                        f.write(f"### {ticket.ticket_id}: {ticket.title}\n\n")
                        f.write(f"**Priority:** {ticket.priority}  \n")
                        f.write(f"**Estimated Hours:** {ticket.estimated_hours}  \n")
                        f.write(f"**File:** `{ticket.debt_item.file_path}`  \n")
                        f.write(f"**Line:** {ticket.debt_item.line_number or 'N/A'}  \n\n")

                        f.write(f"**Description:**\n{ticket.description}\n\n")

                        f.write("**Step-by-Step Remediation:**\n")
                        for step in ticket.step_by_steps:
                            f.write(f"{step}\n")
                        f.write("\n")

                        f.write("**Verification Steps:**\n")
                        for step in ticket.verification_steps:
                            f.write(f"- [ ] {step}\n")
                        f.write("\n")

                        if ticket.code_examples:
                            f.write("**Code Examples:**\n")
                            for example in ticket.code_examples:
                                f.write(f"##### {example['title']}\n\n")
                                f.write(f"**Before:**\n```python\n{example['before']}\n```\n\n")
                                f.write(f"**After:**\n```python\n{example['after']}\n```\n\n")

                        if ticket.related_files:
                            f.write("**Related Files:**\n")
                            for file in ticket.related_files:
                                f.write(f"- `{file}`\n")
                            f.write("\n")

                        f.write("---\n\n")

        print(f"📄 Remediation plan exported to: {output_path}")
        return str(output_path)


# Demonstration
async def demonstrate_remediation_planning():
    """Demonstrate the remediation plan generator."""

    from technical_debt_remediation import TechnicalDebtScanner, DebtType, DebtSeverity

    print("📋 Technical Debt Remediation Plan Generator")
    print("=" * 80)
    print("This system generates detailed, actionable remediation plans")
    print("with code examples instead of attempting automated fixes.\n")

    # Create sample debt items for demonstration
    sample_items = [
        TechnicalDebtItem(
            id="sample_1",
            type=DebtType.CODE_COMPLEXITY,
            severity=DebtSeverity.HIGH,
            description="Deeply nested code with multiple conditionals",
            file_path="src/complex_module.py",
            line_number=45,
            estimated_effort=4,
            business_impact="Difficult to understand and maintain",
            remediation_suggestion="Extract nested logic into separate functions",
            discovered_at=time.time(),
            metadata={"complexity_score": 12}
        ),
        TechnicalDebtItem(
            id="sample_2",
            type=DebtType.SECURITY_ISSUES,
            severity=DebtSeverity.CRITICAL,
            description="Hardcoded API key detected",
            file_path="src/config.py",
            line_number=12,
            estimated_effort=2,
            business_impact="Security vulnerability - exposed credentials",
            remediation_suggestion="Move to environment variables",
            discovered_at=time.time(),
            metadata={"secret_type": "api_key"}
        ),
        TechnicalDebtItem(
            id="sample_3",
            type=DebtType.DOCUMENTATION_DRIFT,
            severity=DebtSeverity.MEDIUM,
            description="Missing docstrings for public functions",
            file_path="src/utils.py",
            line_number=1,
            estimated_effort=3,
            business_impact="Reduced code understandability",
            remediation_suggestion="Add comprehensive docstrings",
            discovered_at=time.time(),
            metadata={"missing_docs": 5}
        )
    ]

    # Generate remediation plans
    generator = RemediationPlanGenerator(".")
    tickets = generator.generate_remediation_plan(sample_items)

    # Export to markdown
    output_file = generator.export_tickets_to_markdown(tickets)

    print(f"\n✅ Generated {len(tickets)} detailed remediation tickets")
    print(f"📄 Plan exported to: {output_file}")

    # Show summary
    print(f"\n📊 Ticket Summary:")
    priority_counts = {}
    total_hours = 0

    for ticket in tickets:
        priority_counts[ticket.priority] = priority_counts.get(ticket.priority, 0) + 1
        total_hours += ticket.estimated_hours

    for priority, count in priority_counts.items():
        print(f"   {priority}: {count} tickets")

    print(f"   Total Estimated Hours: {total_hours}")

    return tickets, output_file


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_remediation_planning())
