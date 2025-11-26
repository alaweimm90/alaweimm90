#!/usr/bin/env python3
"""
Template Generator - Automated README customization tool

Generates customized README files from templates by substituting variables.
Supports YAML/JSON configuration files and interactive prompts.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TemplateConfig:
    """Configuration for template substitution"""
    template_path: Path
    output_path: Path
    variables: Dict[str, str]


class TemplateGenerator:
    """Generates customized documents from template files"""

    VARIABLE_PATTERN = re.compile(r'\{\{([A-Z_][A-Z0-9_]*)\}\}')

    def __init__(self, template_dir: Path = None):
        """
        Initialize template generator

        Args:
            template_dir: Directory containing templates (default: current dir)
        """
        self.template_dir = template_dir or Path.cwd()

    def find_variables(self, template_content: str) -> set:
        """Extract all template variables from content"""
        return set(self.VARIABLE_PATTERN.findall(template_content))

    def load_template(self, template_path: Path) -> str:
        """Load template file"""
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text()

    def load_variables(self, config_path: Path) -> Dict[str, str]:
        """Load variables from YAML or JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        content = config_path.read_text()

        if config_path.suffix == '.json':
            return json.loads(content)

        # Simple YAML parsing (minimal, for key=value pairs)
        variables = {}
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                variables[key.strip()] = value.strip().strip('"\'')
        return variables

    def substitute(self, template: str, variables: Dict[str, str]) -> str:
        """Substitute variables in template"""
        result = template

        for var_name, var_value in variables.items():
            pattern = f'{{{{{var_name}}}}}'
            result = result.replace(pattern, str(var_value))

        return result

    def validate(self, template: str, variables: Dict[str, str]) -> Dict[str, list]:
        """Validate template and variables"""
        issues = {
            'missing_variables': [],
            'unused_variables': []
        }

        # Find variables in template
        template_vars = self.find_variables(template)

        # Check for missing variable values
        for var in template_vars:
            if var not in variables or not variables[var]:
                issues['missing_variables'].append(var)

        # Check for unused variables
        for var in variables:
            if var not in template_vars:
                issues['unused_variables'].append(var)

        return issues

    def generate(self, config: TemplateConfig) -> str:
        """Generate document from template with variables"""
        # Load template
        template = self.load_template(config.template_path)

        # Validate
        issues = self.validate(template, config.variables)

        if issues['missing_variables']:
            print(f"⚠️  Missing variables: {', '.join(issues['missing_variables'])}")

        if issues['unused_variables']:
            print(f"ℹ️  Unused variables: {', '.join(issues['unused_variables'])}")

        # Substitute
        result = self.substitute(template, config.variables)

        # Write output
        config.output_path.write_text(result)
        print(f"✅ Generated: {config.output_path}")

        return result

    def interactive_prompt(self, template_path: Path) -> Dict[str, str]:
        """Prompt user for variable values interactively"""
        template = self.load_template(template_path)
        variables = self.find_variables(template)

        values = {}
        print(f"\nGenerating from: {template_path.name}")
        print(f"Found {len(variables)} variables\n")

        for var in sorted(variables):
            prompt_text = var.replace('_', ' ').title()
            value = input(f"{prompt_text}: ").strip()
            if value:
                values[var] = value

        return values


def main():
    """CLI interface for template generator"""
    if len(sys.argv) < 2:
        print("Usage: python template_generator.py [profile|org|consumer] [--config FILE] [--output FILE] [--interactive]")
        print("\nExamples:")
        print("  # Interactive mode (prompts for variables)")
        print("  python template_generator.py profile --interactive --output README.md")
        print("\n  # Config file mode")
        print("  python template_generator.py org --config org-vars.json --output README.md")
        print("\n  # Help")
        print("  python template_generator.py --help")
        sys.exit(1)

    template_type = sys.argv[1]
    interactive = '--interactive' in sys.argv
    config_file = None
    output_file = None

    # Parse arguments
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    # Determine template path
    template_map = {
        'profile': '.metaHub/templates/profiles/README_PROFILE_TEMPLATE.md',
        'org': '.metaHub/templates/organizations/README_ORG_TEMPLATE.md',
        'consumer': '.metaHub/templates/consumer-repos/README_CONSUMER_TEMPLATE.md',
    }

    if template_type not in template_map:
        print(f"Unknown template type: {template_type}")
        print(f"Available: {', '.join(template_map.keys())}")
        sys.exit(1)

    template_path = Path(template_map[template_type])
    output_path = Path(output_file or 'README.md')

    # Generate
    generator = TemplateGenerator()

    try:
        if interactive:
            variables = generator.interactive_prompt(template_path)
        elif config_file:
            variables = generator.load_variables(Path(config_file))
        else:
            print("Error: Provide --config FILE or --interactive")
            sys.exit(1)

        config = TemplateConfig(
            template_path=template_path,
            output_path=output_path,
            variables=variables
        )

        generator.generate(config)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
