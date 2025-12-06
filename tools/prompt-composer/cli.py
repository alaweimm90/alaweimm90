"""CLI for prompt composition"""
import json
from pathlib import Path
from composer import PromptComposer
import sys as system

def main():
    if len(system.argv) < 2:
        print("Usage: python cli.py <template> [variables.json]")
        print("\nExamples:")
        print("  python cli.py templates/fullstack-workflow.md")
        print("  python cli.py templates/fullstack-workflow.md vars.json")
        return
    
    template_path = system.argv[1]
    variables = {}
    
    if len(system.argv) > 2:
        vars_file = Path(system.argv[2])
        if vars_file.exists():
            variables = json.loads(vars_file.read_text())
    
    composer = PromptComposer()
    template = composer.load_template(template_path)
    result = composer.compose(template, variables)
    
    # Write to stdout with UTF-8 encoding
    system.stdout.reconfigure(encoding='utf-8')
    print(result)

if __name__ == "__main__":
    main()
