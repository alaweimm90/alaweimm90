"""Compose prompts from components"""
import re
from pathlib import Path
from typing import Dict

class PromptComposer:
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
        self.prompts_dir = Path(prompts_dir)
    
    def compose(self, template: str, variables: Dict[str, str] = None) -> str:
        """Compose prompt from template with variable substitution"""
        result = template
        
        # Variable substitution first: {{var_name}}
        if variables:
            result = self._substitute_variables(result, variables)
        
        # Process components: {{component:name|param=value}}
        result = self._process_components(result)
        
        # Include other prompts: {{include:path/to/prompt.md}}
        result = self._process_includes(result)
        
        return result
    
    def _process_includes(self, text: str) -> str:
        """Process {{include:path}} directives"""
        pattern = r'\{\{include:([^}]+)\}\}'
        
        def replace_include(match):
            path = match.group(1).strip()
            file_path = self.prompts_dir / path
            if file_path.exists():
                return file_path.read_text(encoding='utf-8')
            return f"[ERROR: {path} not found]"
        
        return re.sub(pattern, replace_include, text)
    
    def _substitute_variables(self, text: str, variables: Dict[str, str]) -> str:
        """Replace {{var}} with values"""
        for key, value in variables.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text
    
    def _process_components(self, text: str) -> str:
        """Process {{component:name|param=value}} directives"""
        from components import PromptComponents
        pattern = r'\{\{component:([^|}]+)(?:\|([^}]+))?\}\}'
        
        def replace_component(match):
            name = match.group(1).strip()
            params_str = match.group(2)
            params = {}
            if params_str:
                for param in params_str.split('|'):
                    if '=' in param:
                        k, v = param.split('=', 1)
                        params[k.strip()] = v.strip()
            return PromptComponents.get_component(name, **params)
        
        return re.sub(pattern, replace_component, text)
    
    def load_template(self, template_path: str) -> str:
        """Load template file"""
        path = Path(template_path)
        if not path.is_absolute():
            # Check composer templates first
            composer_template = Path(__file__).parent / path
            if composer_template.exists():
                path = composer_template
            else:
                path = self.prompts_dir / path
        return path.read_text(encoding='utf-8')
    
    def save_composed(self, content: str, output_path: str) -> None:
        """Save composed prompt"""
        Path(output_path).write_text(content, encoding='utf-8')

if __name__ == "__main__":
    composer = PromptComposer()
    
    template = """# Test Composition

{{include:superprompts/README.md}}

Project: {{project_name}}
Language: {{language}}
"""
    
    result = composer.compose(template, {
        "project_name": "MyApp",
        "language": "Python"
    })
    
    print(result[:200] + "...")
