"""Reusable prompt components"""
from typing import Dict

class PromptComponents:
    """Library of reusable prompt components"""
    
    @staticmethod
    def get_component(name: str, **kwargs) -> str:
        """Get a component by name with optional parameters"""
        components = {
            "code-review-header": PromptComponents._code_review_header,
            "testing-requirements": PromptComponents._testing_requirements,
            "optimization-goals": PromptComponents._optimization_goals,
            "security-checklist": PromptComponents._security_checklist,
            "documentation-standards": PromptComponents._documentation_standards,
        }
        
        if name in components:
            return components[name](**kwargs)
        return f"[Component '{name}' not found]"
    
    @staticmethod
    def _code_review_header(**kwargs) -> str:
        lang = kwargs.get('language', 'Python')
        return f"""## Code Review - {lang}

**Focus Areas:**
- Code quality and maintainability
- Performance optimization
- Security vulnerabilities
- Best practices adherence
"""
    
    @staticmethod
    def _testing_requirements(**kwargs) -> str:
        coverage = kwargs.get('coverage', '80')
        return f"""## Testing Requirements

- Minimum coverage: {coverage}%
- Unit tests for all functions
- Integration tests for APIs
- Edge case handling
"""
    
    @staticmethod
    def _optimization_goals(**kwargs) -> str:
        target = kwargs.get('target', 'performance')
        return f"""## Optimization Goals

**Target:** {target}

- Profile before optimizing
- Measure improvements
- Document trade-offs
- Maintain readability
"""
    
    @staticmethod
    def _security_checklist(**kwargs) -> str:
        return """## Security Checklist

- [ ] Input validation
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] Authentication/authorization
- [ ] Secrets management
- [ ] HTTPS enforcement
"""
    
    @staticmethod
    def _documentation_standards(**kwargs) -> str:
        style = kwargs.get('style', 'Google')
        return f"""## Documentation Standards

**Style:** {style}

- Docstrings for all public functions
- Type hints
- Usage examples
- API documentation
"""

if __name__ == "__main__":
    print(PromptComponents.get_component("code-review-header", language="TypeScript"))
    print(PromptComponents.get_component("testing-requirements", coverage="90"))
