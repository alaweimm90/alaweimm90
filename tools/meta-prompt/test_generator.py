#!/usr/bin/env python3
"""Test suite for meta-prompt generator."""

import pytest
from pathlib import Path
from generator import PromptGenerator, PromptSpec

def test_parse_optimization_requirement():
    """Test parsing optimization requirements."""
    gen = PromptGenerator(Path("templates"))
    spec = gen._parse_requirement("optimize database queries for better performance")
    
    assert "optimize" in spec.title.lower()
    assert spec.domain == "optimization"
    assert len(spec.use_cases) > 0

def test_parse_testing_requirement():
    """Test parsing testing requirements."""
    gen = PromptGenerator(Path("templates"))
    spec = gen._parse_requirement("write comprehensive tests for REST API")
    
    assert "test" in spec.title.lower()
    assert spec.domain == "testing"

def test_generate_prompt():
    """Test full prompt generation."""
    gen = PromptGenerator(Path("templates"))
    prompt = gen.generate("optimize memory usage in Python")
    
    assert "# " in prompt
    assert "## Purpose" in prompt
    assert "## When to Use" in prompt
    assert len(prompt) > 100

def test_quality_score():
    """Test quality scoring."""
    gen = PromptGenerator(Path("templates"))
    
    # Good prompt
    good_prompt = """# Test
## Purpose
Test purpose
## When to Use
- Use case 1
## Prompt
```
code
```
## Examples
### Input
test
### Output
result
## Success Criteria
- [ ] Done
## Related Prompts
- prompt1
"""
    score = gen._calculate_quality_score(good_prompt)
    assert score > 0.8
    
    # Poor prompt
    poor_prompt = "# Test\nShort prompt"
    score = gen._calculate_quality_score(poor_prompt)
    assert score < 0.5

def test_domain_detection():
    """Test domain detection."""
    gen = PromptGenerator(Path("templates"))
    
    assert gen._extract_domain("optimize performance") == "optimization"
    assert gen._extract_domain("write tests") == "testing"
    assert gen._extract_domain("design system") == "architecture"
    assert gen._extract_domain("debug issue") == "general"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
