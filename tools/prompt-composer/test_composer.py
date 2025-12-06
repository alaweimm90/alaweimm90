"""Test prompt composition"""
from composer import PromptComposer
from components import PromptComponents

def test_composition():
    print("\n[TEST] Prompt Composition")
    
    composer = PromptComposer()
    
    # Test 1: Variable substitution
    print("\n  Test 1: Variable Substitution")
    template = "Project: {{name}}, Language: {{lang}}"
    result = composer.compose(template, {"name": "TestApp", "lang": "Python"})
    print(f"    Result: {result}")
    
    # Test 2: Component usage
    print("\n  Test 2: Component Usage")
    template = "{{component:code-review-header|language=TypeScript}}"
    result = composer.compose(template)
    print(f"    Result:\n{result}")
    
    # Test 3: Multiple components
    print("\n  Test 3: Multiple Components")
    template = """{{component:testing-requirements|coverage=90}}

{{component:security-checklist}}"""
    result = composer.compose(template)
    print(f"    Result:\n{result[:150]}...")
    
    # Test 4: Full template
    print("\n  Test 4: Full Template Composition")
    template = composer.load_template("templates/fullstack-workflow.md")
    result = composer.compose(template, {
        "project_name": "MyApp",
        "tech_stack": "React + FastAPI",
        "backend_lang": "Python",
        "frontend_lang": "TypeScript",
        "coverage_target": "85"
    })
    print(f"    Composed {len(result)} characters")
    print(f"    Preview:\n{result[:200]}...")
    
    print("\n[OK] Composition test complete")

if __name__ == "__main__":
    test_composition()
