"""Test the testing framework"""
from validator import PromptValidator
from tester import PromptTester
from regression import RegressionTester
from pathlib import Path

def test_framework():
    print("\n[TEST] Prompt Testing Framework")
    
    # Test 1: Validation
    print("\n  Test 1: Prompt Validation")
    validator = PromptValidator()
    
    # Validate a few prompts
    test_prompts = [
        "superprompts/optimization-framework.md",
        "code-review/agentic-code-review.md",
        "debugging/debugger.md"
    ]
    
    for prompt in test_prompts:
        try:
            result = validator.validate(prompt)
            print(f"    {Path(prompt).stem}: {result['score']:.2f} ({'PASS' if result['valid'] else 'FAIL'})")
        except Exception as e:
            print(f"    {Path(prompt).stem}: ERROR - {e}")
    
    # Test 2: Batch validation
    print("\n  Test 2: Batch Validation")
    all_results = validator.validate_all()
    valid_count = sum(1 for r in all_results if r['valid'])
    avg_score = sum(r['score'] for r in all_results) / len(all_results) if all_results else 0
    print(f"    Validated: {len(all_results)} prompts")
    print(f"    Valid: {valid_count} ({valid_count/len(all_results)*100:.0f}%)")
    print(f"    Avg Score: {avg_score:.2f}")
    
    # Test 3: Prompt Testing
    print("\n  Test 3: Prompt Testing")
    tester = PromptTester()
    tester.add_test("code-review", {"code": "test"}, {"issues": 0})
    tester.add_test("optimization", {"code": "test"}, {"score": 0.8})
    
    test_results = tester.run_tests()
    passed = sum(1 for r in test_results if r['passed'])
    print(f"    Tests: {len(test_results)}")
    print(f"    Passed: {passed}/{len(test_results)}")
    
    # Test 4: Benchmarking
    print("\n  Test 4: Benchmarking")
    bench = tester.benchmark("test-prompt", iterations=5)
    print(f"    Prompt: {bench['prompt']}")
    print(f"    Avg: {bench['avg_duration']*1000:.1f}ms")
    
    # Test 5: Regression Testing
    print("\n  Test 5: Regression Testing")
    regression = RegressionTester()
    
    # Save baseline
    baseline = {
        "code-review": {"score": 0.85, "hash": "abc123"},
        "optimization": {"score": 0.90, "hash": "def456"}
    }
    regression.save_baseline(baseline)
    
    # Check regression
    check = regression.check_regression("code-review", 0.80)
    print(f"    {check['prompt']}: {check['delta']:+.2f} ({'REGRESSION' if check['regression'] else 'OK'})")
    
    print("\n[OK] Testing framework complete")

if __name__ == "__main__":
    test_framework()
