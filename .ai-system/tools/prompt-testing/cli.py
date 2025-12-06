"""CLI for prompt testing"""
from validator import PromptValidator
from tester import PromptTester
from regression import RegressionTester
from pathlib import Path
import sys as system
import json

def main():
    if len(system.argv) < 2:
        print("Usage: python cli.py [validate|test|benchmark|regression] [args]")
        print("\nCommands:")
        print("  validate <prompt>     - Validate single prompt")
        print("  validate --all        - Validate all prompts")
        print("  benchmark <prompt>    - Benchmark prompt performance")
        print("  regression --check    - Check for regressions")
        print("  regression --save     - Save current as baseline")
        return
    
    command = system.argv[1]
    
    if command == "validate":
        validator = PromptValidator()
        
        if len(system.argv) > 2 and system.argv[2] == "--all":
            results = validator.validate_all()
            valid = sum(1 for r in results if r['valid'])
            avg_score = sum(r['score'] for r in results) / len(results)
            
            print(f"\n[VALIDATE] All Prompts")
            print(f"  Total: {len(results)}")
            print(f"  Valid: {valid} ({valid/len(results)*100:.0f}%)")
            print(f"  Avg Score: {avg_score:.2f}")
            
            # Show issues
            issues = [r for r in results if not r['valid']]
            if issues:
                print(f"\n  Issues Found:")
                for r in issues[:5]:
                    print(f"    {Path(r['path']).name}: {', '.join(r['issues'])}")
        
        elif len(system.argv) > 2:
            prompt = system.argv[2]
            result = validator.validate(prompt)
            
            print(f"\n[VALIDATE] {Path(result['path']).name}")
            print(f"  Score: {result['score']:.2f}")
            print(f"  Valid: {result['valid']}")
            print(f"  Metrics:")
            for k, v in result['metrics'].items():
                print(f"    {k}: {v}")
            if result['issues']:
                print(f"  Issues:")
                for issue in result['issues']:
                    print(f"    - {issue}")
    
    elif command == "benchmark":
        if len(system.argv) < 3:
            print("Error: Provide prompt name")
            return
        
        prompt = system.argv[2]
        iterations = int(system.argv[3]) if len(system.argv) > 3 else 10
        
        tester = PromptTester()
        result = tester.benchmark(prompt, iterations)
        
        print(f"\n[BENCHMARK] {result['prompt']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Avg: {result['avg_duration']*1000:.1f}ms")
        print(f"  Min: {result['min_duration']*1000:.1f}ms")
        print(f"  Max: {result['max_duration']*1000:.1f}ms")
    
    elif command == "regression":
        regression = RegressionTester()
        
        if len(system.argv) > 2 and system.argv[2] == "--save":
            validator = PromptValidator()
            results = validator.validate_all()
            
            baseline = {}
            for r in results:
                name = Path(r['path']).stem
                baseline[name] = {
                    'score': r['score'],
                    'hash': regression.compute_hash(Path(r['path']))
                }
            
            regression.save_baseline(baseline)
            print(f"\n[REGRESSION] Saved baseline for {len(baseline)} prompts")
        
        elif len(system.argv) > 2 and system.argv[2] == "--check":
            validator = PromptValidator()
            results = validator.validate_all()
            
            regressions = []
            for r in results:
                name = Path(r['path']).stem
                check = regression.check_regression(name, r['score'])
                if check['regression']:
                    regressions.append(check)
            
            print(f"\n[REGRESSION] Check Results")
            print(f"  Total: {len(results)}")
            print(f"  Regressions: {len(regressions)}")
            
            if regressions:
                print(f"\n  Regressions Found:")
                for reg in regressions[:10]:
                    print(f"    {reg['prompt']}: {reg['delta']:+.2f}")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
