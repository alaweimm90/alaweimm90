#!/usr/bin/env python
"""
HELIOS Basic Usage Example

This example demonstrates:
1. Generating hypotheses for a research topic
2. Validating hypotheses with the Turing suite
3. Filtering for strong hypotheses
"""

def main():
    """Run basic HELIOS workflow."""
    print("=" * 60)
    print("HELIOS: Basic Usage Example")
    print("=" * 60)
    print()

    # Example 1: Generate Hypotheses
    print("[1/3] Generating hypotheses...")
    print("-" * 60)

    try:
        from helios import HypothesisGenerator

        generator = HypothesisGenerator(num_hypotheses=3)
        hypotheses = generator.generate("machine learning optimization")

        print(f"✓ Generated {len(hypotheses)} hypotheses")
        for i, h in enumerate(hypotheses, 1):
            print(f"  {i}. {h.get('text', 'Hypothesis')[:80]}...")
    except Exception as e:
        print(f"Note: Full generation requires LLM API keys. Error: {type(e).__name__}")
        hypotheses = [
            {
                "id": "h1",
                "text": "Adaptive learning rates improve convergence speed",
                "domain": "ml",
            },
            {
                "id": "h2",
                "text": "Neural networks with residual connections enable deeper architectures",
                "domain": "ml",
            },
        ]
        print(f"Using mock hypotheses: {len(hypotheses)} examples")
        for i, h in enumerate(hypotheses, 1):
            print(f"  {i}. {h['text'][:80]}...")

    print()

    # Example 2: Validate Hypotheses
    print("[2/3] Validating hypotheses...")
    print("-" * 60)

    try:
        from helios import TuringValidator

        validator = TuringValidator()
        results = validator.validate(hypotheses)

        print(f"✓ Validated {len(results)} hypotheses")
        for i, r in enumerate(results, 1):
            score = r.get("overall_score", 0)
            print(f"  {i}. Score: {score:.1f}/100")
    except Exception as e:
        print(f"Note: Validation requires full setup. Error: {type(e).__name__}")
        results = [
            {"hypothesis_id": "h1", "overall_score": 75.5, "weaknesses": []},
            {"hypothesis_id": "h2", "overall_score": 82.3, "weaknesses": []},
        ]
        print(f"Using mock validation results:")
        for i, r in enumerate(results, 1):
            score = r.get("overall_score", 0)
            print(f"  {i}. Score: {score:.1f}/100")

    print()

    # Example 3: Filter Strong Hypotheses
    print("[3/3] Filtering strong hypotheses (score > 70)...")
    print("-" * 60)

    strong = []
    if results:
        for h, r in zip(hypotheses, results):
            if r.get("overall_score", 0) > 70:
                strong.append((h, r))

    print(f"✓ Found {len(strong)} strong hypothesis(es)")
    for i, (h, r) in enumerate(strong, 1):
        score = r.get("overall_score", 0)
        text = h.get("text", "")[:60]
        print(f"  {i}. [{score:.1f}] {text}...")

    print()
    print("=" * 60)
    print("✓ Basic workflow complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. See helios/examples/ for more examples")
    print("  2. Read helios/docs/GETTING_STARTED.md for full setup")
    print("  3. Check helios/docs/API.md for API reference")


if __name__ == "__main__":
    main()
