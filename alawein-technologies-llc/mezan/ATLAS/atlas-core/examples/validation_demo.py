"""Demo script for MEZAN Validation & A/B Testing Framework.

This demonstrates the key capabilities of the validation system.

Author: Meshal Alawein
Date: 2025-11-18
"""

print("=" * 60)
print("MEZAN VALIDATION FRAMEWORK DEMO")
print("=" * 60)

# Note: This demo requires the validation dependencies to be installed:
# pip install -r requirements-validation.txt

try:
    from atlas_core.validation import StatisticalValidator, quick_t_test
    from atlas_core.ab_testing import (
        ABTestingFramework, ExperimentConfig, Variant, Metric, MetricType
    )
    from atlas_core.formal_verification import (
        WorkflowVerifier, WorkflowState, WorkflowTransition, check_workflow_safety
    )
    from atlas_core.cross_validation import (
        CrossValidator, BaselineSystem, BenchmarkCase
    )
    import numpy as np

    print("\n✅ All validation modules loaded successfully!\n")

    # 1. STATISTICAL VALIDATION DEMO
    print("1. STATISTICAL VALIDATION")
    print("-" * 40)

    validator = StatisticalValidator(confidence_level=0.95)

    # Generate sample data
    np.random.seed(42)
    baseline_performance = np.random.normal(100, 15, 50)  # baseline: mean=100, std=15
    improved_performance = np.random.normal(110, 15, 50)  # improved: mean=110, std=15

    # Perform t-test
    result = validator.t_test(improved_performance, baseline_performance)

    print(f"T-Test Results:")
    print(f"  - P-value: {result.p_value:.4f}")
    print(f"  - Significant improvement: {result.is_significant}")
    print(f"  - Effect size (Cohen's d): {result.effect_size:.3f}")
    print(f"  - Mean difference: {result.additional_metrics['mean_difference']:.2f}")

    # 2. A/B TESTING DEMO
    print("\n2. A/B TESTING")
    print("-" * 40)

    # Create A/B test framework
    ab_framework = ABTestingFramework()

    # Define experiment
    config = ExperimentConfig(
        name="Homepage Redesign Test",
        description="Testing new homepage design impact on conversions",
        variants=[
            Variant("control", "Original homepage", weight=0.5),
            Variant("treatment", "New design", weight=0.5)
        ],
        metrics=[
            Metric("conversion", MetricType.CONVERSION,
                  "Conversion rate", is_primary=True),
            Metric("time_on_page", MetricType.DURATION,
                  "Time spent on page", is_primary=False)
        ],
        confidence_level=0.95,
        enable_bayesian=True
    )

    experiment = ab_framework.create_experiment(config)
    experiment.start()

    # Simulate user interactions
    np.random.seed(42)
    for i in range(200):
        user_id = f"user_{i:04d}"
        variant = experiment.assign_variant(user_id)

        # Simulate different conversion rates
        if variant == "control":
            converted = np.random.random() < 0.10  # 10% conversion rate
            time_on_page = np.random.exponential(30)  # avg 30 seconds
        else:
            converted = np.random.random() < 0.15  # 15% conversion rate (50% lift!)
            time_on_page = np.random.exponential(45)  # avg 45 seconds

        experiment.track_metric(user_id, "conversion", 1 if converted else 0)
        experiment.track_metric(user_id, "time_on_page", time_on_page)

    # Analyze results
    results = experiment.analyze_results()

    print(f"A/B Test Results:")
    print(f"  Experiment: {config.name}")
    print(f"  Sample size: {sum(experiment.variant_counts.values())} users")

    conv_results = results['metrics']['conversion']
    print(f"\n  Conversion Rate:")
    print(f"    - Control: {conv_results['control']['mean']*100:.1f}%")
    print(f"    - Treatment: {conv_results['treatment']['mean']*100:.1f}%")
    print(f"    - Lift: {conv_results['frequentist']['lift']:.1f}%")
    print(f"    - P-value: {conv_results['frequentist']['p_value']:.4f}")
    print(f"    - Significant: {conv_results['frequentist']['is_significant']}")

    if 'bayesian' in conv_results:
        print(f"\n  Bayesian Analysis:")
        print(f"    - P(Treatment > Control): {conv_results['bayesian']['prob_treatment_better']:.1%}")

    print(f"\n  Recommendation: {results['recommendation']['action']}")

    # 3. WORKFLOW VERIFICATION DEMO
    print("\n3. FORMAL WORKFLOW VERIFICATION")
    print("-" * 40)

    verifier = WorkflowVerifier()

    # Define a simple workflow
    states = [
        WorkflowState("init", "Initialize", is_initial=True),
        WorkflowState("fetch_data", "Fetch Data"),
        WorkflowState("process", "Process Data"),
        WorkflowState("validate", "Validate Results"),
        WorkflowState("store", "Store Results"),
        WorkflowState("complete", "Complete", is_final=True),
        WorkflowState("error", "Error Handler", is_error=True)
    ]

    for state in states:
        verifier.add_state(state)

    # Define transitions
    transitions = [
        WorkflowTransition("init", "fetch_data", "start"),
        WorkflowTransition("fetch_data", "process", "data_ready"),
        WorkflowTransition("fetch_data", "error", "fetch_failed"),
        WorkflowTransition("process", "validate", "processed"),
        WorkflowTransition("process", "error", "process_failed"),
        WorkflowTransition("validate", "store", "valid"),
        WorkflowTransition("validate", "error", "invalid"),
        WorkflowTransition("store", "complete", "stored"),
        WorkflowTransition("error", "complete", "handled")
    ]

    for transition in transitions:
        verifier.add_transition(transition)

    # Run verification checks
    print("Workflow Verification Results:")

    # Check for deadlocks
    deadlock_result = verifier.detect_deadlocks()
    print(f"  - Deadlock Detection: {deadlock_result.status.value}")

    # Check reachability
    reachability_result = verifier.verify_reachability()
    print(f"  - All States Reachable: {reachability_result.status.value}")

    # Check termination
    termination_result = verifier.verify_termination()
    print(f"  - Can Terminate Properly: {termination_result.status.value}")

    # 4. CROSS-VALIDATION DEMO
    print("\n4. CROSS-VALIDATION")
    print("-" * 40)

    # Create mock MEZAN executor
    def mezan_executor(input_data):
        # Simulate MEZAN processing (fast)
        import time
        time.sleep(0.001)
        return sum(input_data.get('values', []))

    # Create baseline executor
    def baseline_executor(input_data):
        # Simulate baseline (slower)
        import time
        time.sleep(0.005)
        return sum(input_data.get('values', []))

    cross_validator = CrossValidator(mezan_executor)

    # Add baseline system
    cross_validator.add_baseline(BaselineSystem(
        name="SimpleBaseline",
        version="1.0",
        executor=baseline_executor
    ))

    # Add benchmarks
    for i in range(3):
        cross_validator.add_benchmark(BenchmarkCase(
            id=f"benchmark_{i}",
            name=f"Test Case {i}",
            category="arithmetic",
            input_data={'values': list(range(i*10, (i+1)*10))},
            expected_output=sum(range(i*10, (i+1)*10))
        ))

    print("Cross-Validation Setup:")
    print(f"  - MEZAN vs SimpleBaseline")
    print(f"  - Benchmarks: {len(cross_validator.benchmarks)}")

    # Note: Full comparison would be run with:
    # results = cross_validator.compare_against_baseline("SimpleBaseline")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Summary of capabilities
    print("\n📊 VALIDATION FRAMEWORK CAPABILITIES:")
    print("  ✓ Statistical significance testing (t-test, chi-square, ANOVA)")
    print("  ✓ Effect size calculation and power analysis")
    print("  ✓ Multiple comparison correction (Bonferroni, FDR)")
    print("  ✓ Confidence intervals and p-value calculation")

    print("\n🔬 A/B TESTING CAPABILITIES:")
    print("  ✓ Multiple traffic splitting strategies")
    print("  ✓ Real-time metrics collection")
    print("  ✓ Frequentist and Bayesian analysis")
    print("  ✓ Early stopping with sequential testing")
    print("  ✓ Automatic recommendation generation")

    print("\n🔍 FORMAL VERIFICATION CAPABILITIES:")
    print("  ✓ Workflow invariant checking")
    print("  ✓ Deadlock detection via graph analysis")
    print("  ✓ State reachability verification")
    print("  ✓ Resource allocation validation")
    print("  ✓ Temporal logic checking (CTL/LTL)")

    print("\n⚖️ CROSS-VALIDATION CAPABILITIES:")
    print("  ✓ K-fold cross-validation for solver selection")
    print("  ✓ Head-to-head system comparison")
    print("  ✓ Performance regression detection")
    print("  ✓ Benchmark replay and reproducibility")
    print("  ✓ Statistical comparison with multiple baselines")

except ImportError as e:
    print(f"\n⚠️ Warning: Some dependencies are missing: {e}")
    print("\nTo run this demo, install the validation dependencies:")
    print("  pip install -r requirements-validation.txt")
    print("\nThe framework modules have been created and are ready to use")
    print("once the dependencies are installed.")

except Exception as e:
    print(f"\n❌ Error during demo: {e}")
    import traceback
    traceback.print_exc()