"""
Example workflow for Neuroscience Research Module.

Demonstrates fMRI/EEG analysis, neural circuit modeling,
cognitive testing, and hypothesis validation.
"""

import asyncio
from datetime import datetime
import numpy as np

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuroscience import (
    NeuroscienceHypothesis,
    NeuroscienceProtocol,
    FMRIData,
    EEGData,
    BrainRegion,
    NeuralCircuit,
    CognitiveModel,
    ConnectomeGraph,
    BCIExperiment,
    CognitiveModelType,
    BrainRegionID,
    EEGChannel,
)


async def example_memory_circuit():
    """Example: Memory consolidation circuit hypothesis."""
    print("=" * 80)
    print("EXAMPLE 1: Hippocampal-Cortical Memory Consolidation Circuit")
    print("=" * 80)

    # Define brain regions involved
    hippocampus = BrainRegion(
        name="Hippocampus",
        id=BrainRegionID.HIPPOCAMPUS,
        coordinates=(26.0, -20.0, -10.0),
        volume=3500.0,
        hemisphere="bilateral",
        neurotransmitters=["glutamate", "GABA", "acetylcholine"]
    )

    pfc = BrainRegion(
        name="Prefrontal Cortex",
        id=BrainRegionID.PFC,
        coordinates=(0.0, 45.0, 30.0),
        volume=10000.0,
        hemisphere="bilateral",
        neurotransmitters=["glutamate", "GABA", "dopamine"]
    )

    # Create memory circuit
    memory_circuit = NeuralCircuit(
        name="hippocampal_pfc_memory",
        regions=[hippocampus, pfc],
        connections=[
            ("Hippocampus", "Prefrontal Cortex", 0.75),
            ("Prefrontal Cortex", "Hippocampus", 0.60)
        ],
        neuron_types={"pyramidal": 70, "interneuron": 25, "astrocyte": 5},
        oscillation_frequency=6.0,  # Theta
        propagation_delay=15.0,
        plasticity_type="LTP",
        function="memory consolidation",
        cognitive_role="episodic memory transfer",
        disorders=["Alzheimer's", "PTSD"]
    )

    # Create hypothesis
    hypothesis = NeuroscienceHypothesis(
        hypothesis_id="neuro-2024-001",
        title="Theta-Gamma Coupling Mediates Memory Consolidation",
        claim="Theta-gamma phase-amplitude coupling between hippocampus and PFC predicts successful memory consolidation during sleep",

        neural_circuits=[memory_circuit],
        brain_regions=[hippocampus, pfc],

        predicted_activity={
            "hippocampus": "theta_oscillations",
            "pfc": "gamma_bursts",
            "coupling": "phase_amplitude"
        },

        predicted_behavior={
            "memory_performance": "enhanced",
            "consolidation_rate": 0.85
        },

        theoretical_basis=[
            "Theta oscillations coordinate memory replay",
            "Gamma bursts encode memory content",
            "Sleep spindles facilitate transfer",
            "Synaptic downscaling during NREM"
        ],

        confidence_score=0.75,
        impact_score=8.5,
        novelty_score=6.5,

        clinical_applications=["Memory enhancement therapy", "PTSD treatment"],
        disorders_addressed=["Alzheimer's disease", "Mild cognitive impairment"]
    )

    # Initialize protocol
    protocol = NeuroscienceProtocol()

    # Validate hypothesis
    print("\nValidating memory circuit hypothesis...")
    results = await protocol.validate_hypothesis(hypothesis, "comprehensive")

    print("\n" + "=" * 40)
    print("VALIDATION RESULTS")
    print("=" * 40)

    print(f"\nOverall Score: {results['scores']['overall']:.2f}")
    print(f"Recommendation: {results['recommendation']}")

    if "circuit_analysis" in results["stages"]:
        circuit_results = results["stages"]["circuit_analysis"]
        print(f"\nCircuit Analysis:")
        print(f"  Plausible: {circuit_results['plausible']}")
        for circuit_name, score in circuit_results["circuit_scores"].items():
            print(f"  {circuit_name}: {score:.2f}")

    return results


async def example_bci_experiment():
    """Example: Motor imagery BCI for stroke rehabilitation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: BCI for Post-Stroke Motor Recovery")
    print("=" * 80)

    # Define motor cortex regions
    m1_left = BrainRegion(
        name="Left M1",
        id=BrainRegionID.M1,
        coordinates=(-38.0, -25.0, 50.0),
        hemisphere="left"
    )

    m1_right = BrainRegion(
        name="Right M1",
        id=BrainRegionID.M1,
        coordinates=(38.0, -25.0, 50.0),
        hemisphere="right"
    )

    # Create hypothesis
    hypothesis = NeuroscienceHypothesis(
        hypothesis_id="neuro-2024-002",
        title="Motor Imagery BCI Enhances Post-Stroke Recovery",
        claim="BCI-driven motor imagery training increases ipsilesional M1 activation and improves motor function",

        brain_regions=[m1_left, m1_right],

        predicted_activity={
            "motor_imagery": "mu_suppression",
            "feedback": "error_related_negativity",
            "learning": "increased_coherence"
        },

        proposed_experiments=[{
            "design": "randomized_controlled_trial",
            "variables": {
                "independent": ["BCI training", "standard therapy"],
                "dependent": ["motor_function", "cortical_activation"]
            },
            "controls": ["sham_BCI"],
            "analysis": {
                "primary": "mixed_anova",
                "correction": "FDR"
            }
        }],

        required_sample_size=40,
        power_analysis={"statistical_power": 0.85},

        theoretical_basis=[
            "Hebbian plasticity through BCI feedback",
            "Motor imagery activates motor networks",
            "Closed-loop stimulation enhances learning"
        ],

        confidence_score=0.72,
        impact_score=9.0,
        novelty_score=5.5,

        clinical_applications=["Stroke rehabilitation", "BCI therapy"],
        disorders_addressed=["Stroke", "Spinal cord injury"],
        therapeutic_implications="BCI can facilitate motor recovery through neural plasticity",

        requires_irb=True,
        ethical_considerations=["Informed consent for vulnerable population", "Risk of frustration"]
    )

    protocol = NeuroscienceProtocol()

    # Generate BCI protocol
    print("\nGenerating BCI experimental protocol...")
    bci_experiment = await protocol.generate_bci_protocol(hypothesis, "motor_imagery")

    print(f"\nBCI Experiment Design:")
    print(f"  Paradigm: {bci_experiment.paradigm.value}")
    print(f"  Channels: {', '.join(bci_experiment.channels[:5])}...")
    print(f"  Sampling Rate: {bci_experiment.sampling_rate} Hz")
    print(f"  Trials: {bci_experiment.num_trials}")
    print(f"  Estimated Accuracy: {bci_experiment.accuracy:.2%}")
    print(f"  ITR: {bci_experiment.information_transfer_rate:.1f} bits/min")

    # Validate full hypothesis
    validation = await protocol.validate_hypothesis(hypothesis, "standard")

    print(f"\nValidation Score: {validation['scores']['overall']:.2f}")
    print(f"Experimental Design Score: {validation['scores']['experimental_design']:.2f}")

    return validation


async def example_cognitive_model():
    """Example: ACT-R model of working memory."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Cognitive Model of Working Memory Load")
    print("=" * 80)

    # Create cognitive model
    cognitive_model = CognitiveModel(
        name="WM_Load_ACT-R",
        type=CognitiveModelType.ACT_R,

        parameters={
            "retrieval_threshold": -0.5,
            "latency_factor": 0.5,
            "decay_rate": 0.5,
            "chunk_capacity": 7,
            "activation_noise": 0.25
        },

        architecture={
            "modules": ["declarative", "procedural", "goal", "visual"],
            "buffers": ["retrieval", "goal", "visual"],
            "productions": 25
        },

        task_domain="working_memory",
        input_format="stimulus_sequence",
        output_format="response_accuracy",

        accuracy=0.82,
        reaction_time_correlation=0.75,
        neural_fit=0.68,

        implementation_language="python",
        libraries=["pyactr", "numpy"]
    )

    # Create hypothesis with model
    hypothesis = NeuroscienceHypothesis(
        hypothesis_id="neuro-2024-003",
        title="Capacity-Limited Working Memory Model",
        claim="Working memory capacity limitations arise from interference between chunk activations in ACT-R",

        cognitive_model=cognitive_model,

        model_predictions={
            "capacity_limit": 7,
            "serial_position": "u_shaped_curve",
            "load_effect": "linear_decline"
        },

        brain_regions=[
            BrainRegion(name="DLPFC", hemisphere="bilateral"),
            BrainRegion(name="Parietal", hemisphere="bilateral")
        ],

        theoretical_basis=[
            "Chunks compete for activation",
            "Decay prevents interference",
            "Rehearsal maintains activation"
        ],

        confidence_score=0.78,
        impact_score=7.5,
        novelty_score=4.0
    )

    protocol = NeuroscienceProtocol()

    print("\nValidating cognitive model...")
    results = await protocol.validate_hypothesis(hypothesis, "standard")

    if "cognitive_model" in results["stages"]:
        model_results = results["stages"]["cognitive_model"]
        print(f"\nModel Performance:")
        print(f"  Accuracy: {model_results['performance']['accuracy']:.2f}")
        print(f"  RT Correlation: {model_results['performance']['reaction_time_fit']:.2f}")
        print(f"  Neural Alignment: {model_results['performance']['neural_alignment']:.2f}")

        if "complexity_analysis" in model_results:
            complexity = model_results["complexity_analysis"]
            print(f"\nComplexity Analysis:")
            print(f"  Parameters: {complexity['num_parameters']}")
            print(f"  Overfitting Risk: {complexity['overfitting_risk']:.2f}")

    print(f"\nOverall Validation Score: {results['scores']['overall']:.2f}")

    return results


async def example_connectome_analysis():
    """Example: Connectome analysis for autism spectrum disorder."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Altered Connectivity in Autism Spectrum Disorder")
    print("=" * 80)

    # Create brain regions for connectome
    regions = [
        BrainRegion(name="mPFC", coordinates=(0, 50, 20)),
        BrainRegion(name="PCC", coordinates=(0, -55, 25)),
        BrainRegion(name="Angular_L", coordinates=(-45, -65, 35)),
        BrainRegion(name="Angular_R", coordinates=(45, -65, 35)),
        BrainRegion(name="STS_L", coordinates=(-55, -45, 15)),
        BrainRegion(name="STS_R", coordinates=(55, -45, 15))
    ]

    # Create connectome
    edges = [
        (0, 1, 0.85),  # mPFC-PCC (strong DMN connection)
        (0, 2, 0.65),  # mPFC-Angular_L
        (0, 3, 0.62),  # mPFC-Angular_R
        (1, 2, 0.71),  # PCC-Angular_L
        (1, 3, 0.73),  # PCC-Angular_R
        (2, 3, 0.45),  # Angular_L-Angular_R
        (4, 5, 0.55),  # STS_L-STS_R (reduced in ASD)
        (0, 4, 0.25),  # mPFC-STS_L (reduced)
        (0, 5, 0.22),  # mPFC-STS_R (reduced)
    ]

    connectome = ConnectomeGraph(
        subject_id="asd_group_average",
        modality="functional",
        nodes=regions,
        edges=edges,
        weighted=True,
        directed=False
    )

    # Create hypothesis
    hypothesis = NeuroscienceHypothesis(
        hypothesis_id="neuro-2024-004",
        title="Reduced DMN-Social Brain Connectivity in ASD",
        claim="ASD is characterized by reduced connectivity between default mode and social processing networks",

        predicted_connectivity=connectome,
        brain_regions=regions,

        predicted_activity={
            "dmn_coherence": "normal",
            "social_network": "reduced",
            "cross_network": "decreased"
        },

        predicted_behavior={
            "social_cognition": "impaired",
            "self_referential": "altered",
            "theory_of_mind": "reduced"
        },

        theoretical_basis=[
            "DMN supports self-referential processing",
            "STS critical for social perception",
            "Reduced integration impairs social cognition"
        ],

        confidence_score=0.82,
        impact_score=8.0,
        novelty_score=5.5,

        clinical_applications=["ASD biomarker", "Treatment target"],
        disorders_addressed=["Autism Spectrum Disorder"]
    )

    protocol = NeuroscienceProtocol()

    print("\nValidating connectivity hypothesis...")
    results = await protocol.validate_hypothesis(hypothesis, "comprehensive")

    if "connectivity" in results["stages"]:
        conn_results = results["stages"]["connectivity"]
        print(f"\nConnectome Analysis:")
        print(f"  Valid: {conn_results['valid']}")

        if "metrics" in conn_results:
            metrics = conn_results["metrics"]
            print(f"  Small-worldness: {metrics.get('small_worldness', 0):.2f}")
            print(f"  Modularity: {metrics.get('modularity', 0):.2f}")

        if "network_properties" in conn_results:
            props = conn_results["network_properties"]
            print(f"  Modular: {props.get('modular', False)}")
            print(f"  Hub consistency: {props.get('hub_consistency', 0):.2f}")

    print(f"\nOverall Score: {results['scores']['overall']:.2f}")

    return results


async def main():
    """Run all neuroscience examples."""
    print("=" * 80)
    print("NEUROSCIENCE RESEARCH MODULE - EXAMPLE WORKFLOWS")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")

    results = []

    # Example 1: Memory circuit
    result1 = await example_memory_circuit()
    results.append(("Memory Circuit", result1))

    # Example 2: BCI experiment
    result2 = await example_bci_experiment()
    results.append(("BCI Experiment", result2))

    # Example 3: Cognitive model
    result3 = await example_cognitive_model()
    results.append(("Cognitive Model", result3))

    # Example 4: Connectome analysis
    result4 = await example_connectome_analysis()
    results.append(("Connectome Analysis", result4))

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)

    for name, result in results:
        score = result["scores"]["overall"]
        rec = result["recommendation"]
        print(f"\n{name}:")
        print(f"  Score: {score:.2f}")
        print(f"  Recommendation: {rec}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())