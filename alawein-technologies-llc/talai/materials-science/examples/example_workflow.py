"""
Example workflow for Materials Science Research Module.

Demonstrates crystal structure prediction, property analysis,
synthesis planning, and hypothesis validation.
"""

import asyncio
from datetime import datetime
import json

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.materials_science import (
    MaterialsHypothesis,
    CrystalStructure,
    MaterialProperties,
    MaterialsScienceProtocol,
    StructurePredictor,
    PropertyPredictor,
    SpaceGroup,
    AtomicPosition,
    MechanicalProperties,
    ElectricalProperties,
    ThermalProperties,
)


async def example_perovskite_hypothesis():
    """Example: Novel perovskite material for solar cells."""
    print("=" * 80)
    print("EXAMPLE 1: Novel Perovskite Solar Cell Material")
    print("=" * 80)

    # Create hypothesis for novel perovskite
    hypothesis = MaterialsHypothesis(
        hypothesis_id="mat-2024-001",
        title="Lead-Free Perovskite CsSnI3 for High-Efficiency Solar Cells",
        claim="CsSnI3 with optimized crystal structure will achieve >20% solar efficiency",

        predicted_structure=CrystalStructure(
            formula="CsSnI3",
            lattice_type="orthorhombic",
            space_group=SpaceGroup(
                number=62,
                symbol="Pnma",
                point_group="mmm",
                crystal_system="orthorhombic"
            ),
            lattice_parameters={
                "a": 8.69, "b": 12.38, "c": 8.64,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
            },
            atomic_positions=[
                AtomicPosition(element="Cs", x=0.25, y=0.25, z=0.50, occupancy=1.0),
                AtomicPosition(element="Sn", x=0.00, y=0.00, z=0.00, occupancy=1.0),
                AtomicPosition(element="I", x=0.25, y=0.00, z=0.25, occupancy=1.0),
                AtomicPosition(element="I", x=0.00, y=0.25, z=0.25, occupancy=1.0),
                AtomicPosition(element="I", x=0.25, y=0.25, z=0.00, occupancy=1.0),
            ],
            volume=929.3,
            density=4.95
        ),

        predicted_properties=MaterialProperties(
            formula="CsSnI3",
            electrical=ElectricalProperties(
                band_gap=1.3,  # Ideal for solar cells
                conductivity=1e4,
                work_function=4.7,
                carrier_concentration=1e16,
                mobility=50.0
            ),
            thermal=ThermalProperties(
                melting_point=723,
                thermal_conductivity=0.5,
                thermal_expansion=3e-5
            ),
            stability=-0.8,  # Formation energy
            synthesizability=0.75
        ),

        theoretical_basis=[
            "Goldschmidt tolerance factor of 0.89 indicates stable perovskite structure",
            "Sn2+ has similar ionic radius to Pb2+ allowing substitution",
            "Band gap tunable through halide mixing",
            "Previous studies show promise but require optimization"
        ],

        confidence_score=0.72,
        impact_score=8.5,
        novelty_score=6.0,

        related_materials=["CsPbI3", "MASnI3", "FASnI3"],
        citations=[
            "DOI: 10.1038/s41560-019-0535-2",
            "DOI: 10.1021/acsenergylett.8b01809"
        ]
    )

    # Initialize protocol
    protocol = MaterialsScienceProtocol()

    # Validate hypothesis
    print("\nValidating hypothesis...")
    validation_results = await protocol.validate_hypothesis(
        hypothesis,
        validation_level="comprehensive"
    )

    # Print results
    print("\n" + "=" * 40)
    print("VALIDATION RESULTS")
    print("=" * 40)

    print(f"\nOverall Scores:")
    for key, value in validation_results["scores"].items():
        print(f"  {key:15s}: {value:.2f}")

    print(f"\nRecommendation: {validation_results['recommendation']}")

    # Structure validation details
    if "structure" in validation_results["stages"]:
        struct_results = validation_results["stages"]["structure"]
        print(f"\nStructure Validation:")
        print(f"  Valid: {struct_results['valid']}")
        if struct_results.get("issues"):
            print(f"  Issues: {', '.join(struct_results['issues'])}")
        print(f"  Structure Type: {struct_results.get('structure_type', 'unknown')}")

    # Generate research plan
    research_plan = await protocol.generate_research_plan(hypothesis)
    print(f"\nResearch Plan:")
    print(f"  Timeline: {research_plan['timeline_months']} months")
    print(f"  Phases:")
    for phase in research_plan["phases"]:
        print(f"    - {phase['name']} ({phase['duration_months']} months)")

    return validation_results


async def example_battery_material():
    """Example: High-capacity battery cathode material."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Next-Generation Battery Cathode Material")
    print("=" * 80)

    # Create hypothesis for battery material
    hypothesis = MaterialsHypothesis(
        hypothesis_id="mat-2024-002",
        title="High-Capacity Li2MnO3-based Cathode with Enhanced Stability",
        claim="Li2MnO3 with cation doping achieves 300 mAh/g capacity with <5% fade over 500 cycles",

        predicted_structure=CrystalStructure(
            formula="Li2MnO3",
            lattice_type="monoclinic",
            space_group=SpaceGroup(
                number=12,
                symbol="C2/m",
                point_group="2/m",
                crystal_system="monoclinic"
            ),
            lattice_parameters={
                "a": 4.94, "b": 8.53, "c": 5.03,
                "alpha": 90.0, "beta": 109.5, "gamma": 90.0
            },
            atomic_positions=[
                AtomicPosition(element="Li", x=0.0, y=0.5, z=0.0, occupancy=1.0),
                AtomicPosition(element="Li", x=0.0, y=0.0, z=0.5, occupancy=1.0),
                AtomicPosition(element="Mn", x=0.0, y=0.167, z=0.0, occupancy=1.0),
                AtomicPosition(element="O", x=0.22, y=0.0, z=0.23, occupancy=1.0),
                AtomicPosition(element="O", x=0.25, y=0.32, z=0.22, occupancy=1.0),
                AtomicPosition(element="O", x=0.73, y=0.32, z=0.78, occupancy=1.0),
            ],
            volume=200.1,
            density=4.18
        ),

        predicted_properties=MaterialProperties(
            formula="Li2MnO3",
            electrical=ElectricalProperties(
                band_gap=2.0,
                conductivity=1e-4,  # Needs conductive coating
            ),
            mechanical=MechanicalProperties(
                bulk_modulus=120.0,
                youngs_modulus=180.0
            ),
            stability=-2.1,  # Good stability
            synthesizability=0.85
        ),

        theoretical_basis=[
            "Layered structure enables high Li+ mobility",
            "Mn4+ provides structural stability",
            "Oxygen redox contributes to high capacity",
            "Cation doping (Ni, Co) can improve conductivity"
        ],

        confidence_score=0.78,
        impact_score=9.0,
        novelty_score=5.5,

        related_materials=["LiMn2O4", "LiNiMnCoO2", "Li2MnSiO4"]
    )

    # Initialize protocol with specific config
    protocol = MaterialsScienceProtocol(
        config={"focus": "energy_storage"}
    )

    # Validate with standard level
    print("\nValidating battery material hypothesis...")
    results = await protocol.validate_hypothesis(
        hypothesis,
        validation_level="standard"
    )

    print("\n" + "=" * 40)
    print("BATTERY MATERIAL VALIDATION")
    print("=" * 40)

    print(f"\nValidation Summary:")
    print(f"  Overall Score: {results['scores']['overall']:.2f}")
    print(f"  Recommendation: {results['recommendation']}")

    # Check synthesis feasibility
    if "synthesis" in results["stages"]:
        synth = results["stages"]["synthesis"]
        print(f"\nSynthesis Analysis:")
        print(f"  Feasible: {synth.get('feasible', False)}")
        print(f"  Score: {synth.get('score', 0):.2f}")
        if synth.get("alternatives"):
            print(f"  Alternative routes available: {len(synth['alternatives'])}")

    return results


async def example_quantum_material():
    """Example: Topological insulator for quantum computing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Topological Insulator for Quantum Applications")
    print("=" * 80)

    # Create hypothesis for quantum material
    hypothesis = MaterialsHypothesis(
        hypothesis_id="mat-2024-003",
        title="Room-Temperature Topological Insulator Bi2Se3 with Enhanced Surface States",
        claim="Bi2Se3 thin films exhibit robust topological surface states at 300K with mobility >10,000 cm²/V·s",

        predicted_structure=CrystalStructure(
            formula="Bi2Se3",
            lattice_type="trigonal",
            space_group=SpaceGroup(
                number=166,
                symbol="R-3m",
                point_group="-3m",
                crystal_system="trigonal"
            ),
            lattice_parameters={
                "a": 4.14, "b": 4.14, "c": 28.64,
                "alpha": 90.0, "beta": 90.0, "gamma": 120.0
            },
            atomic_positions=[
                AtomicPosition(element="Bi", x=0.0, y=0.0, z=0.399, occupancy=1.0),
                AtomicPosition(element="Bi", x=0.0, y=0.0, z=0.601, occupancy=1.0),
                AtomicPosition(element="Se", x=0.0, y=0.0, z=0.0, occupancy=1.0),
                AtomicPosition(element="Se", x=0.0, y=0.0, z=0.211, occupancy=1.0),
                AtomicPosition(element="Se", x=0.0, y=0.0, z=0.789, occupancy=1.0),
            ],
            volume=425.2,
            density=7.51
        ),

        predicted_properties=MaterialProperties(
            formula="Bi2Se3",
            electrical=ElectricalProperties(
                band_gap=0.3,  # Bulk band gap
                conductivity=1e5,  # Surface state conductivity
                mobility=10000.0,  # High mobility for surface states
                work_function=5.3
            ),
            thermal=ThermalProperties(
                melting_point=979,
                thermal_conductivity=2.0,
                debye_temperature=185
            ),
            stability=-0.45,
            synthesizability=0.90
        ),

        theoretical_basis=[
            "Strong spin-orbit coupling in Bi creates band inversion",
            "Time-reversal symmetry protects surface states",
            "Quintuple layer structure with van der Waals gaps",
            "Surface states form Dirac cone at Gamma point"
        ],

        confidence_score=0.85,
        impact_score=9.5,
        novelty_score=7.0,

        related_materials=["Bi2Te3", "Sb2Te3", "Bi2Te2Se"],
        citations=[
            "DOI: 10.1038/nphys1270",
            "DOI: 10.1126/science.1173034"
        ]
    )

    protocol = MaterialsScienceProtocol()

    print("\nValidating topological insulator hypothesis...")
    results = await protocol.validate_hypothesis(
        hypothesis,
        validation_level="comprehensive"
    )

    print("\n" + "=" * 40)
    print("QUANTUM MATERIAL VALIDATION")
    print("=" * 40)

    # Focus on electrical properties validation
    if "properties" in results["stages"]:
        props = results["stages"]["properties"]
        print(f"\nProperty Validation:")
        print(f"  Plausible: {props.get('plausible', False)}")
        if props.get("warnings"):
            print(f"  Warnings:")
            for warning in props["warnings"][:3]:  # Show first 3 warnings
                print(f"    - {warning}")

    # Check novelty
    if "database" in results["stages"]:
        db_results = results["stages"]["database"]
        print(f"\nNovelty Assessment:")
        print(f"  Novel: {db_results.get('novel', False)}")
        if db_results.get("similar_materials"):
            print(f"  Similar materials found: {len(db_results['similar_materials'])}")

    print(f"\nFinal Score: {results['scores']['overall']:.2f}")
    print(f"Recommendation: {results['recommendation']}")

    return results


async def example_structure_prediction():
    """Example: Ab initio structure prediction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Crystal Structure Prediction from Composition")
    print("=" * 80)

    # Initialize structure predictor
    predictor = StructurePredictor()

    # Predict structure for a new composition
    formula = "MgAl2O4"
    print(f"\nPredicting structure for {formula}...")

    structure = await predictor.predict_structure(formula, method="evolutionary")

    print(f"\nPredicted Structure:")
    print(f"  Formula: {structure.formula}")
    print(f"  Lattice Type: {structure.lattice_type.value}")
    print(f"  Space Group: {structure.space_group.symbol} (#{structure.space_group.number})")
    print(f"  Lattice Parameters:")
    for param, value in structure.lattice_parameters.items():
        unit = "Å" if param in ["a", "b", "c"] else "°"
        print(f"    {param}: {value:.2f} {unit}")

    # Validate the predicted structure
    print(f"\nValidating predicted structure...")

    is_valid = await predictor.validate_space_group(structure)
    print(f"  Space group consistency: {'✓' if is_valid else '✗'}")

    issues = await predictor.check_atomic_positions(structure)
    if issues:
        print(f"  Position issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  Atomic positions: ✓")

    packing = await predictor.calculate_packing_fraction(structure)
    print(f"  Packing fraction: {packing:.3f}")

    structure_type = await predictor.identify_structure_type(structure)
    print(f"  Structure type: {structure_type}")

    # Now predict properties from structure
    prop_predictor = PropertyPredictor()
    predicted_props = await prop_predictor.predict_from_structure(structure)

    print(f"\nPredicted Properties from Structure:")
    if "mechanical" in predicted_props:
        mech = predicted_props["mechanical"]
        print(f"  Mechanical:")
        print(f"    Bulk modulus: {mech.get('bulk_modulus', 0):.1f} GPa")
        print(f"    Young's modulus: {mech.get('youngs_modulus', 0):.1f} GPa")

    if "electrical" in predicted_props:
        elec = predicted_props["electrical"]
        print(f"  Electrical:")
        print(f"    Band gap: {elec.get('band_gap', 0):.1f} eV")
        print(f"    Conductivity: {elec.get('conductivity', 0):.1e} S/m")

    return structure


async def main():
    """Run all examples."""
    print("=" * 80)
    print("MATERIALS SCIENCE RESEARCH MODULE - EXAMPLE WORKFLOWS")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")

    # Run examples
    results = []

    # Example 1: Perovskite solar cell
    result1 = await example_perovskite_hypothesis()
    results.append(("Perovskite Solar Cell", result1))

    # Example 2: Battery material
    result2 = await example_battery_material()
    results.append(("Battery Cathode", result2))

    # Example 3: Quantum material
    result3 = await example_quantum_material()
    results.append(("Topological Insulator", result3))

    # Example 4: Structure prediction
    result4 = await example_structure_prediction()
    results.append(("Structure Prediction", result4))

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)

    for name, result in results[:3]:  # First 3 have validation scores
        if isinstance(result, dict) and "scores" in result:
            score = result["scores"]["overall"]
            rec = result["recommendation"]
            print(f"\n{name}:")
            print(f"  Score: {score:.2f}")
            print(f"  Recommendation: {rec}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())