"""Multiphysics Coupling and Simulation Framework.
This package implements comprehensive multiphysics coupling methods
for scientific computing, including fluid-structure interaction,
thermal-mechanical coupling, electromagnetic-thermal problems,
and multi-scale modeling approaches.
Modules:
    coupling: Coupling strategies and interfaces
    fluid_structure: Fluid-structure interaction (FSI)
    thermal_mechanical: Thermal-mechanical coupling
    electromagnetic: Electromagnetic-thermal coupling
    transport: Multi-physics transport phenomena
    solvers: Coupled system solvers
    visualization: Multiphysics visualization
    utils: Utility functions for multiphysics
Examples:
    >>> from Multiphysics import CoupledSystem, FluidStructureInteraction
    >>> fsi = FluidStructureInteraction(fluid_model, structure_model)
    >>> result = fsi.solve(time_span=(0, 10), coupling_scheme='implicit')
Author: Berkeley SciComp Team
Date: 2024
"""
from .coupling import (
    CoupledSystem,
    CouplingInterface,
    CouplingScheme,
    FieldTransfer,
    create_coupling_interface,
    monolithic_coupling,
    partitioned_coupling
)
from .fluid_structure import (
    FluidStructureInteraction,
    FluidSolver,
    StructuralSolver,
    MeshMotion,
    ALE,
    fsi_benchmark,
    vortex_induced_vibration
)
from .thermal_mechanical import (
    ThermalMechanicalCoupling,
    ThermalExpansion,
    ThermoelasticSolver,
    HeatGenerationModel,
    thermal_stress_analysis,
    coupled_heat_conduction
)
from .electromagnetic import (
    ElectromagneticThermalCoupling,
    MaxwellSolver,
    JouleHeating,
    InductionHeating,
    EddyCurrentSolver,
    electromagnetic_heating,
    coupled_em_thermal
)
from .transport import (
    MultiphysicsTransport,
    ReactiveTransport,
    PorousMediaFlow,
    ConvectionDiffusionReaction,
    species_transport,
    coupled_flow_transport
)
from .solvers import (
    MultiphysicsSolver,
    MonolithicSolver,
    PartitionedSolver,
    StaggeredSolver,
    NewtonRaphson,
    FixedPointIteration,
    solve_coupled_system
)
from .visualization import (
    MultiphysicsVisualizer,
    plot_coupled_fields,
    plot_interface_data,
    animate_multiphysics,
    create_multiphysics_plot
)
from .utils import (
    interpolate_fields,
    project_solution,
    compute_interface_forces,
    check_conservation,
    multiphysics_metrics
)
__all__ = [
    # Coupling
    'CoupledSystem',
    'CouplingInterface',
    'CouplingScheme',
    'FieldTransfer',
    'create_coupling_interface',
    'monolithic_coupling',
    'partitioned_coupling',
    # Fluid-Structure
    'FluidStructureInteraction',
    'FluidSolver',
    'StructuralSolver',
    'MeshMotion',
    'ALE',
    'fsi_benchmark',
    'vortex_induced_vibration',
    # Thermal-Mechanical
    'ThermalMechanicalCoupling',
    'ThermalExpansion',
    'ThermoelasticSolver',
    'HeatGenerationModel',
    'thermal_stress_analysis',
    'coupled_heat_conduction',
    # Electromagnetic
    'ElectromagneticThermalCoupling',
    'MaxwellSolver',
    'JouleHeating',
    'InductionHeating',
    'EddyCurrentSolver',
    'electromagnetic_heating',
    'coupled_em_thermal',
    # Transport
    'MultiphysicsTransport',
    'ReactiveTransport',
    'PorousMediaFlow',
    'ConvectionDiffusionReaction',
    'species_transport',
    'coupled_flow_transport',
    # Solvers
    'MultiphysicsSolver',
    'MonolithicSolver',
    'PartitionedSolver',
    'StaggeredSolver',
    'NewtonRaphson',
    'FixedPointIteration',
    'solve_coupled_system',
    # Visualization
    'MultiphysicsVisualizer',
    'plot_coupled_fields',
    'plot_interface_data',
    'animate_multiphysics',
    'create_multiphysics_plot',
    # Utils
    'interpolate_fields',
    'project_solution',
    'compute_interface_forces',
    'check_conservation',
    'multiphysics_metrics'
]
__version__ = '1.0.0'
__author__ = 'Berkeley SciComp Team'
__email__ = 'scicomp@berkeley.edu'