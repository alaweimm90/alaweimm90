"""
Beginner Example: Simple Crystal Structure Analysis
This example demonstrates basic crystallographic calculations using simple
crystal structures like sodium chloride (NaCl) and diamond.
Learning Objectives:
- Understand crystal lattice parameters
- Calculate unit cell volume and density
- Compute interatomic distances
- Determine d-spacings and Bragg angles
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# Add Crystallography module to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.crystal_structure import CrystalStructure, LatticeParameters, AtomicPosition
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
BERKELEY_GOLD = '#FDB515'
def create_nacl_structure():
    """Create sodium chloride (NaCl) crystal structure."""
    print("Creating NaCl (Sodium Chloride) Structure")
    print("-" * 40)
    # NaCl has a face-centered cubic structure with a = 5.64 Å
    lattice = LatticeParameters(
        a=5.64, b=5.64, c=5.64,  # Cubic structure
        alpha=90, beta=90, gamma=90
    )
    # Atomic positions in the unit cell
    atoms = [
        # Sodium atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        AtomicPosition('Na', 0.0, 0.0, 0.0),
        AtomicPosition('Na', 0.5, 0.5, 0.0),
        AtomicPosition('Na', 0.5, 0.0, 0.5),
        AtomicPosition('Na', 0.0, 0.5, 0.5),
        # Chlorine atoms at (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0.5)
        AtomicPosition('Cl', 0.5, 0.0, 0.0),
        AtomicPosition('Cl', 0.0, 0.5, 0.0),
        AtomicPosition('Cl', 0.0, 0.0, 0.5),
        AtomicPosition('Cl', 0.5, 0.5, 0.5),
    ]
    crystal = CrystalStructure(lattice, atoms)
    print(f"Lattice parameters: a = b = c = {lattice.a} Å")
    print(f"Number of atoms in unit cell: {len(atoms)}")
    print(f"Unit cell volume: {crystal.unit_cell_volume():.2f} Å³")
    # Calculate density
    # NaCl molecular weight = 22.99 (Na) + 35.45 (Cl) = 58.44 g/mol
    # 4 formula units per unit cell (FCC structure)
    molecular_weight = 58.44  # g/mol
    z = 4  # formula units per unit cell
    density = crystal.density(molecular_weight, z)
    print(f"Calculated density: {density:.2f} g/cm³")
    print(f"Experimental density: ~2.16 g/cm³")
    return crystal
def analyze_interatomic_distances(crystal):
    """Analyze interatomic distances in the crystal."""
    print("\nInteratomic Distance Analysis")
    print("-" * 30)
    # Calculate Na-Cl distances (nearest neighbors)
    na_indices = [i for i, atom in enumerate(crystal.atoms) if atom.element == 'Na']
    cl_indices = [i for i, atom in enumerate(crystal.atoms) if atom.element == 'Cl']
    na_cl_distances = []
    for na_idx in na_indices:
        for cl_idx in cl_indices:
            distance = crystal.interatomic_distance(na_idx, cl_idx, include_symmetry=True)
            na_cl_distances.append(distance)
    # Find unique distances (within tolerance)
    unique_distances = []
    tolerance = 0.01
    for distance in na_cl_distances:
        is_unique = True
        for unique_dist in unique_distances:
            if abs(distance - unique_dist) < tolerance:
                is_unique = False
                break
        if is_unique:
            unique_distances.append(distance)
    unique_distances.sort()
    print("Na-Cl distances:")
    for i, dist in enumerate(unique_distances[:3]):  # Show first few
        print(f"  Distance {i+1}: {dist:.3f} Å")
    # Nearest neighbor distance should be a/2 = 2.82 Å
    theoretical_nn = crystal.lattice.a / 2
    print(f"Theoretical nearest neighbor distance: {theoretical_nn:.3f} Å")
    return unique_distances
def calculate_diffraction_data(crystal):
    """Calculate basic diffraction information."""
    print("\nX-ray Diffraction Analysis")
    print("-" * 30)
    # Common X-ray wavelength (Cu Kα)
    wavelength = 1.54056  # Å
    # Calculate d-spacings and Bragg angles for low-index reflections
    reflections = [
        (1, 0, 0), (1, 1, 0), (1, 1, 1),
        (2, 0, 0), (2, 1, 0), (2, 1, 1),
        (2, 2, 0), (3, 1, 0), (2, 2, 2)
    ]
    print("Miller Indices | d-spacing (Å) | 2θ (degrees)")
    print("-" * 45)
    diffraction_data = []
    for h, k, l in reflections:
        try:
            d = crystal.d_spacing(h, k, l)
            two_theta = crystal.bragg_angle(h, k, l, wavelength) * 2
            print(f"    ({h},{k},{l})     |    {d:.3f}     |    {two_theta:.2f}")
            diffraction_data.append({
                'hkl': (h, k, l),
                'd_spacing': d,
                'two_theta': two_theta
            })
        except ValueError:
            # Some reflections may not be allowed
            print(f"    ({h},{k},{l})     |   forbidden   |     N/A")
    return diffraction_data
def create_diamond_structure():
    """Create diamond crystal structure for comparison."""
    print("\n\nCreating Diamond Structure")
    print("-" * 30)
    # Diamond cubic structure with a = 3.567 Å
    lattice = LatticeParameters(
        a=3.567, b=3.567, c=3.567,
        alpha=90, beta=90, gamma=90
    )
    # Carbon atoms in diamond structure
    atoms = [
        AtomicPosition('C', 0.0, 0.0, 0.0),
        AtomicPosition('C', 0.25, 0.25, 0.25),
        AtomicPosition('C', 0.5, 0.5, 0.0),
        AtomicPosition('C', 0.75, 0.75, 0.25),
        AtomicPosition('C', 0.5, 0.0, 0.5),
        AtomicPosition('C', 0.75, 0.25, 0.75),
        AtomicPosition('C', 0.0, 0.5, 0.5),
        AtomicPosition('C', 0.25, 0.75, 0.75),
    ]
    crystal = CrystalStructure(lattice, atoms)
    print(f"Lattice parameter: a = {lattice.a} Å")
    print(f"Unit cell volume: {crystal.unit_cell_volume():.2f} Å³")
    # Calculate density
    # Carbon atomic weight = 12.01 g/mol
    # 8 atoms per unit cell
    molecular_weight = 12.01
    z = 8
    density = crystal.density(molecular_weight, z)
    print(f"Calculated density: {density:.2f} g/cm³")
    print(f"Experimental density: ~3.52 g/cm³")
    return crystal
def plot_comparison(nacl_data, diamond_data):
    """Plot comparison of diffraction patterns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # NaCl diffraction pattern
    nacl_angles = [d['two_theta'] for d in nacl_data]
    nacl_intensities = [100 / (i + 1) for i in range(len(nacl_data))]  # Mock intensities
    ax1.stem(nacl_angles, nacl_intensities, basefmt=' ',
             linefmt=BERKELEY_BLUE, markerfmt='o')
    ax1.set_xlabel('2θ (degrees)')
    ax1.set_ylabel('Relative Intensity')
    ax1.set_title('NaCl Diffraction Pattern')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 80)
    # Add Miller indices labels
    for i, (angle, intensity) in enumerate(zip(nacl_angles[:5], nacl_intensities[:5])):
        hkl = nacl_data[i]['hkl']
        ax1.annotate(f'{hkl[0]}{hkl[1]}{hkl[2]}',
                    (angle, intensity + 5),
                    ha='center', fontsize=8)
    # Diamond diffraction pattern
    diamond_angles = [d['two_theta'] for d in diamond_data]
    diamond_intensities = [100 / (i + 1) for i in range(len(diamond_data))]
    ax2.stem(diamond_angles, diamond_intensities, basefmt=' ',
             linefmt=BERKELEY_GOLD, markerfmt='o')
    ax2.set_xlabel('2θ (degrees)')
    ax2.set_ylabel('Relative Intensity')
    ax2.set_title('Diamond Diffraction Pattern')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 80)
    # Add Miller indices labels
    for i, (angle, intensity) in enumerate(zip(diamond_angles[:5], diamond_intensities[:5])):
        hkl = diamond_data[i]['hkl']
        ax2.annotate(f'{hkl[0]}{hkl[1]}{hkl[2]}',
                    (angle, intensity + 5),
                    ha='center', fontsize=8)
    plt.tight_layout()
    plt.show()
def coordination_analysis(crystal, structure_name):
    """Analyze coordination numbers."""
    print(f"\nCoordination Analysis for {structure_name}")
    print("-" * 40)
    # Calculate coordination numbers for first few atoms
    cutoff_radius = 3.5  # Å
    for i, atom in enumerate(crystal.atoms[:4]):  # Analyze first 4 atoms
        coord_num = crystal.coordination_number(i, cutoff_radius)
        print(f"Atom {i+1} ({atom.element}): coordination number = {coord_num}")
        print(f"  Position: ({atom.x:.3f}, {atom.y:.3f}, {atom.z:.3f})")
def main():
    """Main crystallographic analysis."""
    print("Simple Crystal Structure Analysis")
    print("=" * 50)
    # Create and analyze NaCl structure
    nacl_crystal = create_nacl_structure()
    # Analyze interatomic distances
    distances = analyze_interatomic_distances(nacl_crystal)
    # Calculate diffraction data
    nacl_diffraction = calculate_diffraction_data(nacl_crystal)
    # Coordination analysis
    coordination_analysis(nacl_crystal, "NaCl")
    # Create and analyze diamond structure
    diamond_crystal = create_diamond_structure()
    diamond_diffraction = calculate_diffraction_data(diamond_crystal)
    coordination_analysis(diamond_crystal, "Diamond")
    # Compare structures
    print("\n\nStructure Comparison")
    print("=" * 30)
    print(f"NaCl unit cell volume: {nacl_crystal.unit_cell_volume():.2f} Å³")
    print(f"Diamond unit cell volume: {diamond_crystal.unit_cell_volume():.2f} Å³")
    print(f"\nNaCl coordination: 6 (octahedral)")
    print(f"Diamond coordination: 4 (tetrahedral)")
    # Plot comparison
    plot_comparison(nacl_diffraction, diamond_diffraction)
    # Demonstrate supercell creation
    print("\nSupercell Example")
    print("-" * 20)
    # Create 2x2x2 supercell of NaCl
    supercell = nacl_crystal.supercell(2, 2, 2)
    print(f"Original unit cell: {len(nacl_crystal.atoms)} atoms")
    print(f"2x2x2 supercell: {len(supercell.atoms)} atoms")
    print(f"Supercell volume: {supercell.unit_cell_volume():.2f} Å³")
    print(f"Volume ratio: {supercell.unit_cell_volume() / nacl_crystal.unit_cell_volume():.0f}")
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # Configure matplotlib for Berkeley style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 12
    main()