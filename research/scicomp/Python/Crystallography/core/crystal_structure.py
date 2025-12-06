"""
Crystal Structure Analysis
Comprehensive crystallographic structure representation and analysis tools.
Includes lattice parameter calculations, unit cell operations, and
crystallographic coordinate transformations.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings
@dataclass
class LatticeParameters:
    """Crystal lattice parameters container."""
    a: float  # Lattice parameter a (Å)
    b: float  # Lattice parameter b (Å)
    c: float  # Lattice parameter c (Å)
    alpha: float  # Angle alpha (degrees)
    beta: float   # Angle beta (degrees)
    gamma: float  # Angle gamma (degrees)
    def __post_init__(self):
        """Validate lattice parameters."""
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("Lattice parameters must be positive")
        if not (0 < self.alpha < 180 and 0 < self.beta < 180 and 0 < self.gamma < 180):
            raise ValueError("Lattice angles must be between 0 and 180 degrees")
    @property
    def alpha_rad(self) -> float:
        """Alpha angle in radians."""
        return np.radians(self.alpha)
    @property
    def beta_rad(self) -> float:
        """Beta angle in radians."""
        return np.radians(self.beta)
    @property
    def gamma_rad(self) -> float:
        """Gamma angle in radians."""
        return np.radians(self.gamma)
@dataclass
class AtomicPosition:
    """Atomic position in crystal structure."""
    element: str
    x: float  # Fractional coordinate x
    y: float  # Fractional coordinate y
    z: float  # Fractional coordinate z
    occupancy: float = 1.0
    thermal_factor: float = 0.0  # Isotropic B-factor (Å²)
    def __post_init__(self):
        """Validate atomic position parameters."""
        if not (0 <= self.occupancy <= 1):
            raise ValueError("Occupancy must be between 0 and 1")
        if self.thermal_factor < 0:
            raise ValueError("Thermal factor must be non-negative")
class CrystalStructure:
    """
    Crystal structure representation and analysis.
    Features:
    - Lattice parameter calculations
    - Unit cell volume and density
    - Coordinate transformations (fractional ↔ Cartesian)
    - Interatomic distance calculations
    - Miller indices operations
    - Structure factor calculations
    Examples:
        >>> lattice = LatticeParameters(a=5.0, b=5.0, c=5.0, alpha=90, beta=90, gamma=90)
        >>> atoms = [AtomicPosition('Si', 0.0, 0.0, 0.0), AtomicPosition('Si', 0.5, 0.5, 0.5)]
        >>> crystal = CrystalStructure(lattice, atoms)
        >>> print(f"Unit cell volume: {crystal.unit_cell_volume():.2f} Å³")
    """
    def __init__(self, lattice_parameters: LatticeParameters, atoms: List[AtomicPosition]):
        """
        Initialize crystal structure.
        Parameters:
            lattice_parameters: Crystal lattice parameters
            atoms: List of atomic positions
        """
        self.lattice = lattice_parameters
        self.atoms = atoms
        # Compute fundamental matrices
        self._metric_tensor = None
        self._direct_matrix = None
        self._reciprocal_matrix = None
        self._compute_matrices()
    def _compute_matrices(self):
        """Compute fundamental crystallographic matrices."""
        # Direct lattice matrix (columns are lattice vectors)
        a, b, c = self.lattice.a, self.lattice.b, self.lattice.c
        alpha, beta, gamma = self.lattice.alpha_rad, self.lattice.beta_rad, self.lattice.gamma_rad
        # Volume calculation
        cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sin_alpha, sin_beta, sin_gamma = np.sin(alpha), np.sin(beta), np.sin(gamma)
        volume = a * b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
                                    + 2*cos_alpha*cos_beta*cos_gamma)
        # Direct lattice matrix
        self._direct_matrix = np.array([
            [a, b*cos_gamma, c*cos_beta],
            [0, b*sin_gamma, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma],
            [0, 0, volume/(a*b*sin_gamma)]
        ])
        # Metric tensor (dot products of lattice vectors)
        self._metric_tensor = np.array([
            [a**2, a*b*cos_gamma, a*c*cos_beta],
            [a*b*cos_gamma, b**2, b*c*cos_alpha],
            [a*c*cos_beta, b*c*cos_alpha, c**2]
        ])
        # Reciprocal lattice matrix
        self._reciprocal_matrix = 2*np.pi * np.linalg.inv(self._direct_matrix).T
    @property
    def direct_matrix(self) -> np.ndarray:
        """Direct lattice matrix (3x3)."""
        return self._direct_matrix.copy()
    @property
    def reciprocal_matrix(self) -> np.ndarray:
        """Reciprocal lattice matrix (3x3)."""
        return self._reciprocal_matrix.copy()
    @property
    def metric_tensor(self) -> np.ndarray:
        """Metric tensor (3x3)."""
        return self._metric_tensor.copy()
    def unit_cell_volume(self) -> float:
        """Calculate unit cell volume in Å³."""
        return np.abs(np.linalg.det(self._direct_matrix))
    def density(self, molecular_weight: float, z: int = 1) -> float:
        """
        Calculate crystal density.
        Parameters:
            molecular_weight: Molecular weight (g/mol)
            z: Number of formula units per unit cell
        Returns:
            Density in g/cm³
        """
        avogadro = 6.02214076e23  # mol⁻¹
        volume_cm3 = self.unit_cell_volume() * 1e-24  # Å³ to cm³
        return (z * molecular_weight) / (avogadro * volume_cm3)
    def fractional_to_cartesian(self, fractional_coords: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to Cartesian coordinates.
        Parameters:
            fractional_coords: Fractional coordinates (Nx3 or 3,)
        Returns:
            Cartesian coordinates in Å
        """
        fractional_coords = np.atleast_2d(fractional_coords)
        return (self._direct_matrix @ fractional_coords.T).T
    def cartesian_to_fractional(self, cartesian_coords: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian coordinates to fractional coordinates.
        Parameters:
            cartesian_coords: Cartesian coordinates in Å (Nx3 or 3,)
        Returns:
            Fractional coordinates
        """
        cartesian_coords = np.atleast_2d(cartesian_coords)
        return (np.linalg.inv(self._direct_matrix) @ cartesian_coords.T).T
    def interatomic_distance(self, atom1_idx: int, atom2_idx: int,
                           include_symmetry: bool = False) -> float:
        """
        Calculate distance between two atoms.
        Parameters:
            atom1_idx: Index of first atom
            atom2_idx: Index of second atom
            include_symmetry: Whether to consider periodic boundary conditions
        Returns:
            Distance in Å
        """
        if atom1_idx >= len(self.atoms) or atom2_idx >= len(self.atoms):
            raise IndexError("Atom index out of range")
        atom1 = self.atoms[atom1_idx]
        atom2 = self.atoms[atom2_idx]
        # Fractional coordinate difference
        df = np.array([atom2.x - atom1.x, atom2.y - atom1.y, atom2.z - atom1.z])
        if include_symmetry:
            # Apply minimum image convention
            df = df - np.round(df)
        # Distance using metric tensor
        distance_squared = df @ self._metric_tensor @ df
        return np.sqrt(distance_squared)
    def d_spacing(self, h: int, k: int, l: int) -> float:
        """
        Calculate d-spacing for Miller indices (hkl).
        Parameters:
            h, k, l: Miller indices
        Returns:
            d-spacing in Å
        """
        # Reciprocal lattice vector
        hkl = np.array([h, k, l])
        # Calculate |G|² where G is reciprocal lattice vector
        reciprocal_metric = np.linalg.inv(self._metric_tensor)
        g_squared = hkl @ reciprocal_metric @ hkl
        if g_squared <= 0:
            raise ValueError("Invalid Miller indices result in zero d-spacing")
        return 1.0 / np.sqrt(g_squared)
    def bragg_angle(self, h: int, k: int, l: int, wavelength: float) -> float:
        """
        Calculate Bragg angle for reflection (hkl).
        Parameters:
            h, k, l: Miller indices
            wavelength: X-ray wavelength in Å
        Returns:
            Bragg angle in degrees
        """
        d = self.d_spacing(h, k, l)
        sin_theta = wavelength / (2 * d)
        if sin_theta > 1:
            raise ValueError(f"No diffraction possible for hkl=({h},{k},{l}) at λ={wavelength} Å")
        return np.degrees(np.arcsin(sin_theta))
    def systematic_absences(self, space_group: str) -> List[Tuple[int, int, int]]:
        """
        Determine systematic absences for given space group.
        Parameters:
            space_group: Space group symbol (e.g., 'Fm3m', 'P21/c')
        Returns:
            List of forbidden reflections
        """
        # Simplified implementation for common space groups
        forbidden = []
        # Face-centered cubic (F)
        if space_group.startswith('F'):
            # h+k, h+l, k+l must all be even
            for h in range(-5, 6):
                for k in range(-5, 6):
                    for l in range(-5, 6):
                        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
                            forbidden.append((h, k, l))
        # Body-centered (I)
        elif space_group.startswith('I'):
            # h+k+l must be even
            for h in range(-5, 6):
                for k in range(-5, 6):
                    for l in range(-5, 6):
                        if (h + k + l) % 2 != 0:
                            forbidden.append((h, k, l))
        # Add more space group rules as needed
        return forbidden
    def coordination_number(self, atom_idx: int, cutoff_radius: float = 3.0) -> int:
        """
        Calculate coordination number for an atom.
        Parameters:
            atom_idx: Index of central atom
            cutoff_radius: Maximum distance for coordination (Å)
        Returns:
            Coordination number
        """
        if atom_idx >= len(self.atoms):
            raise IndexError("Atom index out of range")
        coordination = 0
        central_atom = self.atoms[atom_idx]
        # Check all other atoms
        for i, atom in enumerate(self.atoms):
            if i == atom_idx:
                continue
            distance = self.interatomic_distance(atom_idx, i, include_symmetry=True)
            if distance <= cutoff_radius:
                coordination += 1
        # Also check periodic images (simplified)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    for i, atom in enumerate(self.atoms):
                        if i == atom_idx:
                            continue
                        # Translate atom position
                        translated_pos = np.array([atom.x + dx, atom.y + dy, atom.z + dz])
                        central_pos = np.array([central_atom.x, central_atom.y, central_atom.z])
                        df = translated_pos - central_pos
                        distance_squared = df @ self._metric_tensor @ df
                        distance = np.sqrt(distance_squared)
                        if distance <= cutoff_radius:
                            coordination += 1
        return coordination
    def powder_pattern_positions(self, wavelength: float, max_2theta: float = 90.0,
                                min_d_spacing: float = 1.0) -> List[Dict]:
        """
        Calculate powder diffraction peak positions.
        Parameters:
            wavelength: X-ray wavelength in Å
            max_2theta: Maximum 2θ angle in degrees
            min_d_spacing: Minimum d-spacing to consider (Å)
        Returns:
            List of reflection information dictionaries
        """
        reflections = []
        # Generate Miller indices up to reasonable limits
        max_h = max(5, int(2*self.lattice.a / min_d_spacing))
        max_k = max(5, int(2*self.lattice.b / min_d_spacing))
        max_l = max(5, int(2*self.lattice.c / min_d_spacing))
        for h in range(-max_h, max_h + 1):
            for k in range(-max_k, max_k + 1):
                for l in range(-max_l, max_l + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    try:
                        d = self.d_spacing(h, k, l)
                        if d < min_d_spacing:
                            continue
                        theta = self.bragg_angle(h, k, l, wavelength)
                        two_theta = 2 * theta
                        if two_theta <= max_2theta:
                            reflections.append({
                                'hkl': (h, k, l),
                                'd_spacing': d,
                                'theta': theta,
                                '2theta': two_theta,
                                'multiplicity': self._calculate_multiplicity(h, k, l)
                            })
                    except ValueError:
                        # Skip reflections that don't satisfy Bragg condition
                        continue
        # Sort by 2θ angle
        reflections.sort(key=lambda x: x['2theta'])
        return reflections
    def _calculate_multiplicity(self, h: int, k: int, l: int) -> int:
        """Calculate multiplicity factor for reflection (simplified)."""
        # This is a simplified calculation
        # Full implementation would require space group information
        unique_indices = len(set([abs(h), abs(k), abs(l)]))
        if unique_indices == 1:  # All indices equal (e.g., 111)
            return 8
        elif unique_indices == 2:  # Two indices equal (e.g., 110)
            return 12
        else:  # All indices different (e.g., 123)
            return 24
    def get_fractional_coordinates(self) -> np.ndarray:
        """Get all atomic fractional coordinates as array."""
        coords = np.zeros((len(self.atoms), 3))
        for i, atom in enumerate(self.atoms):
            coords[i] = [atom.x, atom.y, atom.z]
        return coords
    def get_cartesian_coordinates(self) -> np.ndarray:
        """Get all atomic Cartesian coordinates as array."""
        fractional = self.get_fractional_coordinates()
        return self.fractional_to_cartesian(fractional)
    def supercell(self, nx: int, ny: int, nz: int) -> 'CrystalStructure':
        """
        Create supercell structure.
        Parameters:
            nx, ny, nz: Supercell dimensions
        Returns:
            New CrystalStructure representing supercell
        """
        # New lattice parameters
        new_lattice = LatticeParameters(
            a=self.lattice.a * nx,
            b=self.lattice.b * ny,
            c=self.lattice.c * nz,
            alpha=self.lattice.alpha,
            beta=self.lattice.beta,
            gamma=self.lattice.gamma
        )
        # Replicate atoms
        new_atoms = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for atom in self.atoms:
                        new_atom = AtomicPosition(
                            element=atom.element,
                            x=(atom.x + i) / nx,
                            y=(atom.y + j) / ny,
                            z=(atom.z + k) / nz,
                            occupancy=atom.occupancy,
                            thermal_factor=atom.thermal_factor
                        )
                        new_atoms.append(new_atom)
        return CrystalStructure(new_lattice, new_atoms)