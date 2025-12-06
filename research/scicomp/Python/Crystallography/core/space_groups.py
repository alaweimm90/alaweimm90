"""
Space Group Operations
Crystallographic space group analysis and symmetry operations.
Includes space group identification, symmetry operation generation,
and equivalent position calculations.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re
class CrystalSystem(Enum):
    """Crystal system enumeration."""
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"
class LatticeType(Enum):
    """Bravais lattice type enumeration."""
    P = "primitive"
    I = "body_centered"
    F = "face_centered"
    C = "c_centered"
    A = "a_centered"
    B = "b_centered"
    R = "rhombohedral"
@dataclass
class SymmetryOperation:
    """Crystallographic symmetry operation."""
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation_vector: np.ndarray  # 3x1 translation vector
    symbol: str = ""  # Hermann-Mauguin symbol
    def __post_init__(self):
        """Validate symmetry operation."""
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if self.translation_vector.shape != (3,):
            raise ValueError("Translation vector must be length 3")
        # Check if rotation matrix is orthogonal (det = ±1)
        det = np.linalg.det(self.rotation_matrix)
        if not np.isclose(abs(det), 1.0, atol=1e-10):
            raise ValueError("Rotation matrix must be orthogonal")
    def apply(self, position: np.ndarray) -> np.ndarray:
        """
        Apply symmetry operation to position.
        Parameters:
            position: Fractional coordinates (3,)
        Returns:
            Transformed fractional coordinates
        """
        return self.rotation_matrix @ position + self.translation_vector
    def inverse(self) -> 'SymmetryOperation':
        """Get inverse symmetry operation."""
        inv_rotation = self.rotation_matrix.T  # For orthogonal matrices
        inv_translation = -inv_rotation @ self.translation_vector
        return SymmetryOperation(
            rotation_matrix=inv_rotation,
            translation_vector=inv_translation,
            symbol=f"inv({self.symbol})"
        )
    def __mul__(self, other: 'SymmetryOperation') -> 'SymmetryOperation':
        """Compose two symmetry operations."""
        new_rotation = self.rotation_matrix @ other.rotation_matrix
        new_translation = self.rotation_matrix @ other.translation_vector + self.translation_vector
        return SymmetryOperation(
            rotation_matrix=new_rotation,
            translation_vector=new_translation,
            symbol=f"{self.symbol}*{other.symbol}"
        )
    def order(self) -> int:
        """Calculate order of rotation operation."""
        # Find eigenvalues to determine rotation angle
        eigenvals = np.linalg.eigvals(self.rotation_matrix)
        # For rotation matrices, one eigenvalue is 1, others are e^(iθ)
        for val in eigenvals:
            if not np.isclose(val.real, 1.0) and np.isclose(val.imag, 0.0):
                continue
            if np.isclose(val.imag, 0.0):
                continue
            # Calculate angle from complex eigenvalue
            angle = np.angle(val)
            if angle < 0:
                angle += 2*np.pi
            # Order is 2π/angle
            order = int(round(2*np.pi / angle))
            if order > 0:
                return order
        return 1  # Identity operation
class SpaceGroup:
    """
    Crystallographic space group representation.
    Features:
    - Space group symbol parsing
    - Symmetry operation generation
    - Equivalent position calculation
    - Systematic absence determination
    - Crystal system identification
    Examples:
        >>> sg = SpaceGroup("P21/c")
        >>> print(f"Crystal system: {sg.crystal_system}")
        >>> ops = sg.generate_symmetry_operations()
        >>> print(f"Number of operations: {len(ops)}")
    """
    def __init__(self, symbol: str):
        """
        Initialize space group.
        Parameters:
            symbol: Hermann-Mauguin space group symbol
        """
        self.symbol = symbol.strip()
        self.number = None
        self._operations = None
        # Parse space group information
        self._parse_symbol()
        # Space group database (simplified)
        self._initialize_database()
    def _parse_symbol(self):
        """Parse Hermann-Mauguin symbol."""
        symbol = self.symbol.replace(" ", "")
        # Extract lattice type
        if symbol[0] in ['P', 'I', 'F', 'C', 'A', 'B', 'R']:
            self.lattice_type = LatticeType(symbol[0])
        else:
            raise ValueError(f"Invalid lattice type in symbol: {symbol}")
        # Extract point group information
        self.point_group = symbol[1:].replace("/", "")
    def _initialize_database(self):
        """Initialize space group database (simplified)."""
        # This is a simplified database - full implementation would have all 230 space groups
        self._database = {
            "P1": {"number": 1, "system": CrystalSystem.TRICLINIC, "operations": ["x,y,z"]},
            "P-1": {"number": 2, "system": CrystalSystem.TRICLINIC, "operations": ["x,y,z", "-x,-y,-z"]},
            "P21/c": {"number": 14, "system": CrystalSystem.MONOCLINIC,
                     "operations": ["x,y,z", "-x,y+1/2,-z+1/2", "-x,-y,-z", "x,-y+1/2,z+1/2"]},
            "Pmmm": {"number": 47, "system": CrystalSystem.ORTHORHOMBIC,
                    "operations": ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                                 "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z"]},
            "Fm3m": {"number": 225, "system": CrystalSystem.CUBIC,
                    "operations": ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                                 "z,x,y", "z,-x,-y", "-z,-x,y", "-z,x,-y",
                                 "y,z,x", "-y,z,-x", "y,-z,-x", "-y,-z,x"]},
            "Im3m": {"number": 229, "system": CrystalSystem.CUBIC,
                    "operations": ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                                 "z,x,y", "z,-x,-y", "-z,-x,y", "-z,x,-y",
                                 "y,z,x", "-y,z,-x", "y,-z,-x", "-y,-z,x"]},
            "P6/mmm": {"number": 191, "system": CrystalSystem.HEXAGONAL,
                      "operations": ["x,y,z", "-y,x-y,z", "-x+y,-x,z", "-x,-y,z",
                                   "y,-x+y,z", "x-y,x,z", "-x,-y,-z", "y,-x+y,-z",
                                   "x-y,x,-z", "x,y,-z", "-y,x-y,-z", "-x+y,-x,-z"]}
        }
        if self.symbol in self._database:
            info = self._database[self.symbol]
            self.number = info["number"]
            self.crystal_system = info["system"]
            self._operation_strings = info["operations"]
        else:
            warnings.warn(f"Space group {self.symbol} not in database. Using default.")
            self.crystal_system = CrystalSystem.TRICLINIC
            self._operation_strings = ["x,y,z"]
    def generate_symmetry_operations(self) -> List[SymmetryOperation]:
        """Generate all symmetry operations for this space group."""
        if self._operations is not None:
            return self._operations
        operations = []
        for op_string in self._operation_strings:
            rotation, translation = self._parse_operation_string(op_string)
            # Add lattice centering operations if needed
            base_operations = [(rotation, translation)]
            if self.lattice_type == LatticeType.I:  # Body-centered
                base_operations.append((rotation, translation + np.array([0.5, 0.5, 0.5])))
            elif self.lattice_type == LatticeType.F:  # Face-centered
                base_operations.extend([
                    (rotation, translation + np.array([0.5, 0.5, 0.0])),
                    (rotation, translation + np.array([0.5, 0.0, 0.5])),
                    (rotation, translation + np.array([0.0, 0.5, 0.5]))
                ])
            elif self.lattice_type == LatticeType.C:  # C-centered
                base_operations.append((rotation, translation + np.array([0.5, 0.5, 0.0])))
            for rot, trans in base_operations:
                # Reduce translation to [0,1)
                trans = trans % 1.0
                op = SymmetryOperation(
                    rotation_matrix=rot,
                    translation_vector=trans,
                    symbol=op_string
                )
                operations.append(op)
        self._operations = operations
        return operations
    def _parse_operation_string(self, op_string: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse symmetry operation string like 'x,y,z' or '-x,y+1/2,-z+1/2'."""
        parts = op_string.split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid operation string: {op_string}")
        rotation = np.zeros((3, 3))
        translation = np.zeros(3)
        variables = ['x', 'y', 'z']
        for i, part in enumerate(parts):
            part = part.strip()
            # Parse each coordinate expression
            for j, var in enumerate(variables):
                if var in part:
                    # Check for coefficient
                    if f"-{var}" in part:
                        rotation[i, j] = -1
                    elif f"+{var}" in part or part.startswith(var):
                        rotation[i, j] = 1
                    else:
                        # Look for coefficient
                        pattern = rf'([+-]?\d*){var}'
                        match = re.search(pattern, part)
                        if match:
                            coeff = match.group(1)
                            if coeff in ['', '+']:
                                rotation[i, j] = 1
                            elif coeff == '-':
                                rotation[i, j] = -1
                            else:
                                rotation[i, j] = int(coeff)
            # Parse translation part
            # Look for fractions like +1/2, -1/4, etc.
            frac_pattern = r'([+-]?\d+)/(\d+)'
            matches = re.findall(frac_pattern, part)
            for match in matches:
                numerator, denominator = int(match[0]), int(match[1])
                translation[i] += numerator / denominator
        return rotation, translation
    def equivalent_positions(self, position: np.ndarray) -> List[np.ndarray]:
        """
        Generate all equivalent positions for given fractional coordinates.
        Parameters:
            position: Fractional coordinates (3,)
        Returns:
            List of equivalent positions
        """
        operations = self.generate_symmetry_operations()
        equivalent = []
        for op in operations:
            new_pos = op.apply(position)
            # Reduce to unit cell
            new_pos = new_pos % 1.0
            # Check if this position is already in the list (within tolerance)
            is_new = True
            for existing_pos in equivalent:
                if np.allclose(new_pos, existing_pos, atol=1e-6):
                    is_new = False
                    break
                # Also check periodic images
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            test_pos = existing_pos + np.array([dx, dy, dz])
                            if np.allclose(new_pos, test_pos, atol=1e-6):
                                is_new = False
                                break
                        if not is_new:
                            break
                    if not is_new:
                        break
                if not is_new:
                    break
            if is_new:
                equivalent.append(new_pos)
        return equivalent
    def multiplicity(self, position: np.ndarray) -> int:
        """Calculate multiplicity of a position."""
        return len(self.equivalent_positions(position))
    def systematic_absences(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Determine systematic absences for this space group.
        Returns:
            Dictionary of absence conditions and forbidden reflections
        """
        absences = {"conditions": [], "forbidden": []}
        # General conditions based on lattice type
        if self.lattice_type == LatticeType.I:
            absences["conditions"].append("h+k+l ≠ 2n")
            # Generate some forbidden reflections
            for h in range(-3, 4):
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        if (h + k + l) % 2 != 0:
                            absences["forbidden"].append((h, k, l))
        elif self.lattice_type == LatticeType.F:
            absences["conditions"].append("h+k, h+l, k+l ≠ 2n")
            for h in range(-3, 4):
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        if ((h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0):
                            absences["forbidden"].append((h, k, l))
        elif self.lattice_type == LatticeType.C:
            absences["conditions"].append("h+k ≠ 2n")
            for h in range(-3, 4):
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        if (h + k) % 2 != 0:
                            absences["forbidden"].append((h, k, l))
        # Add specific conditions for space groups with screw axes or glide planes
        if "21" in self.symbol:
            absences["conditions"].append("Special conditions for 21 screw axis")
        if "/c" in self.symbol:
            absences["conditions"].append("Special conditions for c glide plane")
        return absences
    def is_centrosymmetric(self) -> bool:
        """Check if space group is centrosymmetric."""
        operations = self.generate_symmetry_operations()
        # Look for inversion operation (-x, -y, -z)
        inversion = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for op in operations:
            if np.allclose(op.rotation_matrix, inversion, atol=1e-6):
                return True
        return False
    def point_group_operations(self) -> List[SymmetryOperation]:
        """Extract point group operations (rotations only, no translations)."""
        operations = self.generate_symmetry_operations()
        point_ops = []
        seen_rotations = set()
        for op in operations:
            # Convert rotation matrix to tuple for hashing
            rot_tuple = tuple(op.rotation_matrix.flatten())
            if rot_tuple not in seen_rotations:
                point_op = SymmetryOperation(
                    rotation_matrix=op.rotation_matrix,
                    translation_vector=np.zeros(3),
                    symbol=op.symbol + "_point"
                )
                point_ops.append(point_op)
                seen_rotations.add(rot_tuple)
        return point_ops
    def wyckoff_positions(self) -> Dict[str, Dict]:
        """
        Get Wyckoff positions for this space group (simplified).
        Returns:
            Dictionary of Wyckoff positions with multiplicities and site symmetries
        """
        # This is a highly simplified implementation
        # Full implementation would require complete Wyckoff position database
        operations = self.generate_symmetry_operations()
        total_ops = len(operations)
        wyckoff = {}
        # General position
        wyckoff['general'] = {
            'multiplicity': total_ops,
            'site_symmetry': '1',
            'coordinates': 'x,y,z'
        }
        # Special positions (simplified)
        if self.is_centrosymmetric():
            wyckoff['inversion'] = {
                'multiplicity': total_ops // 2,
                'site_symmetry': '-1',
                'coordinates': '0,0,0'
            }
        return wyckoff
    def __str__(self) -> str:
        """String representation."""
        return f"SpaceGroup({self.symbol}, #{self.number}, {self.crystal_system.value})"
    def __repr__(self) -> str:
        """Detailed representation."""
        ops = len(self.generate_symmetry_operations())
        return f"SpaceGroup(symbol='{self.symbol}', number={self.number}, " \
               f"system={self.crystal_system.value}, operations={ops})"
def identify_space_group(lattice_parameters: 'LatticeParameters',
                        systematic_absences: List[Tuple[int, int, int]]) -> List[str]:
    """
    Identify possible space groups from lattice parameters and systematic absences.
    Parameters:
        lattice_parameters: Crystal lattice parameters
        systematic_absences: List of systematically absent reflections
    Returns:
        List of possible space group symbols
    """
    # Determine crystal system from lattice parameters
    a, b, c = lattice_parameters.a, lattice_parameters.b, lattice_parameters.c
    alpha, beta, gamma = lattice_parameters.alpha, lattice_parameters.beta, lattice_parameters.gamma
    tol = 1e-3  # Tolerance for parameter comparison
    # Identify crystal system
    if (abs(a - b) < tol and abs(b - c) < tol and
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        system = CrystalSystem.CUBIC
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and
          abs(beta - 90) < tol and abs(gamma - 90) < tol):
        system = CrystalSystem.TETRAGONAL
    elif (abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        system = CrystalSystem.ORTHORHOMBIC
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and
          abs(beta - 90) < tol and abs(gamma - 120) < tol):
        system = CrystalSystem.HEXAGONAL
    elif (abs(beta - 90) < tol and abs(gamma - 90) < tol):
        system = CrystalSystem.MONOCLINIC
    else:
        system = CrystalSystem.TRICLINIC
    # Based on system and absences, suggest space groups
    candidates = []
    if system == CrystalSystem.CUBIC:
        if not systematic_absences:
            candidates.extend(["Pm3m", "P432", "P23"])
        # Add more cubic space groups based on absences
    elif system == CrystalSystem.TETRAGONAL:
        candidates.extend(["P4/mmm", "P422", "P4"])
    elif system == CrystalSystem.ORTHORHOMBIC:
        candidates.extend(["Pmmm", "P222", "Pmm2"])
    elif system == CrystalSystem.MONOCLINIC:
        candidates.extend(["P21/c", "P21/m", "C2/c"])
    elif system == CrystalSystem.TRICLINIC:
        candidates.extend(["P1", "P-1"])
    return candidates