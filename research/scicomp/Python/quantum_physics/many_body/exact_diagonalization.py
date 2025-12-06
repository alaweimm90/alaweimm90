#!/usr/bin/env python3
"""
Exact Diagonalization for Many-Body Quantum Systems
Implementation of exact diagonalization methods for small many-body quantum
systems, including sparse matrix techniques and symmetry exploitation.
Mathematical Foundation:
Exact diagonalization solves the eigenvalue problem:
H|ψ⟩ = E|ψ⟩
For many-body systems, this becomes computationally intensive due to
exponential scaling of Hilbert space dimension.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import Dict, List, Tuple, Optional, Callable, Union
import matplotlib.pyplot as plt
from pathlib import Path
import time
from ...utils.constants import hbar
from ...visualization.berkeley_style import BerkeleyPlot, BERKELEY_BLUE, CALIFORNIA_GOLD
class ExactDiagonalization:
    """
    Exact diagonalization solver for many-body quantum systems.
    Provides efficient methods for constructing and diagonalizing
    many-body Hamiltonians using sparse matrix techniques.
    """
    def __init__(self,
                 n_sites: int,
                 hilbert_space_dimension: Optional[int] = None,
                 conserved_quantities: Optional[List[str]] = None):
        """
        Initialize exact diagonalization solver.
        Parameters
        ----------
        n_sites : int
            Number of lattice sites
        hilbert_space_dimension : int, optional
            Dimension of Hilbert space (computed if not provided)
        conserved_quantities : list, optional
            List of conserved quantities for symmetry exploitation
        """
        self.n_sites = n_sites
        self.conserved_quantities = conserved_quantities or []
        # Hilbert space dimension (default: spin-1/2 system)
        if hilbert_space_dimension is None:
            self.hilbert_dim = 2**n_sites
        else:
            self.hilbert_dim = hilbert_space_dimension
        # Storage for results
        self.hamiltonian = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.ground_state = None
        self.ground_energy = None
        # Symmetry sectors
        self.symmetry_sectors = {}
    def build_basis_states(self, particle_number: Optional[int] = None) -> List[int]:
        """
        Build basis states, optionally restricting to fixed particle number.
        Parameters
        ----------
        particle_number : int, optional
            Fixed particle number (for conservation of particle number)
        Returns
        -------
        list
            List of basis state indices
        """
        if particle_number is None:
            # Full Hilbert space
            return list(range(self.hilbert_dim))
        else:
            # Fixed particle number sector
            basis_states = []
            for state in range(self.hilbert_dim):
                if bin(state).count('1') == particle_number:
                    basis_states.append(state)
            return basis_states
    def create_operator(self,
                       op_type: str,
                       site: int,
                       spin: Optional[str] = None) -> sparse.csr_matrix:
        """
        Create single-site operators in many-body basis.
        Parameters
        ----------
        op_type : str
            Operator type ('c', 'cdag', 'n', 'sx', 'sy', 'sz')
        site : int
            Site index
        spin : str, optional
            Spin component ('up', 'down') for fermionic operators
        Returns
        -------
        csr_matrix
            Sparse operator matrix
        """
        if op_type in ['c', 'cdag', 'n']:  # Fermionic operators
            return self._create_fermionic_operator(op_type, site, spin)
        elif op_type in ['sx', 'sy', 'sz', 'sp', 'sm']:  # Spin operators
            return self._create_spin_operator(op_type, site)
        else:
            raise ValueError(f"Unknown operator type: {op_type}")
    def _create_fermionic_operator(self,
                                 op_type: str,
                                 site: int,
                                 spin: str) -> sparse.csr_matrix:
        """Create fermionic creation/annihilation operators."""
        if spin == 'up':
            bit_position = 2 * site
        elif spin == 'down':
            bit_position = 2 * site + 1
        else:
            raise ValueError("Spin must be 'up' or 'down'")
        rows, cols, data = [], [], []
        for state in range(self.hilbert_dim):
            if op_type == 'c':  # Annihilation
                if (state >> bit_position) & 1:  # Site is occupied
                    new_state = state & ~(1 << bit_position)  # Remove particle
                    # Jordan-Wigner string
                    sign = (-1) ** bin(state & ((1 << bit_position) - 1)).count('1')
                    rows.append(new_state)
                    cols.append(state)
                    data.append(sign)
            elif op_type == 'cdag':  # Creation
                if not ((state >> bit_position) & 1):  # Site is empty
                    new_state = state | (1 << bit_position)  # Add particle
                    # Jordan-Wigner string
                    sign = (-1) ** bin(state & ((1 << bit_position) - 1)).count('1')
                    rows.append(new_state)
                    cols.append(state)
                    data.append(sign)
            elif op_type == 'n':  # Number operator
                if (state >> bit_position) & 1:  # Site is occupied
                    rows.append(state)
                    cols.append(state)
                    data.append(1.0)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.hilbert_dim, self.hilbert_dim))
    def _create_spin_operator(self, op_type: str, site: int) -> sparse.csr_matrix:
        """Create spin operators for spin-1/2 systems."""
        rows, cols, data = [], [], []
        for state in range(self.hilbert_dim):
            if op_type == 'sz':  # S^z
                spin_up = (state >> (2 * site)) & 1
                spin_down = (state >> (2 * site + 1)) & 1
                sz_value = 0.5 * (spin_up - spin_down)
                if sz_value != 0:
                    rows.append(state)
                    cols.append(state)
                    data.append(sz_value)
            elif op_type == 'sx':  # S^x = (S^+ + S^-)/2
                # S^+ component
                if ((state >> (2 * site + 1)) & 1) and not ((state >> (2 * site)) & 1):
                    new_state = state ^ (3 << (2 * site))  # Flip spins
                    rows.append(new_state)
                    cols.append(state)
                    data.append(0.5)
                # S^- component
                if ((state >> (2 * site)) & 1) and not ((state >> (2 * site + 1)) & 1):
                    new_state = state ^ (3 << (2 * site))  # Flip spins
                    rows.append(new_state)
                    cols.append(state)
                    data.append(0.5)
            elif op_type == 'sy':  # S^y = (S^+ - S^-)/2i
                # S^+ component
                if ((state >> (2 * site + 1)) & 1) and not ((state >> (2 * site)) & 1):
                    new_state = state ^ (3 << (2 * site))
                    rows.append(new_state)
                    cols.append(state)
                    data.append(-0.5j)
                # S^- component
                if ((state >> (2 * site)) & 1) and not ((state >> (2 * site + 1)) & 1):
                    new_state = state ^ (3 << (2 * site))
                    rows.append(new_state)
                    cols.append(state)
                    data.append(0.5j)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.hilbert_dim, self.hilbert_dim))
    def build_hamiltonian(self,
                         terms: List[Dict],
                         basis_states: Optional[List[int]] = None) -> sparse.csr_matrix:
        """
        Build many-body Hamiltonian from list of terms.
        Parameters
        ----------
        terms : list
            List of Hamiltonian terms, each containing:
            - 'type': term type ('hopping', 'interaction', 'onsite')
            - 'sites': involved sites
            - 'strength': coupling strength
            - Additional parameters specific to term type
        basis_states : list, optional
            Restricted basis states
        Returns
        -------
        csr_matrix
            Sparse Hamiltonian matrix
        """
        if basis_states is None:
            basis_states = list(range(self.hilbert_dim))
        dim = len(basis_states)
        H = sparse.csr_matrix((dim, dim), dtype=complex)
        print(f"🔧 Building Hamiltonian with {len(terms)} terms...")
        for i, term in enumerate(terms):
            print(f"   Adding term {i+1}/{len(terms)}: {term['type']}")
            if term['type'] == 'hopping':
                H += self._build_hopping_term(term, basis_states)
            elif term['type'] == 'interaction':
                H += self._build_interaction_term(term, basis_states)
            elif term['type'] == 'onsite':
                H += self._build_onsite_term(term, basis_states)
            elif term['type'] == 'magnetic':
                H += self._build_magnetic_term(term, basis_states)
            else:
                raise ValueError(f"Unknown term type: {term['type']}")
        self.hamiltonian = H
        print(f"✅ Hamiltonian built: {dim}×{dim} sparse matrix")
        return H
    def _build_hopping_term(self, term: Dict, basis_states: List[int]) -> sparse.csr_matrix:
        """Build hopping term: t(c†ᵢcⱼ + c†ⱼcᵢ)."""
        i, j = term['sites']
        t = term['strength']
        spin = term.get('spin', 'up')
        # c†ᵢcⱼ
        cdag_i = self.create_operator('cdag', i, spin)
        c_j = self.create_operator('c', j, spin)
        hop_term = t * cdag_i @ c_j
        # Hermitian conjugate
        if i != j:
            hop_term += hop_term.H
        # Restrict to basis states if needed
        if len(basis_states) < self.hilbert_dim:
            hop_term = self._restrict_operator(hop_term, basis_states)
        return hop_term
    def _build_interaction_term(self, term: Dict, basis_states: List[int]) -> sparse.csr_matrix:
        """Build interaction term: U nᵢ↑nᵢ↓."""
        site = term['sites'][0]
        U = term['strength']
        n_up = self.create_operator('n', site, 'up')
        n_down = self.create_operator('n', site, 'down')
        int_term = U * n_up @ n_down
        if len(basis_states) < self.hilbert_dim:
            int_term = self._restrict_operator(int_term, basis_states)
        return int_term
    def _build_onsite_term(self, term: Dict, basis_states: List[int]) -> sparse.csr_matrix:
        """Build onsite energy term: ε nᵢ."""
        site = term['sites'][0]
        epsilon = term['strength']
        spin = term.get('spin', 'up')
        n_i = self.create_operator('n', site, spin)
        onsite_term = epsilon * n_i
        if len(basis_states) < self.hilbert_dim:
            onsite_term = self._restrict_operator(onsite_term, basis_states)
        return onsite_term
    def _build_magnetic_term(self, term: Dict, basis_states: List[int]) -> sparse.csr_matrix:
        """Build magnetic exchange term: J SᵢᵃSⱼᵃ."""
        i, j = term['sites']
        J = term['strength']
        component = term.get('component', 'z')
        if component == 'z':
            Si = self.create_operator('sz', i)
            Sj = self.create_operator('sz', j)
        elif component == 'x':
            Si = self.create_operator('sx', i)
            Sj = self.create_operator('sx', j)
        elif component == 'y':
            Si = self.create_operator('sy', i)
            Sj = self.create_operator('sy', j)
        else:
            raise ValueError(f"Unknown spin component: {component}")
        mag_term = J * Si @ Sj
        if len(basis_states) < self.hilbert_dim:
            mag_term = self._restrict_operator(mag_term, basis_states)
        return mag_term
    def _restrict_operator(self, op: sparse.csr_matrix, basis_states: List[int]) -> sparse.csr_matrix:
        """Restrict operator to subset of basis states."""
        # Create mapping from full basis to restricted basis
        basis_map = {state: i for i, state in enumerate(basis_states)}
        # Extract submatrix
        restricted_op = op[basis_states, :][:, basis_states]
        return restricted_op
    def diagonalize(self,
                   n_eigenvalues: int = 6,
                   which: str = 'SA',
                   return_eigenvectors: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Diagonalize the Hamiltonian using sparse eigenvalue solvers.
        Parameters
        ----------
        n_eigenvalues : int, default 6
            Number of eigenvalues to compute
        which : str, default 'SA'
            Which eigenvalues to find ('SA'=smallest algebraic, 'LA'=largest)
        return_eigenvectors : bool, default True
            Whether to return eigenvectors
        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues
        eigenvectors : ndarray, optional
            Eigenvectors (if requested)
        """
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian() first.")
        print(f"🔍 Diagonalizing {self.hamiltonian.shape[0]}×{self.hamiltonian.shape[1]} Hamiltonian...")
        print(f"   Computing {n_eigenvalues} eigenvalues ({which})")
        start_time = time.time()
        try:
            if return_eigenvectors:
                eigenvals, eigenvecs = sparse_linalg.eigsh(
                    self.hamiltonian,
                    k=n_eigenvalues,
                    which=which,
                    return_eigenvectors=True
                )
                # Sort eigenvalues and eigenvectors
                idx = np.argsort(eigenvals)
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                self.eigenvalues = eigenvals
                self.eigenvectors = eigenvecs
                self.ground_energy = eigenvals[0]
                self.ground_state = eigenvecs[:, 0]
                elapsed = time.time() - start_time
                print(f"✅ Diagonalization completed in {elapsed:.2f}s")
                print(f"   Ground state energy: {self.ground_energy:.6f}")
                return eigenvals, eigenvecs
            else:
                eigenvals = sparse_linalg.eigsh(
                    self.hamiltonian,
                    k=n_eigenvalues,
                    which=which,
                    return_eigenvectors=False
                )
                eigenvals = np.sort(eigenvals)
                self.eigenvalues = eigenvals
                self.ground_energy = eigenvals[0]
                elapsed = time.time() - start_time
                print(f"✅ Diagonalization completed in {elapsed:.2f}s")
                print(f"   Ground state energy: {self.ground_energy:.6f}")
                return eigenvals, None
        except Exception as e:
            print(f"❌ Diagonalization failed: {e}")
            raise
    def compute_expectation_value(self,
                                operator: Union[sparse.csr_matrix, str],
                                state: Optional[np.ndarray] = None) -> complex:
        """
        Compute expectation value of operator in given state.
        Parameters
        ----------
        operator : csr_matrix or str
            Operator matrix or operator name
        state : ndarray, optional
            Quantum state (uses ground state if not provided)
        Returns
        -------
        complex
            Expectation value
        """
        if state is None:
            if self.ground_state is None:
                raise ValueError("No ground state available. Run diagonalize() first.")
            state = self.ground_state
        if isinstance(operator, str):
            # Build operator from name (placeholder)
            raise NotImplementedError("Operator building from string not implemented")
        exp_val = np.conj(state) @ (operator @ state)
        return exp_val
    def correlation_function(self,
                           op1: str, site1: int,
                           op2: str, site2: int,
                           state: Optional[np.ndarray] = None) -> complex:
        """
        Compute two-point correlation function ⟨O₁(i)O₂(j)⟩.
        Parameters
        ----------
        op1, op2 : str
            Operator types
        site1, site2 : int
            Site indices
        state : ndarray, optional
            Quantum state
        Returns
        -------
        complex
            Correlation function value
        """
        if state is None:
            state = self.ground_state
        # Create operators
        O1 = self.create_operator(op1, site1)
        O2 = self.create_operator(op2, site2)
        # Compute ⟨O₁O₂⟩
        corr = np.conj(state) @ (O1 @ O2 @ state)
        return corr
    def plot_spectrum(self,
                     n_levels: Optional[int] = None,
                     energy_shift: bool = True,
                     output_dir: Optional[Path] = None) -> plt.Figure:
        """
        Plot energy spectrum with Berkeley styling.
        Parameters
        ----------
        n_levels : int, optional
            Number of energy levels to plot
        energy_shift : bool, default True
            Shift energies relative to ground state
        output_dir : Path, optional
            Directory to save figure
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if self.eigenvalues is None:
            raise ValueError("No eigenvalues available. Run diagonalize() first.")
        if n_levels is None:
            n_levels = len(self.eigenvalues)
        else:
            n_levels = min(n_levels, len(self.eigenvalues))
        energies = self.eigenvalues[:n_levels]
        if energy_shift:
            energies = energies - self.ground_energy
        # Create Berkeley-styled plot
        berkeley_plot = BerkeleyPlot(figsize=(10, 8))
        fig, ax = berkeley_plot.create_figure()
        # Plot energy levels
        for i, E in enumerate(energies):
            ax.axhline(y=E, color=BERKELEY_BLUE, linewidth=2, alpha=0.8)
            ax.text(0.02, E, f'E_{i}', fontsize=12, va='center',
                   color=BERKELEY_BLUE, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylabel('Energy' + (' - E₀' if energy_shift else ''))
        ax.set_title('Many-Body Energy Spectrum', fontweight='bold', color=BERKELEY_BLUE)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)
        # Add ground state info
        info_text = f'Ground State Energy: {self.ground_energy:.6f}\n'
        info_text += f'Hilbert Space Dimension: {self.hamiltonian.shape[0]}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
               ha='right', va='bottom', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # Save if requested
        if output_dir:
            fig.savefig(output_dir / "many_body_spectrum.png", dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {output_dir}/many_body_spectrum.png")
        return fig
# Utility functions for common many-body problems
def hubbard_chain_terms(n_sites: int, t: float, U: float, mu: float = 0.0) -> List[Dict]:
    """
    Generate Hamiltonian terms for 1D Hubbard model.
    H = -t∑ᵢσ(c†ᵢσcᵢ₊₁σ + h.c.) + U∑ᵢnᵢ↑nᵢ↓ - μ∑ᵢσnᵢσ
    """
    terms = []
    # Hopping terms
    for i in range(n_sites - 1):
        for spin in ['up', 'down']:
            terms.append({
                'type': 'hopping',
                'sites': [i, i + 1],
                'strength': -t,
                'spin': spin
            })
    # Interaction terms
    for i in range(n_sites):
        terms.append({
            'type': 'interaction',
            'sites': [i],
            'strength': U
        })
    # Chemical potential
    if mu != 0.0:
        for i in range(n_sites):
            for spin in ['up', 'down']:
                terms.append({
                    'type': 'onsite',
                    'sites': [i],
                    'strength': -mu,
                    'spin': spin
                })
    return terms
def heisenberg_chain_terms(n_sites: int, J: float, h: float = 0.0) -> List[Dict]:
    """
    Generate Hamiltonian terms for 1D Heisenberg model.
    H = J∑ᵢ(SᵢˣSᵢ₊₁ˣ + SᵢʸSᵢ₊₁ʸ + SᵢᶻSᵢ₊₁ᶻ) - h∑ᵢSᵢᶻ
    """
    terms = []
    # Exchange terms
    for i in range(n_sites - 1):
        for component in ['x', 'y', 'z']:
            terms.append({
                'type': 'magnetic',
                'sites': [i, i + 1],
                'strength': J,
                'component': component
            })
    # Magnetic field
    if h != 0.0:
        for i in range(n_sites):
            terms.append({
                'type': 'magnetic',
                'sites': [i, i],  # Single-site term
                'strength': -h,
                'component': 'z'
            })
    return terms
def solve_many_body_system(terms: List[Dict],
                          n_sites: int,
                          n_eigenvalues: int = 6,
                          particle_number: Optional[int] = None) -> ExactDiagonalization:
    """
    Solve many-body system using exact diagonalization.
    Parameters
    ----------
    terms : list
        Hamiltonian terms
    n_sites : int
        Number of sites
    n_eigenvalues : int, default 6
        Number of eigenvalues to compute
    particle_number : int, optional
        Fixed particle number
    Returns
    -------
    ExactDiagonalization
        Solved system
    """
    # Create solver
    ed = ExactDiagonalization(n_sites)
    # Build basis (with particle number conservation if specified)
    basis_states = ed.build_basis_states(particle_number)
    print(f"📊 Hilbert space dimension: {len(basis_states)}")
    # Build and diagonalize Hamiltonian
    H = ed.build_hamiltonian(terms, basis_states)
    eigenvals, eigenvecs = ed.diagonalize(n_eigenvalues, return_eigenvectors=True)
    return ed
if __name__ == "__main__":
    # Example usage: 4-site Hubbard model
    print("🔬 Exact Diagonalization Example: 4-site Hubbard Model")
    # Parameters
    n_sites = 4
    t = 1.0  # Hopping
    U = 2.0  # Interaction
    mu = U / 2  # Half-filling
    # Generate Hamiltonian terms
    terms = hubbard_chain_terms(n_sites, t, U, mu)
    # Solve system
    ed = solve_many_body_system(terms, n_sites, n_eigenvalues=8, particle_number=n_sites)
    # Plot spectrum
    fig = ed.plot_spectrum()
    plt.show()
    # Compute some observables
    if ed.ground_state is not None:
        print(f"\nGround State Properties:")
        print(f"  Energy: {ed.ground_energy:.6f}")
        # Double occupancy
        double_occ = 0
        for i in range(n_sites):
            n_up = ed.create_operator('n', i, 'up')
            n_down = ed.create_operator('n', i, 'down')
            double_occ += ed.compute_expectation_value(n_up @ n_down).real
        print(f"  Double occupancy: {double_occ:.4f}")
        print(f"  Double occupancy per site: {double_occ/n_sites:.4f}")