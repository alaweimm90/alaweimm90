"""
Analysis tools for magnetization data from OOMMF/MuMax3 simulations.

Includes domain detection, energy calculations, and topological analysis.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

from ..core.constants import PHYSICAL_CONSTANTS
from ..parsers import OOMMFParser, MuMax3Parser


class MagnetizationAnalyzer:
    """
    Analyzes magnetization data from OVF files.
    
    Detects domains, calculates energies, and finds topological features
    like vortices and skyrmions.
    
    Example:
        >>> analyzer = MagnetizationAnalyzer()
        >>> results = analyzer.analyze_ovf_file('magnetization.ovf')
        >>> analyzer.plot_magnetization_map(results)
    """
    
    def __init__(self, material_params: Optional[Dict[str, float]] = None):
        """
        Initialize magnetization analyzer.
        
        Args:
            material_params: Material parameters dictionary
        """
        self.material_params = material_params or self._get_default_material_params()
        self.oommf_parser = OOMMFParser(verbose=False)
        self.mumax3_parser = MuMax3Parser(verbose=False)
        
    def _get_default_material_params(self) -> Dict[str, float]:
        """Get default material parameters for permalloy."""
        return {
            'Ms': 8.6e5,           # Saturation magnetization (A/m)
            'A': 1.3e-11,          # Exchange constant (J/m)
            'K1': 0.0,             # Anisotropy constant (J/m³)
            'gamma': PHYSICAL_CONSTANTS['gamma_e'],  # Gyromagnetic ratio
            'alpha': 0.008         # Gilbert damping
        }
    
    def analyze_ovf_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze OVF magnetization file.
        
        Args:
            filepath: Path to OVF file
            
        Returns:
            Dict with domain_analysis, energy_analysis, spatial_analysis,
            and topological_analysis results.
        """
        # Parse magnetization data
        data = self.oommf_parser.parse_ovf(filepath)
        magnetization = data['magnetization']
        coordinates = data['coordinates']
        metadata = data['metadata']
        
        results = {
            'filepath': str(filepath),
            'metadata': metadata,
            'domain_analysis': self.analyze_domains(magnetization, coordinates),
            'energy_analysis': self.calculate_energy_densities(magnetization, coordinates),
            'spatial_analysis': self.spatial_statistics(magnetization),
            'topological_analysis': self.analyze_topology(magnetization),
            'texture_analysis': self.analyze_texture(magnetization)
        }
        
        return results
    
    def analyze_domains(self, magnetization: Dict[str, np.ndarray], 
                       coordinates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Find magnetic domains via clustering of magnetization directions.
        
        Args:
            magnetization: Magnetization data (mx, my, mz, theta, phi)
            coordinates: Spatial coordinate arrays
            
        Returns:
            Dict with num_domains, domain_labels, domain_walls, statistics
        """
        mx, my, mz = magnetization['mx'], magnetization['my'], magnetization['mz']
        
        # Calculate local magnetization direction
        theta = magnetization['theta']  # Polar angle
        phi = magnetization['phi']      # Azimuthal angle
        
        # Detect domains using clustering of magnetization directions
        domains = self._detect_domains(theta, phi)
        
        # Calculate domain walls
        domain_walls = self._detect_domain_walls(domains)
        
        # Calculate domain statistics
        domain_stats = self._calculate_domain_statistics(domains, magnetization)
        
        results = {
            'num_domains': len(np.unique(domains)) - 1,  # Exclude background
            'domain_labels': domains,
            'domain_walls': domain_walls,
            'domain_statistics': domain_stats,
            'average_domain_size': np.mean([stats['size'] for stats in domain_stats.values()]),
            'domain_wall_density': np.sum(domain_walls) / domain_walls.size
        }
        
        return results
    
    def calculate_energy_densities(self, magnetization: Dict[str, np.ndarray],
                                 coordinates: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate exchange, demagnetization, and anisotropy energies.
        
        Args:
            magnetization: Magnetization components (mx, my, mz)
            coordinates: Spatial coordinates for gradient calculation
            
        Returns:
            Dict with energy densities and totals for each contribution
        """
        mx, my, mz = magnetization['mx'], magnetization['my'], magnetization['mz']
        
        # Exchange energy density
        exchange_energy = self._calculate_exchange_energy(mx, my, mz, coordinates)
        
        # Demagnetization energy (approximation)
        demag_energy = self._calculate_demagnetization_energy(mx, my, mz)
        
        # Anisotropy energy
        anisotropy_energy = self._calculate_anisotropy_energy(mx, my, mz)
        
        # Total energy
        total_energy = exchange_energy + demag_energy + anisotropy_energy
        
        results = {
            'exchange_energy': {
                'density': exchange_energy,
                'total': np.sum(exchange_energy),
                'average': np.mean(exchange_energy)
            },
            'demagnetization_energy': {
                'density': demag_energy,
                'total': np.sum(demag_energy),
                'average': np.mean(demag_energy)
            },
            'anisotropy_energy': {
                'density': anisotropy_energy,
                'total': np.sum(anisotropy_energy),
                'average': np.mean(anisotropy_energy)
            },
            'total_energy': {
                'density': total_energy,
                'total': np.sum(total_energy),
                'average': np.mean(total_energy)
            }
        }
        
        return results
    
    def spatial_statistics(self, magnetization: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate spatial statistics of magnetization.
        
        Args:
            magnetization: Magnetization data dictionary
            
        Returns:
            Spatial statistics
        """
        mx, my, mz = magnetization['mx'], magnetization['my'], magnetization['mz']
        magnitude = magnetization['magnitude']
        
        # Basic statistics
        stats = {
            'mx_stats': self._calculate_field_statistics(mx),
            'my_stats': self._calculate_field_statistics(my),
            'mz_stats': self._calculate_field_statistics(mz),
            'magnitude_stats': self._calculate_field_statistics(magnitude)
        }
        
        # Correlation functions
        correlations = self._calculate_correlations(mx, my, mz)
        
        # Gradient analysis
        gradients = self._calculate_gradients(mx, my, mz)
        
        results = {
            'field_statistics': stats,
            'correlations': correlations,
            'gradients': gradients,
            'uniformity_index': self._calculate_uniformity_index(magnitude),
            'coherence_length': self._estimate_coherence_length(mx, my, mz)
        }
        
        return results
    
    def analyze_topology(self, magnetization: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze topological features of magnetization.
        
        Args:
            magnetization: Magnetization data dictionary
            
        Returns:
            Topological analysis results
        """
        mx, my, mz = magnetization['mx'], magnetization['my'], magnetization['mz']
        
        # Detect vortices and antivortices
        vortices = self._detect_vortices(mx, my)
        
        # Calculate topological charge density
        topological_charge = self._calculate_topological_charge(mx, my, mz)
        
        # Detect skyrmions (simplified)
        skyrmions = self._detect_skyrmions(mx, my, mz)
        
        results = {
            'vortices': vortices,
            'total_topological_charge': np.sum(topological_charge),
            'topological_charge_density': topological_charge,
            'skyrmions': skyrmions,
            'num_topological_defects': len(vortices['positions'])
        }
        
        return results
    
    def analyze_texture(self, magnetization: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze magnetization texture and patterns.
        
        Args:
            magnetization: Magnetization data dictionary
            
        Returns:
            Texture analysis results
        """
        mx, my, mz = magnetization['mx'], magnetization['my'], magnetization['mz']
        
        # Calculate local texture metrics
        texture_metrics = self._calculate_texture_metrics(mx, my, mz)
        
        # Pattern recognition
        patterns = self._recognize_patterns(mx, my, mz)
        
        results = {
            'texture_metrics': texture_metrics,
            'pattern_analysis': patterns,
            'texture_complexity': self._calculate_texture_complexity(mx, my, mz)
        }
        
        return results
    
    def _detect_domains(self, theta: np.ndarray, phi: np.ndarray, 
                       min_domain_size: int = 100) -> np.ndarray:
        """Cluster similar magnetization directions into domains."""
        # Flatten arrays for clustering
        theta_flat = theta.flatten()
        phi_flat = phi.flatten()
        
        # Create feature vectors (unit vectors on sphere)
        features = np.column_stack([
            np.sin(theta_flat) * np.cos(phi_flat),
            np.sin(theta_flat) * np.sin(phi_flat),
            np.cos(theta_flat)
        ])
        
        # Use hierarchical clustering
        try:
            distances = cdist(features[::10], features[::10])  # Subsample for speed
            linkage_matrix = linkage(distances, method='ward')
            n_clusters = min(10, len(np.unique(theta_flat)) // 100)  # Adaptive number
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Map back to full array
            full_labels = np.zeros_like(theta_flat)
            full_labels[::10] = cluster_labels
            
            # Fill in missing labels with nearest neighbor
            for i in range(len(full_labels)):
                if i % 10 != 0:
                    full_labels[i] = full_labels[(i // 10) * 10]
            
            domains = full_labels.reshape(theta.shape)
            
        except Exception:
            # Fallback: simple threshold-based domains
            domains = np.zeros_like(theta)
            domains[mz > 0.5] = 1  # Up domains
            domains[mz < -0.5] = 2  # Down domains
        
        # Remove small domains
        for label in np.unique(domains):
            if np.sum(domains == label) < min_domain_size:
                domains[domains == label] = 0
        
        return domains
    
    def _detect_domain_walls(self, domains: np.ndarray) -> np.ndarray:
        """Detect domain walls as boundaries between domains."""
        # Calculate gradients to find boundaries
        grad_x = np.abs(np.gradient(domains.astype(float), axis=-1))
        grad_y = np.abs(np.gradient(domains.astype(float), axis=-2))
        
        if domains.ndim == 3:
            grad_z = np.abs(np.gradient(domains.astype(float), axis=-3))
            domain_walls = (grad_x + grad_y + grad_z) > 0.1
        else:
            domain_walls = (grad_x + grad_y) > 0.1
        
        return domain_walls.astype(bool)
    
    def _calculate_domain_statistics(self, domains: np.ndarray, 
                                   magnetization: Dict[str, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """Calculate statistics for each domain."""
        stats = {}
        
        for label in np.unique(domains):
            if label == 0:  # Skip background
                continue
                
            mask = domains == label
            size = np.sum(mask)
            
            if size > 0:
                mx_domain = magnetization['mx'][mask]
                my_domain = magnetization['my'][mask]
                mz_domain = magnetization['mz'][mask]
                
                stats[int(label)] = {
                    'size': size,
                    'avg_mx': np.mean(mx_domain),
                    'avg_my': np.mean(my_domain),
                    'avg_mz': np.mean(mz_domain),
                    'std_mx': np.std(mx_domain),
                    'std_my': np.std(my_domain),
                    'std_mz': np.std(mz_domain),
                    'uniformity': 1.0 - (np.std(mx_domain) + np.std(my_domain) + np.std(mz_domain)) / 3.0
                }
        
        return stats
    
    def _calculate_exchange_energy(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray,
                                 coordinates: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate exchange energy density."""
        A = self.material_params['A']
        
        # Calculate gradients
        grad_mx = np.gradient(mx)
        grad_my = np.gradient(my)
        grad_mz = np.gradient(mz)
        
        # Exchange energy density: A * |∇m|²
        exchange_energy = np.zeros_like(mx)
        
        for i, (gx, gy, gz) in enumerate(zip(grad_mx, grad_my, grad_mz)):
            exchange_energy += A * (gx**2 + gy**2 + gz**2)
        
        return exchange_energy
    
    def _calculate_demagnetization_energy(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
        """Calculate approximate demagnetization energy density."""
        # Simplified approximation using shape demagnetization factors
        mu0 = PHYSICAL_CONSTANTS['mu_0']
        Ms = self.material_params['Ms']
        
        # Approximate demagnetization factors for thin film (Nx ≈ Ny ≈ 0, Nz ≈ 1)
        Nx, Ny, Nz = 0.0, 0.0, 1.0
        
        demag_energy = 0.5 * mu0 * Ms**2 * (Nx * mx**2 + Ny * my**2 + Nz * mz**2)
        
        return demag_energy
    
    def _calculate_anisotropy_energy(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
        """Calculate uniaxial anisotropy energy density."""
        K1 = self.material_params['K1']
        
        # Assume easy axis along z
        anisotropy_energy = K1 * (1 - mz**2)
        
        return anisotropy_energy
    
    def _calculate_field_statistics(self, field: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics for a field."""
        return {
            'mean': float(np.mean(field)),
            'std': float(np.std(field)),
            'min': float(np.min(field)),
            'max': float(np.max(field)),
            'median': float(np.median(field)),
            'rms': float(np.sqrt(np.mean(field**2)))
        }
    
    def _calculate_correlations(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Dict[str, float]:
        """Calculate correlation coefficients between components."""
        mx_flat = mx.flatten()
        my_flat = my.flatten()
        mz_flat = mz.flatten()
        
        corr_xy = float(np.corrcoef(mx_flat, my_flat)[0, 1])
        corr_xz = float(np.corrcoef(mx_flat, mz_flat)[0, 1])
        corr_yz = float(np.corrcoef(my_flat, mz_flat)[0, 1])
        
        return {
            'mx_my_correlation': corr_xy if not np.isnan(corr_xy) else 0.0,
            'mx_mz_correlation': corr_xz if not np.isnan(corr_xz) else 0.0,
            'my_mz_correlation': corr_yz if not np.isnan(corr_yz) else 0.0
        }
    
    def _calculate_gradients(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Dict[str, Any]:
        """Calculate gradient statistics."""
        grad_mx = np.gradient(mx)
        grad_my = np.gradient(my)
        grad_mz = np.gradient(mz)
        
        # Calculate gradient magnitudes
        grad_mag_mx = np.sqrt(sum(g**2 for g in grad_mx))
        grad_mag_my = np.sqrt(sum(g**2 for g in grad_my))
        grad_mag_mz = np.sqrt(sum(g**2 for g in grad_mz))
        
        return {
            'mx_gradient_stats': self._calculate_field_statistics(grad_mag_mx),
            'my_gradient_stats': self._calculate_field_statistics(grad_mag_my),
            'mz_gradient_stats': self._calculate_field_statistics(grad_mag_mz),
            'total_gradient_magnitude': float(np.mean(grad_mag_mx + grad_mag_my + grad_mag_mz))
        }
    
    def _calculate_uniformity_index(self, magnitude: np.ndarray) -> float:
        """Calculate uniformity index (1 - coefficient of variation)."""
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        
        if mean_mag > 0:
            cv = std_mag / mean_mag
            return float(1.0 - cv)
        else:
            return 0.0
    
    def _estimate_coherence_length(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> float:
        """Estimate magnetic coherence length."""
        # Simple approximation based on correlation decay
        try:
            # Calculate autocorrelation for mx along one direction
            if mx.ndim >= 2:
                line = mx[mx.shape[0]//2, :]  # Middle line
                autocorr = np.correlate(line, line, mode='full')
                autocorr = autocorr[autocorr.size//2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Find where correlation drops to 1/e
                coherence_idx = np.where(autocorr < 1/np.e)[0]
                if len(coherence_idx) > 0:
                    return float(coherence_idx[0])
            
            return float(min(mx.shape) // 4)  # Default estimate
            
        except Exception:
            return float(min(mx.shape) // 4)
    
    def _detect_vortices(self, mx: np.ndarray, my: np.ndarray) -> Dict[str, Any]:
        """Detect magnetic vortices using winding number."""
        if mx.ndim < 2:
            return {'positions': [], 'charges': []}
        
        # Take 2D slice if 3D
        if mx.ndim == 3:
            mx_2d = mx[mx.shape[0]//2, :, :]
            my_2d = my[my.shape[0]//2, :, :]
        else:
            mx_2d = mx
            my_2d = my
        
        vortex_positions = []
        vortex_charges = []
        
        # Scan for vortices using a simplified approach
        h, w = mx_2d.shape
        for i in range(2, h-2):
            for j in range(2, w-2):
                # Calculate winding number in a small loop
                angles = []
                for di, dj in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                    mx_val = mx_2d[i + di, j + dj]
                    my_val = my_2d[i + di, j + dj]
                    angle = np.arctan2(my_val, mx_val)
                    angles.append(angle)
                
                # Calculate total winding
                winding = 0
                for k in range(len(angles)):
                    angle_diff = angles[(k+1) % len(angles)] - angles[k]
                    # Handle angle wrapping
                    if angle_diff > np.pi:
                        angle_diff -= 2*np.pi
                    elif angle_diff < -np.pi:
                        angle_diff += 2*np.pi
                    winding += angle_diff
                
                winding_number = winding / (2*np.pi)
                
                # Check if this is a vortex (|winding| > 0.5)
                if abs(winding_number) > 0.5:
                    vortex_positions.append((i, j))
                    vortex_charges.append(int(np.sign(winding_number)))
        
        return {
            'positions': vortex_positions,
            'charges': vortex_charges
        }
    
    def _calculate_topological_charge(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
        """Calculate topological charge density."""
        # Simplified calculation for 2D case
        if mx.ndim == 3:
            mx_2d = mx[mx.shape[0]//2, :, :]
            my_2d = my[my.shape[0]//2, :, :]
        else:
            mx_2d = mx
            my_2d = my
        
        # Calculate derivatives
        dmx_dx = np.gradient(mx_2d, axis=1)
        dmx_dy = np.gradient(mx_2d, axis=0)
        dmy_dx = np.gradient(my_2d, axis=1)
        dmy_dy = np.gradient(my_2d, axis=0)
        
        # Topological charge density
        charge_density = (dmx_dx * dmy_dy - dmx_dy * dmy_dx) / (4 * np.pi)
        
        return charge_density
    
    def _detect_skyrmions(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Dict[str, Any]:
        """Detect skyrmion-like structures (simplified)."""
        # Look for regions where mz changes sign rapidly while mx, my rotate
        if mx.ndim == 3:
            mz_2d = mz[mz.shape[0]//2, :, :]
        else:
            mz_2d = mz
        
        # Find regions with large mz gradients
        grad_mz = np.gradient(mz_2d)
        grad_magnitude = np.sqrt(grad_mz[0]**2 + grad_mz[1]**2)
        
        # Threshold for potential skyrmions
        threshold = np.mean(grad_magnitude) + 2*np.std(grad_magnitude)
        skyrmion_candidates = grad_magnitude > threshold
        
        # Find connected components
        labeled, num_features = ndimage.label(skyrmion_candidates)
        
        skyrmions = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            if np.sum(mask) > 10:  # Minimum size filter
                center = ndimage.center_of_mass(mask)
                skyrmions.append({
                    'center': center,
                    'size': np.sum(mask)
                })
        
        return {
            'skyrmions': skyrmions,
            'num_skyrmions': len(skyrmions)
        }
    
    def _calculate_texture_metrics(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Dict[str, float]:
        """Calculate texture complexity metrics."""
        # Calculate local variations
        grad_mx = np.gradient(mx)
        grad_my = np.gradient(my)
        grad_mz = np.gradient(mz)
        
        # Texture roughness
        roughness = np.mean([np.std(g) for g in grad_mx + grad_my + grad_mz])
        
        # Texture directionality
        angles = np.arctan2(my, mx)
        angle_variation = np.std(angles)
        
        return {
            'roughness': float(roughness),
            'directionality': float(angle_variation),
            'complexity_index': float(roughness * angle_variation)
        }
    
    def _recognize_patterns(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> Dict[str, Any]:
        """Recognize common magnetization patterns."""
        patterns = {
            'uniform_state': self._check_uniform_state(mx, my, mz),
            'vortex_state': self._check_vortex_state(mx, my, mz),
            'stripe_domains': self._check_stripe_domains(mx, my, mz),
            'flux_closure': self._check_flux_closure(mx, my, mz)
        }
        
        return patterns
    
    def _check_uniform_state(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> bool:
        """Check if magnetization is in uniform state."""
        # Calculate standard deviations
        std_mx = np.std(mx)
        std_my = np.std(my)
        std_mz = np.std(mz)
        
        # Threshold for uniformity
        threshold = 0.1
        return std_mx < threshold and std_my < threshold and std_mz < threshold
    
    def _check_vortex_state(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> bool:
        """Check if magnetization contains vortex structures."""
        vortices = self._detect_vortices(mx, my)
        return len(vortices['positions']) > 0
    
    def _check_stripe_domains(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> bool:
        """Check for stripe domain patterns."""
        # Look for periodic variations in one direction
        if mz.ndim >= 2:
            # Check for periodicity along one axis
            line = mz[mz.shape[0]//2, :] if mz.ndim == 2 else mz[mz.shape[0]//2, mz.shape[1]//2, :]
            fft = np.fft.fft(line)
            power = np.abs(fft)**2
            
            # Look for dominant frequency components
            max_power = np.max(power[1:len(power)//2])  # Exclude DC component
            mean_power = np.mean(power[1:len(power)//2])
            
            return max_power > 5 * mean_power
        
        return False
    
    def _check_flux_closure(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> bool:
        """Check for flux closure patterns."""
        # Calculate divergence of magnetization
        div_m = np.gradient(mx, axis=-1) + np.gradient(my, axis=-2)
        if mx.ndim == 3:
            div_m += np.gradient(mz, axis=-3)
        
        # Flux closure should have low divergence
        mean_div = np.mean(np.abs(div_m))
        return mean_div < 0.1
    
    def _calculate_texture_complexity(self, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> float:
        """Calculate overall texture complexity index."""
        # Combine multiple metrics
        metrics = self._calculate_texture_metrics(mx, my, mz)
        
        # Normalize and combine
        complexity = 0.5 * metrics['roughness'] + 0.3 * metrics['directionality']
        
        return float(complexity)
    
    def plot_magnetization_map(self, analysis_results: Dict[str, Any], 
                             component: str = 'mz', figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot magnetization component map with analysis overlay.
        
        Args:
            analysis_results: Results from analyze_ovf_file
            component: Magnetization component to plot ('mx', 'my', 'mz')
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Re-parse the file to get magnetization data
        filepath = analysis_results['filepath']
        data = self.oommf_parser.parse_ovf(filepath)
        magnetization = data['magnetization']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot magnetization component
        mag_data = magnetization[component]
        if mag_data.ndim == 3:
            mag_data = mag_data[mag_data.shape[0]//2, :, :]  # Take middle slice
        
        im = ax.imshow(mag_data, cmap='RdBu_r', aspect='equal')
        plt.colorbar(im, ax=ax, label=f'{component.upper()}')
        
        # Overlay domain walls if available
        domain_analysis = analysis_results.get('domain_analysis', {})
        if 'domain_walls' in domain_analysis:
            domain_walls = domain_analysis['domain_walls']
            if domain_walls.ndim == 3:
                domain_walls = domain_walls[domain_walls.shape[0]//2, :, :]
            
            # Overlay domain walls
            ax.contour(domain_walls, levels=[0.5], colors='black', linewidths=1, alpha=0.7)
        
        # Overlay vortices if available
        topo_analysis = analysis_results.get('topological_analysis', {})
        if 'vortices' in topo_analysis:
            vortices = topo_analysis['vortices']
            for pos, charge in zip(vortices['positions'], vortices['charges']):
                color = 'red' if charge > 0 else 'blue'
                ax.plot(pos[1], pos[0], 'o', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=2)
        
        ax.set_title(f'Magnetization {component.upper()} with Analysis Overlay')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add analysis summary as text
        analysis_text = self._format_analysis_summary(analysis_results)
        ax.text(0.02, 0.98, analysis_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _format_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Format analysis results for display."""
        lines = []
        
        # Domain analysis
        domain_analysis = results.get('domain_analysis', {})
        if domain_analysis:
            lines.append(f"Domains: {domain_analysis.get('num_domains', 0)}")
            lines.append(f"Avg size: {domain_analysis.get('average_domain_size', 0):.0f} cells")
        
        # Topological analysis
        topo_analysis = results.get('topological_analysis', {})
        if topo_analysis:
            lines.append(f"Vortices: {topo_analysis.get('num_topological_defects', 0)}")
            lines.append(f"Topo charge: {topo_analysis.get('total_topological_charge', 0):.2f}")
        
        # Energy analysis
        energy_analysis = results.get('energy_analysis', {})
        if energy_analysis:
            total_energy = energy_analysis.get('total_energy', {}).get('total', 0)
            lines.append(f"Total energy: {total_energy:.2e} J")
        
        return '\n'.join(lines)