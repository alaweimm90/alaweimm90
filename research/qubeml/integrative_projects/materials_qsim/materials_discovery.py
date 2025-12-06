"""
Materials Discovery Pipeline using Machine Learning and Quantum Simulation

This module implements a comprehensive pipeline for discovering new materials
with desired properties using ML models and quantum simulations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Material:
    """Representation of a material with its properties."""
    formula: str
    composition: Dict[str, float]
    structure_params: Dict[str, float]
    properties: Dict[str, float] = field(default_factory=dict)
    features: Optional[np.ndarray] = None
    quantum_features: Optional[np.ndarray] = None


@dataclass
class CrystalStructure:
    """Crystal structure representation."""
    lattice_vectors: np.ndarray
    atom_positions: np.ndarray
    atom_types: List[str]
    space_group: int
    
    def get_structural_features(self) -> np.ndarray:
        """Extract structural features."""
        features = []
        
        a, b, c = np.linalg.norm(self.lattice_vectors, axis=1)
        features.extend([a, b, c])
        
        volume = np.abs(np.linalg.det(self.lattice_vectors))
        features.append(volume)
        
        angles = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            cos_angle = np.dot(self.lattice_vectors[i], self.lattice_vectors[j]) / \
                       (np.linalg.norm(self.lattice_vectors[i]) * np.linalg.norm(self.lattice_vectors[j]))
            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        features.extend(angles)
        
        features.append(self.space_group)
        
        return np.array(features)


class CompositionFeaturizer:
    """Extract features from chemical composition."""
    
    def __init__(self):
        self.element_properties = self._load_element_properties()
        
    def _load_element_properties(self) -> Dict[str, Dict[str, float]]:
        """Load periodic table properties."""
        properties = {
            'H': {'atomic_number': 1, 'atomic_mass': 1.008, 'electronegativity': 2.20, 
                  'atomic_radius': 25, 'ionization_energy': 13.6},
            'Li': {'atomic_number': 3, 'atomic_mass': 6.94, 'electronegativity': 0.98,
                   'atomic_radius': 145, 'ionization_energy': 5.39},
            'C': {'atomic_number': 6, 'atomic_mass': 12.01, 'electronegativity': 2.55,
                  'atomic_radius': 70, 'ionization_energy': 11.26},
            'N': {'atomic_number': 7, 'atomic_mass': 14.01, 'electronegativity': 3.04,
                  'atomic_radius': 65, 'ionization_energy': 14.53},
            'O': {'atomic_number': 8, 'atomic_mass': 16.00, 'electronegativity': 3.44,
                  'atomic_radius': 60, 'ionization_energy': 13.62},
            'Si': {'atomic_number': 14, 'atomic_mass': 28.09, 'electronegativity': 1.90,
                   'atomic_radius': 110, 'ionization_energy': 8.15},
            'Fe': {'atomic_number': 26, 'atomic_mass': 55.85, 'electronegativity': 1.83,
                   'atomic_radius': 140, 'ionization_energy': 7.90},
        }
        return properties
    
    def featurize(self, composition: Dict[str, float]) -> np.ndarray:
        """Extract features from composition."""
        features = []
        
        total_atoms = sum(composition.values())
        fractions = {elem: count/total_atoms for elem, count in composition.items()}
        
        weighted_props = {prop: 0 for prop in ['atomic_number', 'atomic_mass', 
                                               'electronegativity', 'atomic_radius']}
        
        for elem, frac in fractions.items():
            if elem in self.element_properties:
                for prop, value in self.element_properties[elem].items():
                    if prop in weighted_props:
                        weighted_props[prop] += frac * value
        
        features.extend(weighted_props.values())
        
        features.append(len(composition))
        features.append(total_atoms)
        
        electroneg_values = [self.element_properties[elem].get('electronegativity', 0) 
                           for elem in composition if elem in self.element_properties]
        if len(electroneg_values) > 1:
            features.append(np.std(electroneg_values))
        else:
            features.append(0)
        
        return np.array(features)


class PropertyPredictor(nn.Module):
    """Deep neural network for property prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ActiveLearner:
    """Active learning for efficient materials exploration."""
    
    def __init__(self, predictor: Callable, acquisition_func: str = 'ucb'):
        self.predictor = predictor
        self.acquisition_func = acquisition_func
        self.observed_materials = []
        self.observed_properties = []
        
    def acquire(self, candidates: List[Material], n_select: int = 5) -> List[int]:
        """Select materials for evaluation using acquisition function."""
        if self.acquisition_func == 'ucb':
            return self._ucb_acquisition(candidates, n_select)
        elif self.acquisition_func == 'ei':
            return self._ei_acquisition(candidates, n_select)
        elif self.acquisition_func == 'random':
            return np.random.choice(len(candidates), n_select, replace=False).tolist()
        
    def _ucb_acquisition(self, candidates: List[Material], n_select: int) -> List[int]:
        """Upper Confidence Bound acquisition."""
        features = np.array([mat.features for mat in candidates])
        
        if hasattr(self.predictor, 'predict'):
            predictions, uncertainties = self.predictor.predict(features, return_std=True)
            scores = predictions + 2.0 * uncertainties
        else:
            predictions = self.predictor(torch.FloatTensor(features)).detach().numpy()
            scores = predictions.flatten()
            
        return np.argsort(scores)[-n_select:].tolist()
    
    def _ei_acquisition(self, candidates: List[Material], n_select: int) -> List[int]:
        """Expected Improvement acquisition."""
        if len(self.observed_properties) == 0:
            return np.random.choice(len(candidates), n_select, replace=False).tolist()
            
        best_observed = np.max(self.observed_properties)
        features = np.array([mat.features for mat in candidates])
        
        if hasattr(self.predictor, 'predict'):
            predictions, uncertainties = self.predictor.predict(features, return_std=True)
            
            z = (predictions - best_observed) / (uncertainties + 1e-9)
            from scipy.stats import norm
            ei = uncertainties * (z * norm.cdf(z) + norm.pdf(z))
            
            return np.argsort(ei)[-n_select:].tolist()
        else:
            predictions = self.predictor(torch.FloatTensor(features)).detach().numpy()
            scores = predictions.flatten() - best_observed
            return np.argsort(scores)[-n_select:].tolist()
    
    def update(self, materials: List[Material], properties: np.ndarray):
        """Update with new observations."""
        self.observed_materials.extend(materials)
        self.observed_properties.extend(properties.tolist())


class MaterialsOptimizer:
    """Optimize material composition for target properties."""
    
    def __init__(self, predictor: Callable, constraints: Dict[str, Any] = None):
        self.predictor = predictor
        self.constraints = constraints or {}
        
    def optimize_composition(self, initial_composition: Dict[str, float],
                            target_property: str, bounds: Dict[str, Tuple[float, float]],
                            n_iterations: int = 100) -> Dict[str, float]:
        """Optimize composition using evolutionary algorithm."""
        
        elements = list(initial_composition.keys())
        n_elements = len(elements)
        
        def objective(x):
            composition = {elem: x[i] for i, elem in enumerate(elements)}
            
            if not self._check_constraints(composition):
                return -1e10
                
            featurizer = CompositionFeaturizer()
            features = featurizer.featurize(composition)
            
            if isinstance(self.predictor, nn.Module):
                with torch.no_grad():
                    pred = self.predictor(torch.FloatTensor(features.reshape(1, -1)))
                    return -pred.item()
            else:
                return -self.predictor.predict(features.reshape(1, -1))[0]
        
        bounds_list = [(bounds.get(elem, (0, 100))[0], bounds.get(elem, (0, 100))[1]) 
                      for elem in elements]
        
        result = differential_evolution(objective, bounds_list, maxiter=n_iterations,
                                      popsize=15, seed=42)
        
        optimal_composition = {elem: result.x[i] for i, elem in enumerate(elements)}
        
        total = sum(optimal_composition.values())
        optimal_composition = {k: v/total * 100 for k, v in optimal_composition.items()}
        
        return optimal_composition
    
    def _check_constraints(self, composition: Dict[str, float]) -> bool:
        """Check if composition satisfies constraints."""
        if 'max_elements' in self.constraints:
            if len([v for v in composition.values() if v > 0]) > self.constraints['max_elements']:
                return False
                
        if 'forbidden_elements' in self.constraints:
            for elem in self.constraints['forbidden_elements']:
                if elem in composition and composition[elem] > 0:
                    return False
                    
        return True


class MaterialsDiscoveryPipeline:
    """Complete pipeline for materials discovery."""
    
    def __init__(self, target_properties: List[str]):
        self.target_properties = target_properties
        self.featurizer = CompositionFeaturizer()
        self.scaler = StandardScaler()
        self.models = {}
        self.materials_database = []
        
    def load_materials_database(self, materials: List[Material]):
        """Load materials into database."""
        self.materials_database = materials
        
        for mat in self.materials_database:
            if mat.features is None:
                mat.features = self.featurizer.featurize(mat.composition)
                
    def train_models(self, training_materials: List[Material]):
        """Train predictive models for each property."""
        X = np.array([mat.features for mat in training_materials])
        X = self.scaler.fit_transform(X)
        
        for prop in self.target_properties:
            y = np.array([mat.properties.get(prop, 0) for mat in training_materials])
            
            print(f"\nTraining models for {prop}...")
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X, y)
            
            kernel = RBF() + WhiteKernel()
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(X, y)
            
            nn_model = PropertyPredictor(
                input_dim=X.shape[1],
                hidden_dims=[128, 64, 32],
                output_dim=1
            )
            
            self._train_neural_network(nn_model, X, y)
            
            self.models[prop] = {
                'random_forest': rf,
                'gradient_boosting': gb,
                'gaussian_process': gp,
                'neural_network': nn_model
            }
            
    def _train_neural_network(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                             epochs: int = 100):
        """Train neural network model."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
    def predict_properties(self, material: Material) -> Dict[str, float]:
        """Predict properties for a material."""
        if material.features is None:
            material.features = self.featurizer.featurize(material.composition)
            
        features = self.scaler.transform(material.features.reshape(1, -1))
        predictions = {}
        
        for prop in self.target_properties:
            if prop in self.models:
                prop_predictions = []
                
                for model_name, model in self.models[prop].items():
                    if isinstance(model, nn.Module):
                        with torch.no_grad():
                            pred = model(torch.FloatTensor(features)).item()
                    else:
                        pred = model.predict(features)[0]
                    prop_predictions.append(pred)
                    
                predictions[prop] = np.mean(prop_predictions)
                
        return predictions
    
    def discover_materials(self, n_candidates: int = 1000, n_select: int = 10,
                          acquisition: str = 'ucb') -> List[Material]:
        """Discover new materials using active learning."""
        candidate_materials = self._generate_candidates(n_candidates)
        
        for mat in candidate_materials:
            mat.features = self.featurizer.featurize(mat.composition)
            
        results = []
        
        for prop in self.target_properties:
            if prop in self.models:
                learner = ActiveLearner(
                    self.models[prop]['gaussian_process'],
                    acquisition_func=acquisition
                )
                
                selected_indices = learner.acquire(candidate_materials, n_select)
                selected_materials = [candidate_materials[i] for i in selected_indices]
                
                for mat in selected_materials:
                    mat.properties[prop] = self.predict_properties(mat)[prop]
                    
                results.extend(selected_materials)
                
        return results
    
    def _generate_candidates(self, n_candidates: int) -> List[Material]:
        """Generate candidate materials."""
        candidates = []
        elements = ['Li', 'C', 'N', 'O', 'Si', 'Fe']
        
        for _ in range(n_candidates):
            n_elements = np.random.randint(2, 5)
            selected_elements = np.random.choice(elements, n_elements, replace=False)
            
            composition = {}
            remaining = 100.0
            
            for i, elem in enumerate(selected_elements[:-1]):
                amount = np.random.uniform(0, remaining)
                composition[elem] = amount
                remaining -= amount
                
            composition[selected_elements[-1]] = remaining
            
            mat = Material(
                formula=self._composition_to_formula(composition),
                composition=composition,
                structure_params={'lattice_a': np.random.uniform(3, 10),
                                'lattice_b': np.random.uniform(3, 10),
                                'lattice_c': np.random.uniform(3, 10)}
            )
            candidates.append(mat)
            
        return candidates
    
    def _composition_to_formula(self, composition: Dict[str, float]) -> str:
        """Convert composition to chemical formula."""
        formula = ""
        for elem, amount in sorted(composition.items()):
            if amount > 0:
                formula += f"{elem}{amount:.1f}"
        return formula
    
    def optimize_for_property(self, target_property: str, initial_material: Material,
                             n_iterations: int = 50) -> Material:
        """Optimize material for specific property."""
        optimizer = MaterialsOptimizer(
            self.models[target_property]['gradient_boosting'],
            constraints={'max_elements': 4}
        )
        
        bounds = {elem: (0, 100) for elem in initial_material.composition}
        
        optimal_composition = optimizer.optimize_composition(
            initial_material.composition,
            target_property,
            bounds,
            n_iterations
        )
        
        optimized_material = Material(
            formula=self._composition_to_formula(optimal_composition),
            composition=optimal_composition,
            structure_params=initial_material.structure_params.copy()
        )
        
        optimized_material.properties = self.predict_properties(optimized_material)
        
        return optimized_material
    
    def analyze_results(self, materials: List[Material]):
        """Analyze and visualize discovered materials."""
        if not materials:
            print("No materials to analyze.")
            return
            
        df_data = []
        for mat in materials:
            row = {'formula': mat.formula}
            row.update(mat.composition)
            row.update(mat.properties)
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if self.target_properties:
            prop = self.target_properties[0]
            if prop in df.columns:
                axes[0, 0].hist(df[prop], bins=20, edgecolor='black')
                axes[0, 0].set_xlabel(prop)
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title(f'Distribution of {prop}')
        
        composition_cols = [col for col in df.columns if col in ['Li', 'C', 'N', 'O', 'Si', 'Fe']]
        if composition_cols:
            composition_data = df[composition_cols].fillna(0)
            axes[0, 1].bar(composition_cols, composition_data.mean())
            axes[0, 1].set_xlabel('Element')
            axes[0, 1].set_ylabel('Average Composition (%)')
            axes[0, 1].set_title('Average Elemental Composition')
        
        if len(self.target_properties) >= 2:
            prop1, prop2 = self.target_properties[:2]
            if prop1 in df.columns and prop2 in df.columns:
                axes[1, 0].scatter(df[prop1], df[prop2], alpha=0.6)
                axes[1, 0].set_xlabel(prop1)
                axes[1, 0].set_ylabel(prop2)
                axes[1, 0].set_title(f'{prop1} vs {prop2}')
        
        if self.target_properties and composition_cols:
            prop = self.target_properties[0]
            if prop in df.columns:
                correlations = df[composition_cols].corrwith(df[prop])
                axes[1, 1].bar(composition_cols, correlations)
                axes[1, 1].set_xlabel('Element')
                axes[1, 1].set_ylabel(f'Correlation with {prop}')
                axes[1, 1].set_title(f'Element-Property Correlations')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        
        print("\nTop 5 materials by each property:")
        for prop in self.target_properties:
            if prop in df.columns:
                top_materials = df.nlargest(5, prop)[['formula', prop]]
                print(f"\n{prop}:")
                print(top_materials.to_string(index=False))


def demonstrate_materials_discovery():
    """Demonstrate the materials discovery pipeline."""
    
    print("Initializing materials discovery pipeline...")
    pipeline = MaterialsDiscoveryPipeline(
        target_properties=['bandgap', 'formation_energy', 'bulk_modulus']
    )
    
    print("\nGenerating synthetic training materials...")
    training_materials = []
    for i in range(200):
        composition = {}
        elements = np.random.choice(['Li', 'C', 'N', 'O', 'Si', 'Fe'], 
                                  np.random.randint(2, 4), replace=False)
        
        for elem in elements:
            composition[elem] = np.random.uniform(10, 50)
            
        total = sum(composition.values())
        composition = {k: v/total * 100 for k, v in composition.items()}
        
        mat = Material(
            formula=f"Material_{i}",
            composition=composition,
            structure_params={'lattice_a': np.random.uniform(3, 10)},
            properties={
                'bandgap': np.random.uniform(0, 6),
                'formation_energy': np.random.uniform(-5, 2),
                'bulk_modulus': np.random.uniform(50, 300)
            }
        )
        training_materials.append(mat)
    
    print("\nTraining predictive models...")
    pipeline.load_materials_database(training_materials)
    pipeline.train_models(training_materials[:150])
    
    print("\nDiscovering new materials...")
    discovered_materials = pipeline.discover_materials(
        n_candidates=500,
        n_select=20,
        acquisition='ucb'
    )
    
    print(f"\nDiscovered {len(discovered_materials)} promising materials")
    
    if discovered_materials:
        print("\nOptimizing best material for bandgap...")
        best_material = max(discovered_materials, 
                          key=lambda m: m.properties.get('bandgap', 0))
        optimized = pipeline.optimize_for_property('bandgap', best_material)
        
        print(f"Original bandgap: {best_material.properties.get('bandgap', 0):.3f}")
        print(f"Optimized bandgap: {optimized.properties.get('bandgap', 0):.3f}")
    
    print("\nAnalyzing results...")
    pipeline.analyze_results(discovered_materials)
    
    return pipeline, discovered_materials


if __name__ == "__main__":
    pipeline, materials = demonstrate_materials_discovery()