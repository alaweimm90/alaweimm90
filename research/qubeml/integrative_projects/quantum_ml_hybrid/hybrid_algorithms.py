"""
Quantum-Classical Hybrid Algorithms for Materials Science and Machine Learning

This module implements various hybrid quantum-classical algorithms combining
quantum computing with classical machine learning for materials discovery.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class MaterialProperty:
    """Container for material properties and features."""
    composition: np.ndarray
    structure: np.ndarray
    electronic: np.ndarray
    target: float
    metadata: Dict[str, Any] = None


class QuantumFeatureMap:
    """Quantum feature mapping for materials data."""
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
    @qml.qnode(device=None)
    def _circuit(self, features, weights):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            
        for layer in range(self.n_layers):
            for i in range(len(features)):
                qml.RY(features[i] * weights[layer, i], wires=i % self.n_qubits)
                
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def encode(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Encode classical features into quantum state."""
        circuit = qml.QNode(self._circuit, self.dev)
        return np.array(circuit(features, weights))


class VQEMolecularOptimizer:
    """Variational Quantum Eigensolver for molecular ground state optimization."""
    
    def __init__(self, n_qubits: int, n_electrons: int):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
    def build_hamiltonian(self, geometry: np.ndarray) -> qml.Hamiltonian:
        """Build molecular Hamiltonian from geometry."""
        coeffs = []
        ops = []
        
        for i in range(self.n_qubits):
            coeffs.append(np.random.randn())
            ops.append(qml.PauliZ(i))
            
        for i in range(self.n_qubits - 1):
            coeffs.append(np.random.randn() * 0.5)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
            
        return qml.Hamiltonian(coeffs, ops)
    
    @qml.qnode(device=None)
    def ansatz(self, weights, wires):
        """Hardware-efficient ansatz for VQE."""
        for i in range(self.n_electrons):
            qml.PauliX(wires=i)
            
        for layer_weights in weights:
            for i in range(self.n_qubits):
                qml.RY(layer_weights[i, 0], wires=i)
                qml.RZ(layer_weights[i, 1], wires=i)
                
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
                
        return qml.expval(self.hamiltonian)
    
    def optimize(self, geometry: np.ndarray, n_layers: int = 4, 
                 n_steps: int = 100) -> Tuple[float, np.ndarray]:
        """Optimize molecular geometry using VQE."""
        self.hamiltonian = self.build_hamiltonian(geometry)
        circuit = qml.QNode(self.ansatz, self.dev)
        
        weights = np.random.randn(n_layers, self.n_qubits, 2) * 0.1
        opt = qml.AdamOptimizer(stepsize=0.1)
        
        energies = []
        for step in range(n_steps):
            weights, energy = opt.step_and_cost(circuit, weights, wires=range(self.n_qubits))
            energies.append(energy)
            
        return energies[-1], weights


class QAOAOptimizer:
    """Quantum Approximate Optimization Algorithm for combinatorial problems."""
    
    def __init__(self, graph: np.ndarray):
        self.graph = graph
        self.n_qubits = len(graph)
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
    def cost_hamiltonian(self) -> List[Tuple[float, qml.Observable]]:
        """Build cost Hamiltonian from graph."""
        terms = []
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if self.graph[i, j] != 0:
                    terms.append((self.graph[i, j], qml.PauliZ(i) @ qml.PauliZ(j)))
        return terms
    
    def mixer_hamiltonian(self) -> List[Tuple[float, qml.Observable]]:
        """Build mixer Hamiltonian."""
        return [(1.0, qml.PauliX(i)) for i in range(self.n_qubits)]
    
    @qml.qnode(device=None)
    def qaoa_circuit(self, gammas, betas):
        """QAOA circuit."""
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            
        for gamma, beta in zip(gammas, betas):
            for coeff, op in self.cost_hamiltonian():
                qml.exp(op, -1j * gamma * coeff)
                
            for coeff, op in self.mixer_hamiltonian():
                qml.exp(op, -1j * beta * coeff)
                
        return qml.expval(qml.Hamiltonian(*zip(*self.cost_hamiltonian())))
    
    def optimize(self, p: int = 2, n_steps: int = 100) -> Tuple[np.ndarray, float]:
        """Optimize QAOA parameters."""
        circuit = qml.QNode(self.qaoa_circuit, self.dev)
        
        gammas = np.random.uniform(0, 2*np.pi, p)
        betas = np.random.uniform(0, np.pi, p)
        params = np.concatenate([gammas, betas])
        
        opt = qml.AdamOptimizer(stepsize=0.1)
        
        for step in range(n_steps):
            params = opt.step(lambda p: circuit(p[:p], p[p:]), params)
            
        optimal_gammas = params[:p]
        optimal_betas = params[p:]
        
        return params, circuit(optimal_gammas, optimal_betas)


class HybridNeuralNetwork(nn.Module):
    """Hybrid quantum-classical neural network for materials property prediction."""
    
    def __init__(self, input_dim: int, quantum_dim: int, output_dim: int,
                 n_qubits: int = 4, n_qlayers: int = 2):
        super().__init__()
        
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, quantum_dim)
        )
        
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        self.quantum_weights = nn.Parameter(
            torch.randn(n_qlayers, n_qubits, 3) * 0.1
        )
        
        self.classical_decoder = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    @qml.qnode(device=None, interface='torch')
    def quantum_layer(self, inputs, weights):
        """Parameterized quantum circuit layer."""
        qml.AngleEmbedding(inputs[:self.n_qubits], wires=range(self.n_qubits))
        
        for layer in range(self.n_qlayers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
                qml.RY(weights[layer, i, 2], wires=i)
                
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward pass through hybrid network."""
        x = self.classical_encoder(x)
        
        circuit = qml.QNode(self.quantum_layer, self.dev, interface='torch')
        quantum_out = []
        for sample in x:
            q_out = circuit(sample, self.quantum_weights)
            quantum_out.append(torch.stack(q_out))
        quantum_out = torch.stack(quantum_out)
        
        x = self.classical_decoder(quantum_out)
        return x


class QuantumKernelRegressor:
    """Quantum kernel-based regression for materials properties."""
    
    def __init__(self, n_qubits: int = 4, feature_map_depth: int = 2):
        self.n_qubits = n_qubits
        self.feature_map_depth = feature_map_depth
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.alpha = None
        self.X_train = None
        
    @qml.qnode(device=None)
    def feature_map(self, x):
        """Quantum feature map circuit."""
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            
        for _ in range(self.feature_map_depth):
            for i, xi in enumerate(x[:self.n_qubits]):
                qml.RZ(xi, wires=i)
                qml.RY(xi**2, wires=i)
                
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
                
        return qml.state()
    
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel between two samples."""
        circuit = qml.QNode(self.feature_map, self.dev)
        
        state1 = circuit(x1)
        state2 = circuit(x2)
        
        return np.abs(np.vdot(state1, state2))**2
    
    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Compute kernel matrix."""
        if X2 is None:
            X2 = X1
            
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.quantum_kernel(X1[i], X2[j])
                
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray, reg: float = 1e-3):
        """Fit the quantum kernel regressor."""
        self.X_train = X
        K = self.compute_kernel_matrix(X)
        K_reg = K + reg * np.eye(len(K))
        self.alpha = np.linalg.solve(K_reg, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        K_test = self.compute_kernel_matrix(X, self.X_train)
        return K_test @ self.alpha


class MaterialsDiscoveryPipeline:
    """Complete pipeline for materials discovery using hybrid algorithms."""
    
    def __init__(self, quantum_features: int = 4):
        self.quantum_features = quantum_features
        self.scaler = StandardScaler()
        self.models = {}
        
    def preprocess_materials(self, materials: List[MaterialProperty]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess materials data for quantum processing."""
        features = []
        targets = []
        
        for mat in materials:
            combined_features = np.concatenate([
                mat.composition.flatten(),
                mat.structure.flatten(),
                mat.electronic.flatten()
            ])
            features.append(combined_features)
            targets.append(mat.target)
            
        X = np.array(features)
        y = np.array(targets)
        
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train_hybrid_model(self, X: np.ndarray, y: np.ndarray, 
                          model_type: str = 'neural') -> Dict[str, Any]:
        """Train a hybrid quantum-classical model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if model_type == 'neural':
            model = HybridNeuralNetwork(
                input_dim=X.shape[1],
                quantum_dim=self.quantum_features,
                output_dim=1,
                n_qubits=self.quantum_features
            )
            
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            X_train_torch = torch.FloatTensor(X_train)
            y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)
            
            losses = []
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_torch)
                loss = criterion(outputs, y_train_torch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
            self.models[model_type] = model
            
            with torch.no_grad():
                X_test_torch = torch.FloatTensor(X_test)
                predictions = model(X_test_torch).numpy()
                mse = np.mean((predictions.flatten() - y_test)**2)
                
        elif model_type == 'kernel':
            model = QuantumKernelRegressor(n_qubits=self.quantum_features)
            
            if len(X_train) > 100:
                indices = np.random.choice(len(X_train), 100, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]
                
            model.fit(X_train, y_train)
            self.models[model_type] = model
            
            predictions = model.predict(X_test[:20])
            mse = np.mean((predictions - y_test[:20])**2)
            
        return {'model': model, 'mse': mse, 'losses': losses if model_type == 'neural' else None}
    
    def optimize_material_structure(self, initial_structure: np.ndarray,
                                   target_property: str = 'bandgap') -> Dict[str, Any]:
        """Optimize material structure using VQE."""
        vqe = VQEMolecularOptimizer(n_qubits=4, n_electrons=2)
        
        energy, optimal_params = vqe.optimize(initial_structure, n_steps=50)
        
        return {
            'optimal_energy': energy,
            'optimal_parameters': optimal_params,
            'target_property': target_property
        }
    
    def find_optimal_composition(self, candidates: List[np.ndarray],
                                constraints: Dict[str, float]) -> np.ndarray:
        """Find optimal material composition using QAOA."""
        n_candidates = len(candidates)
        interaction_matrix = np.random.randn(n_candidates, n_candidates)
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
        
        qaoa = QAOAOptimizer(interaction_matrix)
        optimal_params, optimal_cost = qaoa.optimize(p=3, n_steps=50)
        
        return optimal_params


def demonstrate_hybrid_algorithms():
    """Demonstrate the hybrid quantum-classical algorithms."""
    
    print("Generating synthetic materials data...")
    materials = []
    for i in range(100):
        mat = MaterialProperty(
            composition=np.random.randn(5),
            structure=np.random.randn(10),
            electronic=np.random.randn(8),
            target=np.random.randn()
        )
        materials.append(mat)
    
    pipeline = MaterialsDiscoveryPipeline(quantum_features=4)
    
    print("\nPreprocessing materials data...")
    X, y = pipeline.preprocess_materials(materials)
    
    print("\nTraining hybrid neural network...")
    nn_results = pipeline.train_hybrid_model(X, y, model_type='neural')
    print(f"Neural Network MSE: {nn_results['mse']:.4f}")
    
    print("\nTraining quantum kernel regressor...")
    kernel_results = pipeline.train_hybrid_model(X[:50], y[:50], model_type='kernel')
    print(f"Quantum Kernel MSE: {kernel_results['mse']:.4f}")
    
    print("\nOptimizing material structure with VQE...")
    structure_opt = pipeline.optimize_material_structure(
        np.random.randn(10),
        target_property='bandgap'
    )
    print(f"Optimal energy: {structure_opt['optimal_energy']:.4f}")
    
    print("\nFinding optimal composition with QAOA...")
    candidates = [np.random.randn(5) for _ in range(5)]
    constraints = {'max_cost': 100, 'min_stability': 0.8}
    optimal_comp = pipeline.find_optimal_composition(candidates, constraints)
    print(f"Optimal composition found with {len(optimal_comp)} parameters")
    
    if nn_results['losses']:
        plt.figure(figsize=(10, 5))
        plt.plot(nn_results['losses'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hybrid Neural Network Training Loss')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    demonstrate_hybrid_algorithms()