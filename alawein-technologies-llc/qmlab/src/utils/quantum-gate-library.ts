/**
 * Comprehensive Quantum Gate Library
 * Educational descriptions and metadata for all quantum gates in QMLab
 *
 * Guidelines:
 * - All descriptions are designed to be accessible and educational
 * - Mathematical notation is supplemented with plain language
 * - Each gate includes practical applications and learning pathways
 */

export interface QuantumGate {
  name: string;
  symbol: string;
  label: string;
  category: 'pauli' | 'hadamard' | 'rotation' | 'cnot' | 'phase' | 'multi-qubit';
  description: string;
  detailedDescription: string;
  mathematicalForm: string;
  practicalUse: string;
  visualEffect: string;
  complexity: 'beginner' | 'intermediate' | 'advanced';
  prerequisites?: string[];
  relatedGates?: string[];
  applications: string[];
  learningTips: string[];
  ariaLabel: string;
}

/**
 * Complete quantum gate library with comprehensive educational content
 */
export const quantumGateLibrary: Record<string, QuantumGate> = {
  // PAULI GATES - Beginner Level
  X: {
    name: 'X',
    symbol: 'X',
    label: 'Pauli-X',
    category: 'pauli',
    description: 'Bit flip operation — flips |0⟩ to |1⟩ and vice versa',
    detailedDescription:
      'The Pauli-X gate is the quantum equivalent of a classical NOT gate. It performs a bit flip operation, transforming the |0⟩ state into |1⟩ and |1⟩ into |0⟩. On the Bloch sphere, it represents a 180° rotation around the X-axis.',
    mathematicalForm: '|0⟩ → |1⟩, |1⟩ → |0⟩ (Matrix: [[0, 1], [1, 0]])',
    practicalUse: 'State inversion, error correction, quantum algorithm initialization',
    visualEffect: '180° rotation around X-axis on Bloch sphere',
    complexity: 'beginner',
    relatedGates: ['Y', 'Z', 'H'],
    applications: [
      'Quantum error correction',
      'State preparation',
      'Grover search algorithm',
      'Quantum teleportation protocols',
    ],
    learningTips: [
      'Think of X gate as a quantum NOT gate — it flips the qubit',
      'Applying X twice returns to the original state (X² = I)',
      'On the Bloch sphere, watch the state flip from north to south pole',
    ],
    ariaLabel: 'Add Pauli-X gate — bit flip operation, transforms zero state to one state and vice versa',
  },

  Y: {
    name: 'Y',
    symbol: 'Y',
    label: 'Pauli-Y',
    category: 'pauli',
    description: 'Bit and phase flip — combines X and Z operations',
    detailedDescription:
      'The Pauli-Y gate performs both a bit flip and a phase flip simultaneously. It is equivalent to applying a Z gate followed by an X gate (or vice versa). On the Bloch sphere, it represents a 180° rotation around the Y-axis.',
    mathematicalForm: '|0⟩ → i|1⟩, |1⟩ → -i|0⟩ (Matrix: [[0, -i], [i, 0]])',
    practicalUse: 'Combined state transformation, quantum simulation, error correction',
    visualEffect: '180° rotation around Y-axis on Bloch sphere',
    complexity: 'beginner',
    relatedGates: ['X', 'Z', 'H'],
    applications: [
      'Quantum simulation of physical systems',
      'Error correction codes',
      'Quantum state manipulation',
    ],
    learningTips: [
      'Y gate is like X and Z combined — it flips both bit and phase',
      'The imaginary unit i is key to quantum superposition',
      'Y = iXZ (up to a global phase)',
    ],
    ariaLabel: 'Add Pauli-Y gate — bit and phase flip operation, rotates around Y-axis',
  },

  Z: {
    name: 'Z',
    symbol: 'Z',
    label: 'Pauli-Z',
    category: 'pauli',
    description: 'Phase flip — leaves |0⟩ unchanged, adds π phase to |1⟩',
    detailedDescription:
      'The Pauli-Z gate performs a phase flip without changing the computational basis states. It leaves |0⟩ unchanged but multiplies |1⟩ by -1. This is crucial for quantum interference effects. On the Bloch sphere, it represents a 180° rotation around the Z-axis.',
    mathematicalForm: '|0⟩ → |0⟩, |1⟩ → -|1⟩ (Matrix: [[1, 0], [0, -1]])',
    practicalUse: 'Phase kickback, quantum interference, oracle construction',
    visualEffect: '180° rotation around Z-axis on Bloch sphere',
    complexity: 'beginner',
    relatedGates: ['X', 'Y', 'S', 'T'],
    applications: [
      'Quantum algorithms (Grover, Deutsch-Jozsa)',
      'Quantum oracles',
      'Phase estimation',
      'Quantum error correction',
    ],
    learningTips: [
      'Z gate changes phase, not probability — you can\'t see it in measurement!',
      'Critical for quantum interference — the heart of quantum speedup',
      'Computational basis states |0⟩ and |1⟩ are Z gate eigenstates',
    ],
    ariaLabel: 'Add Pauli-Z gate — phase flip operation, adds negative phase to one state',
  },

  // HADAMARD GATE - Beginner Level (Most Important!)
  H: {
    name: 'H',
    symbol: 'H',
    label: 'Hadamard',
    category: 'hadamard',
    description: 'Creates superposition — basis for quantum algorithms',
    detailedDescription:
      'The Hadamard gate is the fundamental superposition gate in quantum computing. It transforms |0⟩ into an equal superposition of |0⟩ and |1⟩, and |1⟩ into a superposition with opposite phases. This gate is essential for most quantum algorithms and creates the quantum parallelism that gives quantum computers their power.',
    mathematicalForm: '|0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ - |1⟩)/√2',
    practicalUse: 'Create superposition, change basis, quantum parallelism',
    visualEffect: 'Rotation from Z-axis to X-axis on Bloch sphere',
    complexity: 'beginner',
    relatedGates: ['X', 'Y', 'Z'],
    applications: [
      'Quantum Fourier Transform',
      'Grover search algorithm',
      'Shor factoring algorithm',
      'Quantum key distribution (BB84)',
      'Superdense coding',
    ],
    learningTips: [
      'H gate is THE superposition gate — master this first!',
      'Applying H twice returns to the original state (H² = I)',
      'H transforms between computational basis (Z) and Hadamard basis (X)',
      'Think of it as rotating the Bloch sphere to see both poles equally',
    ],
    ariaLabel: 'Add Hadamard gate — creates quantum superposition, essential for quantum algorithms',
  },

  // ROTATION GATES - Intermediate Level
  RX: {
    name: 'RX',
    symbol: 'Rₓ',
    label: 'X-Rotation',
    category: 'rotation',
    description: 'Rotation around X-axis by angle θ',
    detailedDescription:
      'The RX gate performs a continuous rotation around the X-axis of the Bloch sphere by an angle θ. This allows fine-grained control of quantum states and is essential for variational quantum algorithms. Unlike discrete gates, rotation gates form a continuous family parameterized by the angle.',
    mathematicalForm: 'RX(θ) = cos(θ/2)I - i·sin(θ/2)X',
    practicalUse: 'Parameterized quantum circuits, optimization, quantum simulation',
    visualEffect: 'Smooth rotation around X-axis by angle θ',
    complexity: 'intermediate',
    prerequisites: ['X', 'H'],
    relatedGates: ['RY', 'RZ', 'U3'],
    applications: [
      'Variational Quantum Eigensolver (VQE)',
      'Quantum Approximate Optimization Algorithm (QAOA)',
      'Quantum machine learning',
      'Quantum chemistry simulations',
    ],
    learningTips: [
      'RX(π) = X — full rotation equals bit flip',
      'Small angles create small perturbations — useful for optimization',
      'Parameter θ is what you optimize in variational algorithms',
    ],
    ariaLabel: 'Add X-rotation gate — parameterized rotation around X-axis, used in quantum machine learning',
  },

  RY: {
    name: 'RY',
    symbol: 'Rᵧ',
    label: 'Y-Rotation',
    category: 'rotation',
    description: 'Rotation around Y-axis by angle θ',
    detailedDescription:
      'The RY gate performs a continuous rotation around the Y-axis of the Bloch sphere by an angle θ. This gate is particularly useful for encoding classical data into quantum states and is widely used in quantum machine learning applications.',
    mathematicalForm: 'RY(θ) = cos(θ/2)I - i·sin(θ/2)Y',
    practicalUse: 'Data encoding, state preparation, quantum neural networks',
    visualEffect: 'Smooth rotation around Y-axis by angle θ',
    complexity: 'intermediate',
    prerequisites: ['Y', 'H'],
    relatedGates: ['RX', 'RZ', 'U3'],
    applications: [
      'Quantum machine learning',
      'Quantum neural networks',
      'Data encoding',
      'Quantum state tomography',
    ],
    learningTips: [
      'RY(π) = Y — full rotation equals Y gate',
      'RY is great for encoding classical data (angles ↔ probabilities)',
      'Combines smoothly with RZ for full state control',
    ],
    ariaLabel: 'Add Y-rotation gate — parameterized rotation around Y-axis, common in quantum neural networks',
  },

  RZ: {
    name: 'RZ',
    symbol: 'Rᵤ',
    label: 'Z-Rotation',
    category: 'rotation',
    description: 'Rotation around Z-axis by angle θ (phase gate)',
    detailedDescription:
      'The RZ gate performs a continuous rotation around the Z-axis (computational basis axis) by an angle θ. This is a phase rotation that doesn\'t change measurement probabilities directly but is crucial for quantum interference. It\'s the simplest rotation to implement on many quantum hardware platforms.',
    mathematicalForm: 'RZ(θ) = exp(-iθ/2)|0⟩⟨0| + exp(iθ/2)|1⟩⟨1|',
    practicalUse: 'Phase manipulation, quantum interference, hardware-efficient circuits',
    visualEffect: 'Rotation around Z-axis (vertical) by angle θ',
    complexity: 'intermediate',
    prerequisites: ['Z', 'S', 'T'],
    relatedGates: ['RX', 'RY', 'S', 'T', 'U1'],
    applications: [
      'Phase estimation',
      'Quantum Fourier Transform',
      'Hardware-efficient ansätze',
      'Quantum error mitigation',
    ],
    learningTips: [
      'RZ only changes phase — invisible in direct measurement!',
      'RZ(π) = Z, RZ(π/2) = S, RZ(π/4) = T',
      'Easiest rotation to implement on superconducting qubits',
    ],
    ariaLabel: 'Add Z-rotation gate — parameterized phase rotation, hardware-efficient operation',
  },

  // CONTROLLED GATES - Advanced Level
  CNOT: {
    name: 'CNOT',
    symbol: '⊕',
    label: 'CNOT',
    category: 'cnot',
    description: 'Controlled NOT — flips target if control is |1⟩',
    detailedDescription:
      'The Controlled-NOT (CNOT) gate is the fundamental two-qubit gate. It flips the target qubit if and only if the control qubit is in state |1⟩. This gate creates entanglement and is universal for quantum computation when combined with single-qubit gates. It\'s the quantum equivalent of classical XOR but can create states with no classical analog.',
    mathematicalForm: '|00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩',
    practicalUse: 'Create entanglement, quantum error correction, universal quantum computation',
    visualEffect: 'Conditional flip — creates quantum correlations between qubits',
    complexity: 'advanced',
    prerequisites: ['X', 'H'],
    relatedGates: ['CZ', 'SWAP', 'Toffoli'],
    applications: [
      'Creating Bell states (maximum entanglement)',
      'Quantum error correction codes',
      'Quantum teleportation',
      'Quantum cryptography',
      'Universal quantum computation',
    ],
    learningTips: [
      'CNOT creates entanglement — the uniquely quantum phenomenon',
      'Try H-CNOT-H to create Bell states — the most entangled states',
      'CNOT is symmetric in the computational basis (classical XOR)',
      'Understanding CNOT unlocks quantum information theory',
    ],
    ariaLabel: 'Add CNOT gate — controlled-NOT operation, creates quantum entanglement between qubits',
  },

  // PHASE GATES - Intermediate Level
  S: {
    name: 'S',
    symbol: 'S',
    label: 'S Gate',
    category: 'phase',
    description: 'Phase gate — adds π/2 phase (quarter turn)',
    detailedDescription:
      'The S gate is a phase gate that adds a π/2 (90°) phase to the |1⟩ state. It\'s equivalent to RZ(π/2) and is one of the Clifford gates. Two S gates equal a Z gate (S² = Z). The S gate is important in quantum error correction and is efficiently implementable on most quantum hardware.',
    mathematicalForm: 'S = [[1, 0], [0, i]] (adds 90° phase)',
    practicalUse: 'Clifford gates, error correction, phase manipulation',
    visualEffect: '90° rotation around Z-axis on Bloch sphere',
    complexity: 'intermediate',
    prerequisites: ['Z'],
    relatedGates: ['T', 'Z', 'RZ'],
    applications: [
      'Stabilizer codes',
      'Clifford circuits',
      'Magic state distillation',
      'Quantum error correction',
    ],
    learningTips: [
      'S is half of a Z gate: S·S = Z',
      'S† (S-dagger) is the inverse, adding -π/2 phase',
      'Part of the Clifford group — efficiently simulatable',
    ],
    ariaLabel: 'Add S gate — phase gate adding 90-degree phase, used in quantum error correction',
  },

  T: {
    name: 'T',
    symbol: 'T',
    label: 'T Gate',
    category: 'phase',
    description: 'T gate — adds π/4 phase (π/8 rotation)',
    detailedDescription:
      'The T gate adds a π/4 (45°) phase to the |1⟩ state. It\'s equivalent to RZ(π/4) and is crucial because it\'s NOT a Clifford gate, which means it\'s necessary for universal quantum computation. The T gate is more difficult to implement fault-tolerantly than Clifford gates, making it a key resource in quantum error correction.',
    mathematicalForm: 'T = [[1, 0], [0, exp(iπ/4)]] (adds 45° phase)',
    practicalUse: 'Universal quantum computation, magic state distillation, quantum algorithms',
    visualEffect: '45° rotation around Z-axis on Bloch sphere',
    complexity: 'advanced',
    prerequisites: ['S', 'Z'],
    relatedGates: ['S', 'RZ'],
    applications: [
      'Fault-tolerant quantum computation',
      'Magic state distillation',
      'Quantum algorithm implementation',
      'T-count optimization',
    ],
    learningTips: [
      'T is NOT a Clifford gate — that makes it special!',
      'T gates are expensive in fault-tolerant quantum computing',
      'Four T gates equal one S gate: T⁴ = S² = Z',
      'Minimizing T-count is key to efficient quantum circuits',
    ],
    ariaLabel: 'Add T gate — π/8 phase gate, enables universal quantum computation beyond Clifford gates',
  },
};

/**
 * Get gate by name
 */
export function getQuantumGate(name: string): QuantumGate | undefined {
  return quantumGateLibrary[name.toUpperCase()];
}

/**
 * Get all gates by category
 */
export function getGatesByCategory(category: QuantumGate['category']): QuantumGate[] {
  return Object.values(quantumGateLibrary).filter((gate) => gate.category === category);
}

/**
 * Get gates by complexity level
 */
export function getGatesByComplexity(complexity: QuantumGate['complexity']): QuantumGate[] {
  return Object.values(quantumGateLibrary).filter((gate) => gate.complexity === complexity);
}

/**
 * Get recommended learning path (ordered gates)
 */
export function getLearningPath(): QuantumGate[] {
  const beginnerGates = ['H', 'X', 'Y', 'Z'];
  const intermediateGates = ['S', 'RX', 'RY', 'RZ'];
  const advancedGates = ['T', 'CNOT'];

  return [
    ...beginnerGates.map((name) => quantumGateLibrary[name]),
    ...intermediateGates.map((name) => quantumGateLibrary[name]),
    ...advancedGates.map((name) => quantumGateLibrary[name]),
  ].filter(Boolean) as QuantumGate[];
}

/**
 * Search gates by keyword
 */
export function searchGates(keyword: string): QuantumGate[] {
  const searchTerm = keyword.toLowerCase();
  return Object.values(quantumGateLibrary).filter(
    (gate) =>
      gate.name.toLowerCase().includes(searchTerm) ||
      gate.label.toLowerCase().includes(searchTerm) ||
      gate.description.toLowerCase().includes(searchTerm) ||
      gate.detailedDescription.toLowerCase().includes(searchTerm) ||
      gate.applications.some((app) => app.toLowerCase().includes(searchTerm))
  );
}
