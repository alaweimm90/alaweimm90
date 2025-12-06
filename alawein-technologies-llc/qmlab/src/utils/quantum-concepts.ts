/**
 * Quantum Computing Concepts Library
 * Educational tooltips and explanations for quantum computing terminology
 *
 * Purpose: Make quantum computing accessible to everyone through clear,
 * jargon-free explanations supplemented with technical details
 */

export interface QuantumConcept {
  term: string;
  simpleExplanation: string;
  detailedExplanation: string;
  analogy?: string;
  mathematicalDefinition?: string;
  relatedConcepts: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  visualizationHint?: string;
  commonMisconceptions?: string[];
  practicalImportance: string;
}

/**
 * Comprehensive quantum concepts library for educational tooltips
 */
export const quantumConceptsLibrary: Record<string, QuantumConcept> = {
  // FUNDAMENTAL CONCEPTS
  qubit: {
    term: 'Qubit',
    simpleExplanation: 'The quantum version of a classical bit — can be 0, 1, or both simultaneously',
    detailedExplanation:
      'A qubit (quantum bit) is the fundamental unit of quantum information. Unlike classical bits that are either 0 or 1, qubits can exist in a superposition of both states simultaneously. A qubit is represented as |ψ⟩ = α|0⟩ + β|1⟩, where α and β are complex numbers satisfying |α|² + |β|² = 1. When measured, the qubit collapses to |0⟩ with probability |α|² or |1⟩ with probability |β|².',
    analogy: 'Think of a coin spinning in the air — it\'s both heads and tails until it lands and you observe it',
    mathematicalDefinition: '|ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1',
    relatedConcepts: ['superposition', 'measurement', 'blochSphere'],
    difficulty: 'beginner',
    visualizationHint: 'Represented as a vector on the Bloch sphere',
    commonMisconceptions: [
      'Qubits are NOT both 0 and 1 at the same time — they\'re in a unique quantum state',
      'Measurement destroys superposition — you can\'t measure without disturbing',
      'More qubits ≠ just more storage — entanglement creates exponential state space',
    ],
    practicalImportance: 'Qubits are the foundation of quantum computing — understanding them unlocks everything else',
  },

  superposition: {
    term: 'Superposition',
    simpleExplanation: 'The ability of quantum systems to be in multiple states at once',
    detailedExplanation:
      'Superposition is the principle that a quantum system can exist in multiple states simultaneously until measured. A qubit in superposition contains information about both |0⟩ and |1⟩ states. This is not just "unknown" (like a classical probability) — it\'s a fundamentally quantum property that enables quantum parallelism. Superposition is created by gates like Hadamard (H) and destroyed by measurement.',
    analogy: 'Like Schrödinger\'s cat being both alive and dead — but this is real for quantum particles!',
    mathematicalDefinition: 'Linear combination of basis states: |ψ⟩ = Σᵢ αᵢ|i⟩',
    relatedConcepts: ['qubit', 'hadamard', 'measurement', 'interference'],
    difficulty: 'beginner',
    visualizationHint: 'Any point on the Bloch sphere surface (except poles) represents superposition',
    commonMisconceptions: [
      'Superposition is NOT just "we don\'t know" — it\'s a real physical state',
      'You can\'t measure superposition directly — measurement projects to a basis state',
      'Superposition alone doesn\'t give quantum speedup — you need interference too',
    ],
    practicalImportance: 'Superposition enables quantum parallelism — the foundation of quantum advantage',
  },

  entanglement: {
    term: 'Entanglement',
    simpleExplanation: 'Quantum correlation where measuring one qubit instantly affects another, no matter the distance',
    detailedExplanation:
      'Quantum entanglement is a correlation between qubits that cannot be explained by classical physics. When qubits are entangled, measuring one instantly determines the state of the others, even if they\'re light-years apart. This "spooky action at a distance" (Einstein\'s words) is not faster-than-light communication, but it is the strongest form of correlation allowed by physics. Entanglement is created by gates like CNOT and is essential for quantum error correction and quantum communication.',
    analogy: 'Like having two magic coins: when you flip one and get heads, you instantly know the other will be tails, no matter how far apart they are',
    mathematicalDefinition: 'Bell state example: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (cannot be factored into single-qubit states)',
    relatedConcepts: ['qubit', 'cnot', 'bellState', 'quantumTeleportation'],
    difficulty: 'intermediate',
    visualizationHint: 'Cannot be visualized on single-qubit Bloch spheres — requires multi-qubit state space',
    commonMisconceptions: [
      'Entanglement is NOT faster-than-light communication',
      'Entanglement doesn\'t mean qubits "know" about each other — they share quantum correlations',
      'You can\'t create entanglement by just preparing qubits in the same state',
    ],
    practicalImportance:
      'Entanglement is the key resource for quantum communication, quantum cryptography, and error correction',
  },

  measurement: {
    term: 'Measurement',
    simpleExplanation: 'Reading a qubit collapses it from superposition to a definite 0 or 1',
    detailedExplanation:
      'Quantum measurement is the process of extracting classical information from a quantum state. When you measure a qubit in superposition |ψ⟩ = α|0⟩ + β|1⟩, it probabilistically collapses to either |0⟩ (with probability |α|²) or |1⟩ (with probability |β|²). This collapse is irreversible and destroys the superposition. Measurement is basis-dependent — measuring in different bases (Z vs X vs Y) gives different results and different post-measurement states.',
    analogy: 'Like finally landing the spinning coin — once it lands, the "both heads and tails" state is gone forever',
    mathematicalDefinition: 'Born rule: P(measuring |i⟩) = |⟨i|ψ⟩|² (probability from amplitude squared)',
    relatedConcepts: ['qubit', 'superposition', 'collapse', 'bornRule'],
    difficulty: 'beginner',
    visualizationHint: 'Projection from anywhere on Bloch sphere to north pole (|0⟩) or south pole (|1⟩)',
    commonMisconceptions: [
      'Measurement doesn\'t "reveal" a pre-existing value — it creates the outcome',
      'You can measure in different bases (X, Y, Z) — choice matters!',
      'Measuring part of an entangled system affects the whole system',
    ],
    practicalImportance: 'Measurement is how we get answers from quantum computers — but timing matters!',
  },

  blochSphere: {
    term: 'Bloch Sphere',
    simpleExplanation: 'A 3D sphere visualization where every point represents a possible qubit state',
    detailedExplanation:
      'The Bloch sphere is a geometric representation of single-qubit quantum states. Every point on the surface represents a pure qubit state. The north pole is |0⟩, the south pole is |1⟩, and points on the equator are equal superpositions with different phases. Quantum gates correspond to rotations of the Bloch sphere. The interior points (if included) represent mixed states (statistical mixtures rather than superpositions).',
    analogy: 'Like a globe where every location is a different quantum state — gates are rotations of the globe',
    mathematicalDefinition:
      '|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩ (θ: latitude, φ: longitude on sphere)',
    relatedConcepts: ['qubit', 'quantumGates', 'superposition'],
    difficulty: 'beginner',
    visualizationHint: 'Sphere with Z-axis vertical: north = |0⟩, south = |1⟩, equator = superpositions',
    commonMisconceptions: [
      'Only works for single qubits — no simple visualization for multi-qubit states',
      'Interior points are mixed states, not superpositions',
      'Antipodal points (opposite poles) are orthogonal, not "opposite"',
    ],
    practicalImportance:
      'Bloch sphere is THE visualization tool for understanding single-qubit quantum mechanics',
  },

  // QUANTUM GATES
  quantumGate: {
    term: 'Quantum Gate',
    simpleExplanation: 'An operation that transforms qubit states — the quantum equivalent of logic gates',
    detailedExplanation:
      'Quantum gates are unitary operations that transform quantum states. Unlike classical logic gates, quantum gates are reversible (bijective) and preserve quantum information. Gates are represented by unitary matrices satisfying U†U = I. Single-qubit gates correspond to rotations on the Bloch sphere, while multi-qubit gates like CNOT can create entanglement. Sequences of gates form quantum circuits that implement quantum algorithms.',
    analogy: 'Like instructions to rotate or flip a globe — each gate is a specific transformation',
    mathematicalDefinition: 'Unitary matrix U where U†U = UU† = I (preserves norm: ⟨ψ|ψ⟩ = 1)',
    relatedConcepts: ['unitarity', 'quantumCircuit', 'universalGates'],
    difficulty: 'beginner',
    visualizationHint: 'Rotations and reflections on the Bloch sphere',
    practicalImportance: 'Quantum gates are the building blocks of quantum algorithms',
  },

  hadamard: {
    term: 'Hadamard Gate',
    simpleExplanation: 'The superposition gate — transforms |0⟩ to equal mix of |0⟩ and |1⟩',
    detailedExplanation:
      'The Hadamard (H) gate is arguably the most important quantum gate. It creates uniform superposition from computational basis states and vice versa. H|0⟩ = (|0⟩ + |1⟩)/√2 and H|1⟩ = (|0⟩ - |1⟩)/√2. On the Bloch sphere, H rotates the qubit by 90° around the X+Z axis. Applying H twice returns to the original state (H² = I). The Hadamard gate is the foundation of quantum parallelism and appears in virtually every quantum algorithm.',
    analogy: 'Like perfectly balancing a coin on edge — equal chance of heads or tails',
    mathematicalDefinition: 'H = (1/√2)[[1, 1], [1, -1]] (normalized Hadamard matrix)',
    relatedConcepts: ['superposition', 'quantumGate', 'basisChange'],
    difficulty: 'beginner',
    visualizationHint: '90° rotation around diagonal axis (X+Z direction) on Bloch sphere',
    practicalImportance:
      'Hadamard is the gateway to quantum advantage — master this to understand quantum algorithms',
  },

  cnot: {
    term: 'CNOT Gate',
    simpleExplanation: 'Controlled-NOT — flips the second qubit if the first qubit is 1',
    detailedExplanation:
      'The Controlled-NOT (CNOT) gate is the fundamental two-qubit gate that creates entanglement. It flips the target qubit if the control qubit is |1⟩, and does nothing if the control is |0⟩. In matrix form: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩. Combined with single-qubit gates, CNOT forms a universal gate set for quantum computing. The CNOT gate is symmetric under basis changes and is essential for quantum error correction.',
    analogy:
      'Like a light switch (control) that turns another light (target) on/off — but quantum!',
    mathematicalDefinition: 'CNOT = [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]',
    relatedConcepts: ['entanglement', 'bellState', 'universalGates', 'quantumCircuit'],
    difficulty: 'intermediate',
    visualizationHint: 'Creates correlation between qubits — cannot be visualized on single Bloch sphere',
    practicalImportance:
      'CNOT creates entanglement and enables universal quantum computation',
  },

  // QUANTUM PHENOMENA
  interference: {
    term: 'Quantum Interference',
    simpleExplanation: 'Quantum probability amplitudes can add or cancel like waves',
    detailedExplanation:
      'Quantum interference is the phenomenon where probability amplitudes (complex numbers) add constructively or destructively, affecting measurement outcomes. This is unlike classical probabilities which always add. In quantum mechanics, paths to the same outcome can interfere: if two paths lead to the same final state, their amplitudes add before squaring to get probabilities. Destructive interference can make certain outcomes impossible, while constructive interference makes others more likely. This is the mechanism behind quantum speedups.',
    analogy:
      'Like wave interference in water — two waves can reinforce each other or cancel out',
    mathematicalDefinition:
      'P(outcome) = |amplitude₁ + amplitude₂|² ≠ |amplitude₁|² + |amplitude₂|² (in general)',
    relatedConcepts: ['superposition', 'amplitude', 'quantumAlgorithm'],
    difficulty: 'intermediate',
    commonMisconceptions: [
      'Interference happens between amplitudes, not probabilities',
      'You need superposition for interference, but superposition alone isn\'t enough',
      'Interference is how quantum algorithms beat classical ones',
    ],
    practicalImportance:
      'Quantum interference is THE mechanism for quantum speedup in algorithms',
  },

  // QUANTUM COMPUTING
  quantumCircuit: {
    term: 'Quantum Circuit',
    simpleExplanation: 'A sequence of quantum gates operating on qubits, like a quantum program',
    detailedExplanation:
      'A quantum circuit is a computational model where qubits flow left-to-right through a sequence of quantum gates, possibly with measurements at the end. Unlike classical circuits, quantum circuits must be reversible (except for measurements). Gates can act on one or more qubits simultaneously. The circuit depth (number of sequential gate layers) affects execution time on real quantum hardware. Circuit optimization minimizes gates and depth while maintaining functionality.',
    analogy: 'Like a flowchart for quantum computation — qubits flow through gates',
    mathematicalDefinition: 'Sequence of unitary operations: |ψ_final⟩ = U_n...U_2 U_1 |ψ_initial⟩',
    relatedConcepts: ['quantumGate', 'qubit', 'circuitDepth', 'quantumAlgorithm'],
    difficulty: 'beginner',
    visualizationHint: 'Horizontal wires = qubits, boxes = gates, time flows left-to-right',
    practicalImportance: 'Quantum circuits are how we express quantum algorithms',
  },

  quantumAlgorithm: {
    term: 'Quantum Algorithm',
    simpleExplanation:
      'A procedure using quantum mechanics to solve problems faster than classical computers',
    detailedExplanation:
      'A quantum algorithm is a computational procedure executed on a quantum computer that exploits superposition, entanglement, and interference to achieve advantage over classical algorithms. Famous examples include Shor\'s algorithm (factoring), Grover\'s algorithm (search), and VQE (quantum chemistry). Quantum advantage comes from exploiting quantum mechanical properties to explore solution spaces more efficiently. Not all problems benefit from quantum computing — the challenge is finding problems where quantum effects help.',
    analogy: 'Like having a magic ability to check multiple puzzle solutions simultaneously',
    relatedConcepts: ['superposition', 'entanglement', 'interference', 'quantumCircuit'],
    difficulty: 'advanced',
    practicalImportance:
      'Quantum algorithms are the reason we build quantum computers — they solve hard problems efficiently',
  },

  // QUANTUM MACHINE LEARNING
  quantumMachineLearning: {
    term: 'Quantum Machine Learning',
    simpleExplanation: 'Using quantum computers to enhance machine learning algorithms',
    detailedExplanation:
      'Quantum Machine Learning (QML) combines quantum computing with machine learning to potentially achieve advantages in training speed, model expressivity, or data efficiency. QML includes quantum versions of classical algorithms (like quantum neural networks), quantum-enhanced feature spaces, and entirely new quantum algorithms. Key approaches include variational quantum circuits (parameterized quantum circuits optimized classically) and quantum kernel methods. QML is an active research area exploring where quantum effects provide advantage.',
    relatedConcepts: ['VQE', 'quantumCircuit', 'parameterizedGate', 'quantumKernel'],
    difficulty: 'advanced',
    practicalImportance:
      'QML could revolutionize AI by handling complex pattern recognition and optimization',
  },

  VQE: {
    term: 'Variational Quantum Eigensolver',
    simpleExplanation: 'Quantum-classical hybrid algorithm for finding lowest energy states',
    detailedExplanation:
      'VQE (Variational Quantum Eigensolver) is a hybrid quantum-classical algorithm for finding ground state energies of quantum systems. It uses a parameterized quantum circuit (ansatz) to prepare trial states, measures the energy expectation value on a quantum computer, then uses classical optimization to update parameters. This process iterates until convergence. VQE is well-suited for near-term quantum computers because it uses shallow circuits and can tolerate some noise.',
    relatedConcepts: ['quantumMachineLearning', 'parameterizedGate', 'quantumChemistry'],
    difficulty: 'advanced',
    practicalImportance:
      'VQE is one of the most promising near-term applications of quantum computing',
  },
};

/**
 * Get concept by term
 */
export function getQuantumConcept(term: string): QuantumConcept | undefined {
  return quantumConceptsLibrary[term.toLowerCase().replace(/\s+/g, '')];
}

/**
 * Get concepts by difficulty
 */
export function getConceptsByDifficulty(
  difficulty: QuantumConcept['difficulty']
): QuantumConcept[] {
  return Object.values(quantumConceptsLibrary).filter(
    (concept) => concept.difficulty === difficulty
  );
}

/**
 * Search concepts by keyword
 */
export function searchConcepts(keyword: string): QuantumConcept[] {
  const searchTerm = keyword.toLowerCase();
  return Object.values(quantumConceptsLibrary).filter(
    (concept) =>
      concept.term.toLowerCase().includes(searchTerm) ||
      concept.simpleExplanation.toLowerCase().includes(searchTerm) ||
      concept.detailedExplanation.toLowerCase().includes(searchTerm) ||
      concept.relatedConcepts.some((related) => related.toLowerCase().includes(searchTerm))
  );
}

/**
 * Get recommended learning sequence
 */
export function getLearningSequence(): QuantumConcept[] {
  const sequence = [
    'qubit',
    'superposition',
    'blochSphere',
    'quantumGate',
    'hadamard',
    'measurement',
    'entanglement',
    'cnot',
    'interference',
    'quantumCircuit',
    'quantumAlgorithm',
    'quantumMachineLearning',
  ];

  return sequence
    .map((term) => getQuantumConcept(term))
    .filter(Boolean) as QuantumConcept[];
}
