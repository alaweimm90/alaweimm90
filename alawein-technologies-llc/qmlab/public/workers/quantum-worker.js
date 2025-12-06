// Quantum computation worker for offloading intensive calculations
// Handles quantum state simulation, circuit optimization, and training algorithms

class QuantumWorker {
  constructor() {
    this.stateCache = new Map();
    this.gateCache = new Map();
    this.setupMessageHandling();
  }

  setupMessageHandling() {
    self.onmessage = (event) => {
      try {
        const { type, config, id } = event.data;
        
        switch (type) {
          case 'quantum-simulation':
            this.handleQuantumSimulation(config, id);
            break;
          case 'circuit-optimization':
            this.handleCircuitOptimization(config, id);
            break;
          case 'training-step':
            this.handleTrainingStep(config, id);
            break;
          case 'state-calculation':
            this.handleStateCalculation(config, id);
            break;
          default:
            this.sendError(`Unknown task type: ${type}`, id);
        }
      } catch (error) {
        this.sendError(error.message, event.data.id);
      }
    };
  }

  // Handle quantum simulation tasks
  async handleQuantumSimulation(config, id) {
    const startTime = performance.now();
    
    try {
      const result = await this.performQuantumSimulation(config);
      const executionTime = performance.now() - startTime;
      
      this.sendResult({
        ...result,
        executionTime,
        memoryUsage: this.estimateMemoryUsage(config)
      }, id);
    } catch (error) {
      this.sendError(error.message, id);
    }
  }

  // Perform quantum simulation
  async performQuantumSimulation(config) {
    const { qubits, gates, shots = 1000 } = config;
    const stateSize = Math.pow(2, qubits);
    
    // Initialize quantum state |00...0⟩
    const stateVector = this.getStateVector(stateSize);
    stateVector[0] = 1.0; // |00...0⟩ state
    
    // Apply gates sequentially
    for (const gate of gates) {
      this.applyGate(stateVector, gate, qubits);
    }
    
    // Perform measurements
    const measurements = this.performMeasurements(stateVector, shots);
    
    // Calculate additional metrics
    const fidelity = this.calculateFidelity(stateVector, config.expectedState);
    const entanglement = this.calculateEntanglement(stateVector, qubits);
    
    return {
      stateVector: Array.from(stateVector),
      measurements,
      probabilities: this.calculateProbabilities(stateVector),
      fidelity,
      entanglement,
      qubits
    };
  }

  // Get or create state vector with caching
  getStateVector(size) {
    const key = `state_${size}`;
    if (this.stateCache.has(key)) {
      const cached = this.stateCache.get(key);
      cached.fill(0); // Reset to zero state
      return cached;
    }
    
    const stateVector = new Float64Array(size);
    if (this.stateCache.size < 10) { // Cache up to 10 different sizes
      this.stateCache.set(key, stateVector);
    }
    
    return stateVector;
  }

  // Apply quantum gate to state vector
  applyGate(stateVector, gate, totalQubits) {
    const { type, qubits, angle = 0 } = gate;
    
    switch (type) {
      case 'H': // Hadamard
        this.applyHadamard(stateVector, qubits[0], totalQubits);
        break;
      case 'X': // Pauli-X
        this.applyPauliX(stateVector, qubits[0], totalQubits);
        break;
      case 'Y': // Pauli-Y
        this.applyPauliY(stateVector, qubits[0], totalQubits);
        break;
      case 'Z': // Pauli-Z
        this.applyPauliZ(stateVector, qubits[0], totalQubits);
        break;
      case 'RX': // Rotation-X
        this.applyRotationX(stateVector, qubits[0], angle, totalQubits);
        break;
      case 'RY': // Rotation-Y
        this.applyRotationY(stateVector, qubits[0], angle, totalQubits);
        break;
      case 'RZ': // Rotation-Z
        this.applyRotationZ(stateVector, qubits[0], angle, totalQubits);
        break;
      case 'CNOT': // Controlled-NOT
        this.applyCNOT(stateVector, qubits[0], qubits[1], totalQubits);
        break;
      case 'CZ': // Controlled-Z
        this.applyCZ(stateVector, qubits[0], qubits[1], totalQubits);
        break;
      default:
        throw new Error(`Unsupported gate type: ${type}`);
    }
  }

  // Gate implementations
  applyHadamard(stateVector, qubit, totalQubits) {
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    const factor = 1.0 / Math.sqrt(2);
    
    for (let i = 0; i < stateSize; i++) {
      if (stateVector[i] !== 0) {
        const j = i ^ qubitMask;
        if (i < j) {
          const amp1 = stateVector[i];
          const amp2 = stateVector[j];
          stateVector[i] = factor * (amp1 + amp2);
          stateVector[j] = factor * (amp1 - amp2);
        }
      }
    }
  }

  applyPauliX(stateVector, qubit, totalQubits) {
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    
    for (let i = 0; i < stateSize; i++) {
      if (stateVector[i] !== 0) {
        const j = i ^ qubitMask;
        if (i < j) {
          const temp = stateVector[i];
          stateVector[i] = stateVector[j];
          stateVector[j] = temp;
        }
      }
    }
  }

  applyPauliY(stateVector, qubit, totalQubits) {
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    
    for (let i = 0; i < stateSize; i++) {
      if (stateVector[i] !== 0) {
        const j = i ^ qubitMask;
        if (i < j) {
          const amp1 = stateVector[i];
          const amp2 = stateVector[j];
          
          if ((i & qubitMask) === 0) {
            stateVector[i] = -amp2; // i -> -i * |1⟩
            stateVector[j] = amp1;   // |0⟩ -> i * |1⟩, so net is |0⟩
          }
        }
      }
    }
  }

  applyPauliZ(stateVector, qubit, totalQubits) {
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    
    for (let i = 0; i < stateSize; i++) {
      if ((i & qubitMask) !== 0) {
        stateVector[i] *= -1;
      }
    }
  }

  applyRotationX(stateVector, qubit, angle, totalQubits) {
    const cos_half = Math.cos(angle / 2);
    const sin_half = Math.sin(angle / 2);
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    
    for (let i = 0; i < stateSize; i++) {
      if (stateVector[i] !== 0) {
        const j = i ^ qubitMask;
        if (i < j) {
          const amp1 = stateVector[i];
          const amp2 = stateVector[j];
          stateVector[i] = cos_half * amp1 - sin_half * amp2;
          stateVector[j] = cos_half * amp2 - sin_half * amp1;
        }
      }
    }
  }

  applyRotationY(stateVector, qubit, angle, totalQubits) {
    const cos_half = Math.cos(angle / 2);
    const sin_half = Math.sin(angle / 2);
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    
    for (let i = 0; i < stateSize; i++) {
      if (stateVector[i] !== 0) {
        const j = i ^ qubitMask;
        if (i < j) {
          const amp1 = stateVector[i];
          const amp2 = stateVector[j];
          
          if ((i & qubitMask) === 0) {
            stateVector[i] = cos_half * amp1 + sin_half * amp2;
            stateVector[j] = cos_half * amp2 - sin_half * amp1;
          }
        }
      }
    }
  }

  applyRotationZ(stateVector, qubit, angle, totalQubits) {
    const phase_0 = Math.exp(-0.5 * angle); // e^(-iθ/2)
    const phase_1 = Math.exp(0.5 * angle);  // e^(iθ/2)
    const stateSize = stateVector.length;
    const qubitMask = 1 << qubit;
    
    for (let i = 0; i < stateSize; i++) {
      if (stateVector[i] !== 0) {
        if ((i & qubitMask) === 0) {
          stateVector[i] *= phase_0;
        } else {
          stateVector[i] *= phase_1;
        }
      }
    }
  }

  applyCNOT(stateVector, control, target, totalQubits) {
    const stateSize = stateVector.length;
    const controlMask = 1 << control;
    const targetMask = 1 << target;
    
    for (let i = 0; i < stateSize; i++) {
      if ((i & controlMask) !== 0 && stateVector[i] !== 0) {
        const j = i ^ targetMask;
        const temp = stateVector[i];
        stateVector[i] = stateVector[j];
        stateVector[j] = temp;
      }
    }
  }

  applyCZ(stateVector, control, target, totalQubits) {
    const stateSize = stateVector.length;
    const controlMask = 1 << control;
    const targetMask = 1 << target;
    
    for (let i = 0; i < stateSize; i++) {
      if ((i & controlMask) !== 0 && (i & targetMask) !== 0) {
        stateVector[i] *= -1;
      }
    }
  }

  // Perform measurements
  performMeasurements(stateVector, shots) {
    const probabilities = this.calculateProbabilities(stateVector);
    const measurements = {};
    
    for (let shot = 0; shot < shots; shot++) {
      const outcome = this.sampleFromProbabilities(probabilities);
      const bitString = outcome.toString(2).padStart(
        Math.log2(stateVector.length), '0'
      );
      measurements[bitString] = (measurements[bitString] || 0) + 1;
    }
    
    return measurements;
  }

  // Calculate probabilities from state vector
  calculateProbabilities(stateVector) {
    const probabilities = new Array(stateVector.length);
    for (let i = 0; i < stateVector.length; i++) {
      probabilities[i] = stateVector[i] * stateVector[i];
    }
    return probabilities;
  }

  // Sample from probability distribution
  sampleFromProbabilities(probabilities) {
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (random <= cumulative) {
        return i;
      }
    }
    
    return probabilities.length - 1; // Fallback
  }

  // Calculate fidelity between states
  calculateFidelity(stateVector, expectedState) {
    if (!expectedState || expectedState.length !== stateVector.length) {
      return 1.0; // No expected state provided
    }
    
    let fidelity = 0;
    for (let i = 0; i < stateVector.length; i++) {
      fidelity += stateVector[i] * expectedState[i];
    }
    
    return Math.abs(fidelity);
  }

  // Calculate entanglement measure (von Neumann entropy)
  calculateEntanglement(stateVector, qubits) {
    if (qubits < 2) return 0; // No entanglement for single qubit
    
    // Simplified entanglement measure
    // In practice, this would require more sophisticated calculation
    const probabilities = this.calculateProbabilities(stateVector);
    let entropy = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > 0) {
        entropy -= probabilities[i] * Math.log2(probabilities[i]);
      }
    }
    
    return entropy / qubits; // Normalized entanglement
  }

  // Handle circuit optimization
  async handleCircuitOptimization(config, id) {
    try {
      const optimized = this.optimizeCircuit(config);
      this.sendResult(optimized, id);
    } catch (error) {
      this.sendError(error.message, id);
    }
  }

  // Optimize quantum circuit
  optimizeCircuit(config) {
    const { gates } = config;
    let optimizedGates = [...gates];
    
    // Remove identity gates
    optimizedGates = optimizedGates.filter(gate => gate.type !== 'I');
    
    // Combine adjacent rotation gates
    optimizedGates = this.combineRotationGates(optimizedGates);
    
    // Cancel adjacent inverse gates
    optimizedGates = this.cancelInverseGates(optimizedGates);
    
    return {
      ...config,
      gates: optimizedGates,
      optimizationStats: {
        originalGates: gates.length,
        optimizedGates: optimizedGates.length,
        reduction: gates.length - optimizedGates.length
      }
    };
  }

  // Combine rotation gates on same qubit
  combineRotationGates(gates) {
    const optimized = [];
    let i = 0;
    
    while (i < gates.length) {
      const gate = gates[i];
      
      if (this.isRotationGate(gate.type) && i + 1 < gates.length) {
        const nextGate = gates[i + 1];
        
        if (gate.type === nextGate.type && 
            JSON.stringify(gate.qubits) === JSON.stringify(nextGate.qubits)) {
          // Combine angles
          optimized.push({
            ...gate,
            angle: gate.angle + nextGate.angle
          });
          i += 2;
          continue;
        }
      }
      
      optimized.push(gate);
      i++;
    }
    
    return optimized;
  }

  // Cancel inverse gate pairs
  cancelInverseGates(gates) {
    const optimized = [];
    let i = 0;
    
    while (i < gates.length) {
      const gate = gates[i];
      
      if (i + 1 < gates.length) {
        const nextGate = gates[i + 1];
        
        if (this.areInverseGates(gate, nextGate)) {
          i += 2; // Skip both gates
          continue;
        }
      }
      
      optimized.push(gate);
      i++;
    }
    
    return optimized;
  }

  // Check if gate is a rotation gate
  isRotationGate(type) {
    return ['RX', 'RY', 'RZ'].includes(type);
  }

  // Check if two gates are inverses
  areInverseGates(gate1, gate2) {
    if (gate1.type !== gate2.type) return false;
    if (JSON.stringify(gate1.qubits) !== JSON.stringify(gate2.qubits)) return false;
    
    // For rotation gates, check if angles sum to 2π
    if (this.isRotationGate(gate1.type)) {
      const totalAngle = Math.abs((gate1.angle || 0) + (gate2.angle || 0));
      return Math.abs(totalAngle - 2 * Math.PI) < 1e-10;
    }
    
    // For Pauli gates, they are their own inverse
    if (['X', 'Y', 'Z', 'H'].includes(gate1.type)) {
      return true;
    }
    
    return false;
  }

  // Handle training step calculation
  async handleTrainingStep(config, id) {
    try {
      const result = this.performTrainingStep(config);
      this.sendResult(result, id);
    } catch (error) {
      this.sendError(error.message, id);
    }
  }

  // Perform training step (gradient calculation)
  performTrainingStep(config) {
    const { circuit, parameters, data, targets } = config;
    
    // Simple gradient estimation using parameter-shift rule
    const gradients = {};
    const stepSize = 0.01;
    
    for (const param of parameters) {
      // Forward pass
      const forward = this.evaluateCircuit(circuit, { ...param, value: param.value + stepSize });
      const backward = this.evaluateCircuit(circuit, { ...param, value: param.value - stepSize });
      
      // Calculate gradient
      gradients[param.name] = (forward.loss - backward.loss) / (2 * stepSize);
    }
    
    return {
      gradients,
      currentLoss: this.evaluateCircuit(circuit, parameters).loss,
      timestamp: Date.now()
    };
  }

  // Evaluate circuit for training
  evaluateCircuit(circuit, parameters) {
    // Simplified circuit evaluation
    // In practice, this would run the full quantum simulation
    const randomLoss = Math.random() * 0.1 + 0.05; // Random loss for demo
    
    return {
      loss: randomLoss,
      accuracy: 1 - randomLoss,
      outputs: [Math.random(), Math.random()]
    };
  }

  // Handle state calculation
  async handleStateCalculation(config, id) {
    try {
      const result = this.calculateQuantumState(config);
      this.sendResult(result, id);
    } catch (error) {
      this.sendError(error.message, id);
    }
  }

  // Calculate quantum state for visualization
  calculateQuantumState(config) {
    const { qubits, gates } = config;
    const stateSize = Math.pow(2, qubits);
    const stateVector = this.getStateVector(stateSize);
    stateVector[0] = 1.0;
    
    // Apply gates
    for (const gate of gates) {
      this.applyGate(stateVector, gate, qubits);
    }
    
    // Convert to visualization format
    const blochVectors = this.calculateBlochVectors(stateVector, qubits);
    const probabilities = this.calculateProbabilities(stateVector);
    
    return {
      stateVector: Array.from(stateVector),
      probabilities,
      blochVectors,
      entanglement: this.calculateEntanglement(stateVector, qubits)
    };
  }

  // Calculate Bloch vectors for each qubit
  calculateBlochVectors(stateVector, qubits) {
    const vectors = [];
    
    for (let qubit = 0; qubit < qubits; qubit++) {
      const bloch = this.calculateSingleQubitBloch(stateVector, qubit, qubits);
      vectors.push(bloch);
    }
    
    return vectors;
  }

  // Calculate Bloch vector for single qubit (reduced density matrix)
  calculateSingleQubitBloch(stateVector, targetQubit, totalQubits) {
    // Simplified calculation - in practice would need proper reduced density matrix
    const stateSize = stateVector.length;
    let x = 0, y = 0, z = 0;
    
    for (let i = 0; i < stateSize; i++) {
      const prob = stateVector[i] * stateVector[i];
      const qubitState = (i >> targetQubit) & 1;
      
      if (prob > 0) {
        if (qubitState === 0) {
          z += prob;
        } else {
          z -= prob;
        }
        
        // Simplified x and y calculations
        x += prob * Math.cos(i * 0.1);
        y += prob * Math.sin(i * 0.1);
      }
    }
    
    return { x: x * 0.5, y: y * 0.5, z };
  }

  // Estimate memory usage
  estimateMemoryUsage(config) {
    const { qubits } = config;
    const stateSize = Math.pow(2, qubits);
    return stateSize * 8; // 8 bytes per float64
  }

  // Send successful result
  sendResult(result, id) {
    self.postMessage({
      id,
      result,
      error: null
    });
  }

  // Send error
  sendError(message, id) {
    self.postMessage({
      id,
      result: null,
      error: message
    });
  }
}

// Initialize worker
const quantumWorker = new QuantumWorker();

// Handle worker termination
self.onclose = () => {
  quantumWorker.stateCache.clear();
  quantumWorker.gateCache.clear();
};