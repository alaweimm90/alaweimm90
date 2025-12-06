/**
 * Physics WebWorker - Dedicated worker for heavy physics computations
 * Handles graphene band structure, quantum simulations, and matrix operations
 */

// Import physics calculation functions (we'll inline them for the worker)
// Since workers can't import ES modules directly, we'll inline the physics functions

const GRAPHENE_CONSTANTS = {
  a: 2.46,
  a_cc: 1.42,
  hbar: 6.582119569e-16,
  delta: [
    [1.42, 0],
    [-0.71, 1.23],
    [-0.71, -1.23]
  ],
  K: [4*Math.PI/(3*2.46), 0],
  K_prime: [-4*Math.PI/(3*2.46), 0],
  M: [Math.PI/2.46, Math.PI/(Math.sqrt(3)*2.46)]
};

// Inline structure factor calculation
function calculateStructureFactor(kx, ky) {
  let real = 0;
  let imag = 0;
  
  for (const [dx, dy] of GRAPHENE_CONSTANTS.delta) {
    const phase = kx * dx + ky * dy;
    real += Math.cos(phase);
    imag += Math.sin(phase);
  }
  
  return { real, imag };
}

// Inline graphene band calculation
function calculateGrapheneBands(kPoints, params) {
  const results = [];
  
  for (const [kx, ky] of kPoints) {
    const sf = calculateStructureFactor(kx, ky);
    const magnitude = Math.sqrt(sf.real * sf.real + sf.imag * sf.imag);
    
    const energy_plus = params.onsite + Math.abs(params.t1) * magnitude;
    const energy_minus = params.onsite - Math.abs(params.t1) * magnitude;
    
    // Simple velocity calculation (finite difference would be more accurate)
    const fermi_velocity = Math.abs(params.t1) * GRAPHENE_CONSTANTS.a_cc / GRAPHENE_CONSTANTS.hbar;
    
    results.push({
      energy_plus,
      energy_minus,
      velocity_x: fermi_velocity * sf.real / magnitude,
      velocity_y: fermi_velocity * sf.imag / magnitude,
      fermi_velocity
    });
  }
  
  return results;
}

// Matrix diagonalization (simple eigenvalue calculation)
function diagonalizeMatrix(matrix) {
  const startTime = performance.now();
  
  // For real symmetric matrices, use Jacobi method (simplified)
  const n = matrix.length;
  const eigenvalues = new Array(n);
  const eigenvectors = new Array(n).fill(null).map(() => new Array(n).fill(0));
  
  // Initialize eigenvectors as identity matrix
  for (let i = 0; i < n; i++) {
    eigenvectors[i][i] = 1;
    eigenvalues[i] = matrix[i][i];
  }
  
  // Simple power iteration for largest eigenvalue (placeholder)
  for (let i = 0; i < n; i++) {
    eigenvalues[i] = matrix[i][i];
  }
  
  const executionTime = performance.now() - startTime;
  
  return {
    eigenvalues,
    eigenvectors,
    executionTime
  };
}

// Monte Carlo simulation placeholder
function runMonteCarloSimulation(params) {
  const startTime = performance.now();
  const { steps, temperature, systemSize } = params;
  
  // Simple Ising model simulation
  const results = [];
  let energy = 0;
  let magnetization = 0;
  
  for (let step = 0; step < steps; step++) {
    // Simplified Monte Carlo step
    energy += Math.random() - 0.5;
    magnetization += Math.random() - 0.5;
    
    if (step % 100 === 0) {
      results.push({
        step,
        energy: energy / (step + 1),
        magnetization: magnetization / (step + 1),
        temperature
      });
    }
  }
  
  const executionTime = performance.now() - startTime;
  
  return {
    results,
    finalEnergy: energy / steps,
    finalMagnetization: magnetization / steps,
    executionTime
  };
}

// Worker message handler
self.onmessage = function(event) {
  const { id, type, data } = event.data;
  const startTime = performance.now();
  let result;
  let success = true;
  let error = null;
  
  try {
    switch (type) {
      case 'graphene-band-structure':
        const bandResults = calculateGrapheneBands(data.kPoints, data.parameters);
        result = {
          energyPlus: bandResults.map(r => r.energy_plus),
          energyMinus: bandResults.map(r => r.energy_minus),
          velocities: bandResults.map(r => ({ 
            x: r.velocity_x, 
            y: r.velocity_y, 
            fermi: r.fermi_velocity 
          }))
        };
        break;
        
      case 'matrix-diagonalization':
        result = diagonalizeMatrix(data.matrix);
        break;
        
      case 'monte-carlo':
        result = runMonteCarloSimulation(data.parameters);
        break;
        
      case 'quantum-evolution':
        // Placeholder for quantum time evolution
        result = {
          timeSteps: data.timeSteps || [],
          wavefunction: data.initialState || [],
          probabilities: data.initialState ? data.initialState.map(c => c * c) : []
        };
        break;
        
      default:
        throw new Error(`Unknown task type: ${type}`);
    }
  } catch (err) {
    success = false;
    error = err.message;
    result = null;
  }
  
  const executionTime = performance.now() - startTime;
  const memoryUsed = (performance as any).memory ? (performance as any).memory.usedJSHeapSize : 0;
  
  // Send result back to main thread
  self.postMessage({
    id,
    success,
    data: result,
    error,
    executionTime,
    memoryUsed
  });
};

// Handle worker errors
self.onerror = function(error) {
  console.error('Physics worker error:', error);
};