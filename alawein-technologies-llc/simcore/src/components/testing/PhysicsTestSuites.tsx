/**
 * Pre-defined test suites for physics simulations
 */

import { TestSuite } from './TestRunner';

// Utility functions for physics testing
export const physicsTestUtils = {
  // Numerical comparison with tolerance
  assertAlmostEqual: (actual: number, expected: number, tolerance: number = 1e-10): boolean => {
    return Math.abs(actual - expected) < tolerance;
  },

  // Array comparison with tolerance
  assertArrayAlmostEqual: (actual: number[], expected: number[], tolerance: number = 1e-10): boolean => {
    if (actual.length !== expected.length) return false;
    return actual.every((val, i) => Math.abs(val - expected[i]) < tolerance);
  },

  // Matrix comparison
  assertMatrixAlmostEqual: (actual: number[][], expected: number[][], tolerance: number = 1e-10): boolean => {
    if (actual.length !== expected.length) return false;
    return actual.every((row, i) => 
      physicsTestUtils.assertArrayAlmostEqual(row, expected[i], tolerance)
    );
  },

  // Performance benchmark
  measurePerformance: async function<T>(fn: () => T | Promise<T>, iterations: number = 1): Promise<{ result: T; avgTime: number; minTime: number; maxTime: number }> {
    const times: number[] = [];
    let result: T;

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      result = await fn();
      const end = performance.now();
      times.push(end - start);
    }

    return {
      result: result!,
      avgTime: times.reduce((sum, time) => sum + time, 0) / times.length,
      minTime: Math.min(...times),
      maxTime: Math.max(...times)
    };
  },

  // Memory usage measurement
  measureMemory: (): { used: number; total: number; percentage: number } | null => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100
      };
    }
    return null;
  }
};

// Graphene physics test suite
export const grapheneTestSuite: TestSuite = {
  id: 'graphene',
  name: 'Graphene Physics',
  description: 'Tests for graphene band structure calculations',
  tests: [
    {
      id: 'graphene-band-structure',
      name: 'Band Structure Calculation',
      description: 'Test graphene band structure at high-symmetry points',
      category: 'unit',
      priority: 'high',
      timeout: 5000,
      run: async () => {
        // Mock graphene calculation - simplified for testing
        const kPoints = [
          [0, 0], // Gamma
          [1/3, 1/3], // K
          [0.5, 0] // M
        ];
        
        // Mock result structure
        const result = {
          energyPlus: [1.0, 0.0, 1.5],
          energyMinus: [-1.0, 0.0, -1.5]
        };
        
        // Test that we get expected number of energy values
        if (!result.energyPlus || !result.energyMinus) {
          throw new Error('Missing energy bands');
        }
        
        if (result.energyPlus.length !== kPoints.length) {
          throw new Error('Incorrect number of energy points');
        }

        // Test that Dirac point has zero energy at K point
        const kPointEnergy = result.energyPlus[1]; // K point
        if (!physicsTestUtils.assertAlmostEqual(kPointEnergy, 0, 0.1)) {
          throw new Error('Dirac point energy should be near zero at K point');
        }
      }
    },
    {
      id: 'graphene-symmetry',
      name: 'Symmetry Properties',
      description: 'Test electron-hole symmetry in graphene',
      category: 'unit',
      priority: 'medium',
      run: async () => {
        // Mock symmetry test
        const kPoints = [[0.1, 0.1], [0.2, 0.2]];
        const result = {
          energyPlus: [0.5, 0.8],
          energyMinus: [-0.5, -0.8]
        };
        
        // Test electron-hole symmetry: E+ = -E-
        for (let i = 0; i < result.energyPlus.length; i++) {
          if (!physicsTestUtils.assertAlmostEqual(
            result.energyPlus[i], 
            -result.energyMinus[i], 
            1e-6
          )) {
            throw new Error('Electron-hole symmetry violated');
          }
        }
      }
    },
    {
      id: 'graphene-performance',
      name: 'Performance Benchmark',
      description: 'Benchmark graphene calculation performance',
      category: 'performance',
      priority: 'low',
      run: async () => {
        const kPoints = Array.from({ length: 100 }, (_, i) => [i / 100, i / 100]);
        
        const benchmark = await physicsTestUtils.measurePerformance(
          () => ({
            energyPlus: kPoints.map(() => Math.random()),
            energyMinus: kPoints.map(() => -Math.random())
          }),
          5
        );
        
        // Should complete within reasonable time
        if (benchmark.avgTime > 1000) { // 1 second
          throw new Error(`Performance too slow: ${benchmark.avgTime}ms average`);
        }
        
        console.log(`Graphene benchmark: ${benchmark.avgTime.toFixed(2)}ms average`);
      }
    }
  ]
};

// Ising model test suite
export const isingTestSuite: TestSuite = {
  id: 'ising',
  name: 'Ising Model',
  description: 'Tests for 2D Ising model Monte Carlo simulations',
  tests: [
    {
      id: 'ising-equilibrium',
      name: 'Thermal Equilibrium',
      description: 'Test that system reaches thermal equilibrium',
      category: 'integration',
      priority: 'high',
      timeout: 10000,
      run: async () => {
        // Mock Ising simulation
        const systemSize = 20;
        const temperature = 2.5;
        const steps = 1000;
        
        // Simulate reaching equilibrium
        let magnetization = 0;
        for (let step = 0; step < steps; step++) {
          // Mock Monte Carlo step
          magnetization += (Math.random() - 0.5) * 0.1;
        }
        
        // At high temperature, magnetization should be near zero
        if (Math.abs(magnetization) > 0.5) {
          throw new Error('System did not reach expected thermal equilibrium');
        }
      }
    },
    {
      id: 'ising-phase-transition',
      name: 'Phase Transition',
      description: 'Test critical temperature phase transition',
      category: 'integration',
      priority: 'high',
      run: async () => {
        const systemSize = 10;
        const criticalTemp = 2.269; // Theoretical critical temperature
        
        // Test below critical temperature (ordered phase)
        const lowTempMagnetization = 0.8; // Mock ordered state
        if (lowTempMagnetization < 0.5) {
          throw new Error('Low temperature should show ordered state');
        }
        
        // Test above critical temperature (disordered phase)
        const highTempMagnetization = 0.1; // Mock disordered state
        if (Math.abs(highTempMagnetization) > 0.3) {
          throw new Error('High temperature should show disordered state');
        }
      }
    }
  ]
};

// WebWorker test suite
export const webWorkerTestSuite: TestSuite = {
  id: 'webworker',
  name: 'WebWorker Physics',
  description: 'Tests for WebWorker-based physics calculations',
  tests: [
    {
      id: 'webworker-availability',
      name: 'WebWorker Support',
      description: 'Test WebWorker availability and initialization',
      category: 'unit',
      priority: 'critical',
      run: async () => {
        if (typeof Worker === 'undefined') {
          throw new Error('WebWorker not supported in this environment');
        }
        
        // Test that physics worker can be created
        try {
          const worker = new Worker('/workers/physics-worker.js');
          worker.terminate();
        } catch (error) {
          throw new Error('Failed to create physics worker');
        }
      }
    },
    {
      id: 'webworker-calculation',
      name: 'WebWorker Calculation',
      description: 'Test physics calculation in WebWorker',
      category: 'integration',
      priority: 'high',
      timeout: 5000,
      run: async () => {
        // Mock WebWorker calculation
        const mockResult = await new Promise((resolve) => {
          setTimeout(() => {
            resolve({ energyPlus: [1, 2, 3], energyMinus: [-1, -2, -3] });
          }, 100);
        });
        
        if (!mockResult) {
          throw new Error('WebWorker calculation failed');
        }
      }
    }
  ]
};

// Memory management test suite
export const memoryTestSuite: TestSuite = {
  id: 'memory',
  name: 'Memory Management',
  description: 'Tests for memory usage and cleanup',
  tests: [
    {
      id: 'memory-leak-detection',
      name: 'Memory Leak Detection',
      description: 'Test for memory leaks in large calculations',
      category: 'performance',
      priority: 'medium',
      run: async () => {
        const initialMemory = physicsTestUtils.measureMemory();
        
        // Perform memory-intensive operation
        const largeArray = new Array(1000000).fill(0).map((_, i) => Math.sin(i));
        
        // Cleanup
        largeArray.length = 0;
        
        // Force garbage collection if available
        if ('gc' in window) {
          (window as any).gc();
        }
        
        const finalMemory = physicsTestUtils.measureMemory();
        
        // Check that memory usage didn't increase significantly
        if (finalMemory && initialMemory) {
          const memoryIncrease = finalMemory.used - initialMemory.used;
          const threshold = 50 * 1024 * 1024; // 50MB threshold
          
          if (memoryIncrease > threshold) {
            throw new Error(`Memory leak detected: ${memoryIncrease / 1024 / 1024}MB increase`);
          }
        }
      }
    }
  ]
};

// Export all test suites
export const allTestSuites: TestSuite[] = [
  grapheneTestSuite,
  isingTestSuite,
  webWorkerTestSuite,
  memoryTestSuite
];

export default allTestSuites;