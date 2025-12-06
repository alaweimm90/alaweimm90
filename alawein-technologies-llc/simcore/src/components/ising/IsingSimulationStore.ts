import create from 'zustand';

export interface IsingState {
  // Simulation parameters
  size: number;
  temperature: number;
  magneticField: number;
  algorithm: 'metropolis' | 'wolff' | 'heat_bath';

  // Simulation state
  isRunning: boolean;
  step: number;
  animationSpeed: number;

  // Visualization
  showDomains: boolean;
  showCorrelation: boolean;

  // Analysis parameters
  equilibrationSteps: number;
  measurementSteps: number;
  skipInitialSteps: boolean;

  // Data storage
  magnetizationHistory: number[];
  energyHistory: number[];
  heatCapacityHistory: number[];
  susceptibilityHistory: number[];
  magnetizationHistogram: { bins: number[], counts: number[] };

  // Phase transition data
  phaseTransitionData: {
    temperatures: number[];
    magnetizations: number[];
    heatCapacities: number[];
    susceptibilities: number[];
  } | null;

  // Correlation data
  correlationData: {
    distances: number[];
    correlations: number[];
    correlationLength: number;
  } | null;

  // Domain analysis
  domainData: {
    domains: number[][];
    domainSizes: number[];
    averageDomainSize: number;
  } | null;
}

export interface IsingActions {
  // Parameter setters
  setSize: (size: number) => void;
  setTemperature: (temperature: number) => void;
  setMagneticField: (field: number) => void;
  setAlgorithm: (algorithm: 'metropolis' | 'wolff' | 'heat_bath') => void;

  // Simulation control
  toggleSimulation: () => void;
  resetSimulation: () => void;
  stepSimulation: () => void;
  setAnimationSpeed: (speed: number) => void;

  // Visualization control
  toggleDomains: () => void;
  toggleCorrelation: () => void;

  // Analysis control
  setEquilibrationSteps: (steps: number) => void;
  setMeasurementSteps: (steps: number) => void;
  toggleSkipInitialSteps: () => void;

  // Data updates
  updateObservables: (mag: number, energy: number, heatCap: number, suscept: number) => void;
  updateMagnetizationHistogram: (histogram: { bins: number[], counts: number[] }) => void;
  setPhaseTransitionData: (data: IsingState['phaseTransitionData']) => void;
  setCorrelationData: (data: IsingState['correlationData']) => void;
  setDomainData: (data: IsingState['domainData']) => void;

  // Utility
  clearHistory: () => void;
  exportData: () => string;
}

export const useIsingStore = create<IsingState & IsingActions>((set, get) => ({
  // Initial state
  size: 32,
  temperature: 2.27,
  magneticField: 0,
  algorithm: 'metropolis',

  isRunning: false,
  step: 0,
  animationSpeed: 100,

  showDomains: false,
  showCorrelation: false,

  equilibrationSteps: 1000,
  measurementSteps: 1000,
  skipInitialSteps: true,

  magnetizationHistory: [],
  energyHistory: [],
  heatCapacityHistory: [],
  susceptibilityHistory: [],
  magnetizationHistogram: { bins: [], counts: [] },

  phaseTransitionData: null,
  correlationData: null,
  domainData: null,

  // Actions
  setSize: (size) => set({ size }),
  setTemperature: (temperature) => set({ temperature }),
  setMagneticField: (magneticField) => set({ magneticField }),
  setAlgorithm: (algorithm) => set({ algorithm }),

  toggleSimulation: () => set((state) => ({ isRunning: !state.isRunning })),
  resetSimulation: () => set({
    step: 0,
    magnetizationHistory: [],
    energyHistory: [],
    heatCapacityHistory: [],
    susceptibilityHistory: [],
    magnetizationHistogram: { bins: [], counts: [] },
    phaseTransitionData: null,
    correlationData: null,
    domainData: null,
    isRunning: false
  }),
  stepSimulation: () => set((state) => ({ step: state.step + 1 })),
  setAnimationSpeed: (animationSpeed) => set({ animationSpeed }),

  toggleDomains: () => set((state) => ({ showDomains: !state.showDomains })),
  toggleCorrelation: () => set((state) => ({ showCorrelation: !state.showCorrelation })),

  setEquilibrationSteps: (equilibrationSteps) => set({ equilibrationSteps }),
  setMeasurementSteps: (measurementSteps) => set({ measurementSteps }),
  toggleSkipInitialSteps: () => set((state) => ({ skipInitialSteps: !state.skipInitialSteps })),

  updateObservables: (mag, energy, heatCap, suscept) => set((state) => ({
    magnetizationHistory: [...state.magnetizationHistory, mag],
    energyHistory: [...state.energyHistory, energy],
    heatCapacityHistory: [...state.heatCapacityHistory, heatCap],
    susceptibilityHistory: [...state.susceptibilityHistory, suscept]
  })),

  updateMagnetizationHistogram: (magnetizationHistogram) => set({ magnetizationHistogram }),
  setPhaseTransitionData: (phaseTransitionData) => set({ phaseTransitionData }),
  setCorrelationData: (correlationData) => set({ correlationData }),
  setDomainData: (domainData) => set({ domainData }),

  clearHistory: () => set({
    magnetizationHistory: [],
    energyHistory: [],
    heatCapacityHistory: [],
    susceptibilityHistory: []
  }),

  exportData: () => {
    const state = get();
    return JSON.stringify({
      parameters: {
        size: state.size,
        temperature: state.temperature,
        magneticField: state.magneticField,
        algorithm: state.algorithm,
        step: state.step
      },
      observables: {
        magnetization: state.magnetizationHistory,
        energy: state.energyHistory,
        heatCapacity: state.heatCapacityHistory,
        susceptibility: state.susceptibilityHistory,
        histogram: state.magnetizationHistogram
      },
      analysis: {
        phaseTransition: state.phaseTransitionData,
        correlation: state.correlationData,
        domains: state.domainData
      },
      timestamp: new Date().toISOString()
    }, null, 2);
  }
}));
