// Comprehensive TypeScript interfaces for better type safety

// Physics simulation interfaces
export interface PhysicsParameters {
  [key: string]: number | string | boolean | number[];
}

export interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  progress: number;
  currentStep: number;
  totalSteps: number;
  elapsed: number;
}

export interface SimulationResult<T = any> {
  data: T;
  metadata: {
    parameters: PhysicsParameters;
    timestamp: number;
    duration: number;
    steps: number;
  };
  error?: string;
}

// Band structure interfaces
export interface BandStructureData {
  kPath: number[];
  energies: number[][];
  kPoints: Array<{
    label: string;
    position: number;
  }>;
  fermiLevel?: number;
}

export interface DOSData {
  energies: number[];
  dos: number[];
  pdos?: Array<{
    label: string;
    values: number[];
  }>;
}

// Quantum state interfaces
export interface QuantumState {
  real: number[];
  imag: number[];
  probability?: number[];
  phase?: number[];
}

export interface WaveFunction {
  psi: QuantumState;
  x: number[];
  t: number;
  energy?: number;
  norm?: number;
}

// 3D visualization interfaces
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface Vector3D extends Point3D {
  magnitude?: number;
}

export interface Mesh3D {
  vertices: Point3D[];
  faces: number[][];
  normals?: Vector3D[];
  colors?: string[];
}

// Crystal structure interfaces
export interface LatticeVector {
  x: number;
  y: number;
  z: number;
}

export interface AtomPosition {
  x: number;
  y: number;
  z: number;
  element: string;
  label?: string;
}

export interface CrystalStructure {
  latticeVectors: [LatticeVector, LatticeVector, LatticeVector];
  atoms: AtomPosition[];
  spaceGroup?: string;
  name?: string;
}

// Statistical physics interfaces
export interface IsingConfiguration {
  spins: number[][];
  energy: number;
  magnetization: number;
  temperature: number;
}

export interface ThermodynamicProperties {
  temperature: number;
  energy: number;
  entropy: number;
  freeEnergy?: number;
  heatCapacity?: number;
  magnetization?: number;
}

// Machine learning interfaces
export interface MLModel {
  type: 'regression' | 'classification' | 'clustering';
  parameters: { [key: string]: any };
  trained: boolean;
  accuracy?: number;
}

export interface MLDataset {
  features: number[][];
  targets?: number[] | number[][];
  labels?: string[];
  metadata: {
    featureNames: string[];
    targetName?: string;
    size: number;
  };
}

export interface MLPrediction {
  prediction: number | number[] | string;
  confidence?: number;
  uncertainty?: number;
}

// Visualization interfaces
export interface PlotData {
  x: number[];
  y: number[];
  z?: number[];
  labels?: string[];
  colors?: string[];
  metadata?: { [key: string]: any };
}

export interface PlotConfig {
  title: string;
  xLabel: string;
  yLabel: string;
  zLabel?: string;
  colorScale?: string;
  showLegend?: boolean;
  responsive?: boolean;
}

export interface Chart3DConfig extends PlotConfig {
  camera?: {
    x: number;
    y: number;
    z: number;
  };
  lighting?: boolean;
  showAxes?: boolean;
}

// Component prop interfaces
export interface ModuleCardProps {
  module: {
    id: string;
    title: string;
    description: string;
    category: string;
    difficulty: string;
    tags: string[];
    equation?: string;
    isImplemented: boolean;
    route?: string;
  };
  onExplore?: (route?: string) => void;
  onTheory?: () => void;
}

export interface ControlPanelProps {
  parameters: PhysicsParameters;
  onParameterChange: (key: string, value: any) => void;
  disabled?: boolean;
  layout?: 'horizontal' | 'vertical';
}

export interface PlotComponentProps {
  data: PlotData | PlotData[];
  config: PlotConfig;
  className?: string;
  onDataUpdate?: (data: PlotData) => void;
}

// Error handling interfaces
export interface ValidationError {
  field: string;
  message: string;
  value: any;
}

export interface SimulationError {
  type: 'physics' | 'numerical' | 'input' | 'system';
  message: string;
  context?: string;
  recoverable: boolean;
  suggestion?: string;
}

// Performance monitoring interfaces
export interface PerformanceMetrics {
  renderTime: number;
  computeTime: number;
  memoryUsage: number;
  fps?: number;
  bundleSize?: number;
}

export interface OptimizationSettings {
  enableWebGPU: boolean;
  enableWebWorkers: boolean;
  maxArraySize: number;
  renderQuality: 'low' | 'medium' | 'high' | 'ultra';
  enableAnimations: boolean;
}

// Accessibility interfaces
export interface AccessibilityFeatures {
  reduceMotion: boolean;
  highContrast: boolean;
  largeText: boolean;
  screenReader: boolean;
  keyboardNavigation: boolean;
}

export interface AriaAttributes {
  'aria-label'?: string;
  'aria-describedby'?: string;
  'aria-expanded'?: boolean;
  'aria-hidden'?: boolean;
  'aria-live'?: 'polite' | 'assertive' | 'off';
  role?: string;
}

// Web Worker interfaces
export interface WorkerMessage<T = any> {
  id: string;
  type: string;
  payload: T;
  timestamp: number;
}

export interface WorkerResponse<T = any> {
  id: string;
  success: boolean;
  result?: T;
  error?: string;
  timestamp: number;
}

// PWA interfaces
export interface PWAConfig {
  enableServiceWorker: boolean;
  enableOffline: boolean;
  enableNotifications: boolean;
  enableBackgroundSync: boolean;
}

export interface OfflineData {
  timestamp: number;
  data: any;
  type: string;
  size: number;
}

// Navigation interfaces
export interface NavigationItem {
  id: string;
  label: string;
  route: string;
  icon?: string;
  children?: NavigationItem[];
  badge?: string | number;
}

export interface BreadcrumbItem {
  label: string;
  route?: string;
  active?: boolean;
}