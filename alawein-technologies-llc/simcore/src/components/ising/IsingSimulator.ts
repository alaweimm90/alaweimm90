// Enhanced Monte Carlo simulator with improved numerical accuracy
export class IsingSimulator {
  private lattice: number[][];
  private size: number;
  private temperature: number;
  private magneticField: number;
  private algorithm: string;
  
  // Physical constants
  private readonly kB = 1.0; // Boltzmann constant (set to 1 in units where J=1)
  
  constructor(size: number, temperature: number, magneticField: number, algorithm: string) {
    this.size = size;
    this.temperature = temperature;
    this.magneticField = magneticField;
    this.algorithm = algorithm;
    this.lattice = this.initializeRandomLattice();
  }
  
  private initializeRandomLattice(): number[][] {
    return Array(this.size).fill(null).map(() =>
      Array(this.size).fill(null).map(() => Math.random() > 0.5 ? 1 : -1)
    );
  }
  
  private getNeighbors(i: number, j: number): number {
    return (
      this.lattice[(i + 1) % this.size][j] +
      this.lattice[(i - 1 + this.size) % this.size][j] +
      this.lattice[i][(j + 1) % this.size] +
      this.lattice[i][(j - 1 + this.size) % this.size]
    );
  }
  
  // Enhanced Metropolis algorithm with proper energy calculation
  private metropolisStep(): void {
    for (let mc = 0; mc < this.size * this.size; mc++) {
      const i = Math.floor(Math.random() * this.size);
      const j = Math.floor(Math.random() * this.size);
      
      const neighbors = this.getNeighbors(i, j);
      const deltaE = 2 * this.lattice[i][j] * (neighbors + this.magneticField);
      
      if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / (this.kB * this.temperature))) {
        this.lattice[i][j] *= -1;
      }
    }
  }
  
  // Wolff cluster algorithm - more efficient near critical temperature
  private wolffStep(): void {
    const visited = Array(this.size).fill(null).map(() => Array(this.size).fill(false));
    const cluster: [number, number][] = [];
    
    // Start with random spin
    const startI = Math.floor(Math.random() * this.size);
    const startJ = Math.floor(Math.random() * this.size);
    const stack: [number, number][] = [[startI, startJ]];
    const startSpin = this.lattice[startI][startJ];
    
    // Build cluster using Wolff probability
    const p_add = 1 - Math.exp(-2 / (this.kB * this.temperature));
    
    while (stack.length > 0) {
      const [i, j] = stack.pop()!;
      if (visited[i][j]) continue;
      
      visited[i][j] = true;
      cluster.push([i, j]);
      
      // Check neighbors
      const neighbors = [
        [(i + 1) % this.size, j],
        [(i - 1 + this.size) % this.size, j],
        [i, (j + 1) % this.size],
        [i, (j - 1 + this.size) % this.size]
      ];
      
      for (const [ni, nj] of neighbors) {
        if (!visited[ni][nj] && this.lattice[ni][nj] === startSpin) {
          if (Math.random() < p_add) {
            stack.push([ni, nj]);
          }
        }
      }
    }
    
    // Flip entire cluster
    for (const [i, j] of cluster) {
      this.lattice[i][j] *= -1;
    }
  }
  
  // Heat bath algorithm - samples from equilibrium distribution
  private heatBathStep(): void {
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const neighbors = this.getNeighbors(i, j);
        const field = neighbors + this.magneticField;
        
        // Calculate probabilities for up and down states
        const expUp = Math.exp(field / (this.kB * this.temperature));
        const expDown = Math.exp(-field / (this.kB * this.temperature));
        const probUp = expUp / (expUp + expDown);
        
        this.lattice[i][j] = Math.random() < probUp ? 1 : -1;
      }
    }
  }
  
  step(): void {
    switch (this.algorithm) {
      case 'metropolis':
        this.metropolisStep();
        break;
      case 'wolff':
        this.wolffStep();
        break;
      case 'heat_bath':
        this.heatBathStep();
        break;
      default:
        this.metropolisStep();
    }
  }
  
  // Calculate magnetization per spin
  calculateMagnetization(): number {
    const total = this.lattice.flat().reduce((sum, spin) => sum + spin, 0);
    return total / (this.size * this.size);
  }
  
  // Calculate total energy of the system
  calculateEnergy(): number {
    let totalEnergy = 0;
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        // Count each pair only once
        const right = this.lattice[i][(j + 1) % this.size];
        const down = this.lattice[(i + 1) % this.size][j];
        totalEnergy -= this.lattice[i][j] * (right + down);
        
        // External field contribution
        totalEnergy -= this.magneticField * this.lattice[i][j];
      }
    }
    return totalEnergy / (this.size * this.size); // Energy per spin
  }
  
  // Calculate correlation function
  calculateCorrelationFunction(): { distances: number[], correlations: number[] } {
    const maxDistance = Math.floor(this.size / 4);
    const distances = Array.from({ length: maxDistance }, (_, i) => i + 1);
    const correlations: number[] = [];
    
    for (const r of distances) {
      let correlation = 0;
      let count = 0;
      
      for (let i = 0; i < this.size; i++) {
        for (let j = 0; j < this.size; j++) {
          const i2 = (i + r) % this.size;
          correlation += this.lattice[i][j] * this.lattice[i2][j];
          count++;
        }
      }
      
      correlations.push(correlation / count);
    }
    
    return { distances, correlations };
  }
  
  // Estimate correlation length from exponential fit
  calculateCorrelationLength(): number {
    const { distances, correlations } = this.calculateCorrelationFunction();
    
    // Simple exponential decay fit: C(r) = C0 * exp(-r/ξ)
    // Taking log: ln(C(r)) = ln(C0) - r/ξ
    if (correlations.length < 3) return 0;
    
    const logCorr = correlations.filter(c => c > 0).map(c => Math.log(Math.abs(c)));
    if (logCorr.length < 3) return 0;
    
    // Linear regression on log scale
    const n = Math.min(logCorr.length, distances.length);
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (let i = 0; i < n; i++) {
      const x = distances[i];
      const y = logCorr[i];
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope !== 0 ? -1 / slope : 0; // ξ = -1/slope
  }
  
  // Domain analysis using connected components
  findDomains(): { domains: number[][], domainSizes: number[] } {
    const visited = Array(this.size).fill(null).map(() => Array(this.size).fill(false));
    const domains = Array(this.size).fill(null).map(() => Array(this.size).fill(0));
    const domainSizes: number[] = [];
    let domainId = 1;
    
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        if (!visited[i][j]) {
          const size = this.floodFill(i, j, visited, domains, domainId);
          domainSizes.push(size);
          domainId++;
        }
      }
    }
    
    return { domains, domainSizes };
  }
  
  private floodFill(startI: number, startJ: number, visited: boolean[][], domains: number[][], domainId: number): number {
    const stack: [number, number][] = [[startI, startJ]];
    const spin = this.lattice[startI][startJ];
    let size = 0;
    
    while (stack.length > 0) {
      const [i, j] = stack.pop()!;
      if (visited[i][j] || this.lattice[i][j] !== spin) continue;
      
      visited[i][j] = true;
      domains[i][j] = domainId;
      size++;
      
      // Add neighbors
      const neighbors = [
        [(i + 1) % this.size, j],
        [(i - 1 + this.size) % this.size, j],
        [i, (j + 1) % this.size],
        [i, (j - 1 + this.size) % this.size]
      ];
      
      for (const [ni, nj] of neighbors) {
        if (!visited[ni][nj] && this.lattice[ni][nj] === spin) {
          stack.push([ni, nj]);
        }
      }
    }
    
    return size;
  }
  
  // Calculate magnetization histogram
  calculateMagnetizationHistogram(measurements: number[], nBins: number = 50): { bins: number[], counts: number[] } {
    if (measurements.length === 0) return { bins: [], counts: [] };
    
    const minM = Math.min(...measurements);
    const maxM = Math.max(...measurements);
    const binWidth = (maxM - minM) / nBins;
    
    const bins = Array.from({ length: nBins }, (_, i) => minM + (i + 0.5) * binWidth);
    const counts = new Array(nBins).fill(0);
    
    for (const m of measurements) {
      const binIndex = Math.min(Math.floor((m - minM) / binWidth), nBins - 1);
      if (binIndex >= 0) counts[binIndex]++;
    }
    
    // Normalize to probability density
    const total = counts.reduce((a, b) => a + b, 0);
    if (total > 0) {
      for (let i = 0; i < counts.length; i++) {
        counts[i] /= (total * binWidth);
      }
    }
    
    return { bins, counts };
  }
  
  // Getters
  getLattice(): number[][] { return this.lattice; }
  getSize(): number { return this.size; }
  
  // Parameter updates
  updateParameters(temperature: number, magneticField: number, algorithm?: string): void {
    this.temperature = temperature;
    this.magneticField = magneticField;
    if (algorithm) this.algorithm = algorithm;
  }
  
  // Reset to random configuration
  reset(): void {
    this.lattice = this.initializeRandomLattice();
  }
  
  // Set specific configuration
  setConfiguration(config: 'all_up' | 'all_down' | 'random' | 'checkerboard'): void {
    switch (config) {
      case 'all_up':
        this.lattice = Array(this.size).fill(null).map(() => Array(this.size).fill(1));
        break;
      case 'all_down':
        this.lattice = Array(this.size).fill(null).map(() => Array(this.size).fill(-1));
        break;
      case 'checkerboard':
        this.lattice = Array(this.size).fill(null).map((_, i) =>
          Array(this.size).fill(null).map((_, j) => (i + j) % 2 === 0 ? 1 : -1)
        );
        break;
      default:
        this.reset();
    }
  }
}