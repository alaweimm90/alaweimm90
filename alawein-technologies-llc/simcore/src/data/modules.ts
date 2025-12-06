export interface PhysicsModule {
  id: string;
  title: string;
  description: string;
  category: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Research';
  tags: string[];
  equation?: string;
  isImplemented: boolean;
  route?: string;
  theory: {
    overview: string;
    mathematics: string[];
    references: string[];
  };
}

export const physicsModules: PhysicsModule[] = [
  // Band Structure
  {
    id: 'graphene-band-structure',
    title: 'Graphene Electronic Structure',
    description: 'Interactive Dirac cone exploration with real-time strain engineering and spin-orbit coupling effects. Visualize how mechanical deformation reshapes the electronic landscape.',
    category: 'Band Structure',
    difficulty: 'Intermediate',
    tags: ['Tight Binding', 'Strain', 'SOC', 'Dirac Cones'],
    equation: 'H(\\mathbf{k}) = t \\sum_{\\langle i,j \\rangle} c_i^\\dagger c_j\\, e^{i \\mathbf{k} \\cdot \\mathbf{r}_{ij}}',
    isImplemented: true,
    route: '/modules/graphene-band-structure',
    theory: {
      overview: 'Graphene\'s electronic structure using tight-binding approximation with nearest and next-nearest neighbor interactions.',
      mathematics: [
        'Tight-binding Hamiltonian construction',
        'Brillouin zone sampling',
        'Strain tensor effects on hopping parameters',
        'Spin-orbit coupling matrix elements'
      ],
      references: [
        'Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009)',
        'Peres, Rev. Mod. Phys. 82, 2673 (2010)'
      ]
    }
  },
  {
    id: 'mos2-valley-physics',
    title: 'Valley Engineering Lab',
    description: 'Manipulate valley degrees of freedom in TMDs through interactive Berry curvature visualization. Explore valley Hall effects and optical selection rules in real-time.',
    category: 'Materials & Crystals',
    difficulty: 'Advanced',
    tags: ['TMDs', 'Berry Curvature', 'Valley Hall', 'Optical'],
    equation: '\\Omega(\\mathbf{k}) = \\nabla_{\\mathbf{k}} \\times \\langle u_{\\mathbf{k}} | i \\nabla_{\\mathbf{k}} | u_{\\mathbf{k}} \\rangle',
    isImplemented: true,
    route: '/modules/mos2-valley-physics',
    theory: {
      overview: 'Valley-dependent physics in TMDs arising from broken inversion symmetry and strong spin-orbit coupling.',
      mathematics: [
        'k·p effective Hamiltonian',
        'Berry curvature calculation',
        'Valley Hall conductivity',
        'Circular dichroism'
      ],
      references: [
        'Xiao et al., Phys. Rev. Lett. 108, 196802 (2012)',
        'Mak et al., Science 344, 1489 (2014)'
      ]
    }
  },
  {
    id: 'bz-folding',
    title: 'Superlattice Zone Engineering',
    description: 'Engineer reciprocal space through zone folding. Interactive moiré pattern construction with real-time band structure reconstruction and symmetry analysis.',
    category: 'Band Structure',
    difficulty: 'Intermediate',
    tags: ['BZ Folding', 'Superlattices', 'Moiré'],
    equation: '\\mathbf{k}_{\\mathrm{fold}} = \\mathbf{k}_{\\mathrm{orig}} + \\mathbf{G}_{\\mathrm{super}}',
    isImplemented: true,
    route: '/modules/bz-folding',
    theory: {
      overview: 'Zone folding in periodic superlattices and its effects on electronic band structure.',
      mathematics: [
        'Reciprocal lattice construction',
        'Zone folding scheme',
        'Supercell periodicity',
        'Band unfolding methods'
      ],
      references: [
        'Popescu & Zunger, Phys. Rev. Lett. 104, 236403 (2010)'
      ]
    }
  },
  {
    id: 'phonon-band-structure',
    title: 'Phonon Band Structure',
    description: 'Quantum lattice dynamics simulation: phonon dispersion calculations, thermal transport properties, and vibrational mode analysis with temperature dependencies.',
    category: 'Band Structure',
    difficulty: 'Advanced',
    tags: ['Phonons', 'Lattice Dynamics', 'Thermal', 'Vibrational Modes'],
    equation: '\\omega^2(\\mathbf{k}) = \\text{eigenvalues}[D(\\mathbf{k})]',
    isImplemented: true,
    route: '/modules/phonon-band-structure',
    theory: {
      overview: 'Lattice dynamics and phonon dispersion relations in crystalline materials with thermal transport properties.',
      mathematics: [
        'Dynamical matrix construction',
        'Force constant calculations',
        'Phonon density of states',
        'Thermal conductivity modeling'
      ],
      references: [
        'Born & Huang, Dynamical Theory of Crystal Lattices (1954)',
        'Srivastava, The Physics of Phonons (1990)'
      ]
    }
  },

  // Quantum Dynamics
  {
    id: 'tdse-solver',
    title: 'TDSE Wave Packet Laboratory',
    description: 'Watch quantum mechanics unfold in real-time. Multi-dimensional wave packet evolution through custom potential landscapes with interactive field control.',
    category: 'Quantum Dynamics',
    difficulty: 'Advanced',
    tags: ['TDSE', 'Wave Packets', 'Tunneling'],
    equation: 'i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H} \\psi',
    isImplemented: true,
    route: '/modules/tdse-solver',
    theory: {
      overview: 'Numerical methods for solving the time-dependent Schrödinger equation using split-operator and Crank-Nicolson schemes.',
      mathematics: [
        'Split-operator method',
        'Crank-Nicolson discretization',
        'FFT for kinetic energy',
        'Absorbing boundary conditions'
      ],
      references: [
        'Tannor, Introduction to Quantum Mechanics (2007)',
        'Kosloff, J. Phys. Chem. 92, 2087 (1988)'
      ]
    }
  },
  {
    id: 'bloch-sphere',
    title: 'Bloch Sphere Dynamics',
    description: 'Geometric visualization of qubit evolution: interactive Bloch sphere manipulation with real-time Hamiltonian control and quantum gate implementation.',
    category: 'Quantum Dynamics',
    difficulty: 'Beginner',
    tags: ['Qubits', 'Bloch Sphere', 'Rabi'],
    equation: '|\\psi\\rangle = \\cos\\left(\\frac{\\theta}{2}\\right) |0\\rangle + e^{i\\phi} \\sin\\left(\\frac{\\theta}{2}\\right) |1\\rangle',
    isImplemented: true,
    route: '/modules/bloch-sphere',
    theory: {
      overview: 'Geometric representation of qubit states and dynamics on the Bloch sphere.',
      mathematics: [
        'Pauli matrices',
        'Bloch vector representation',
        'Rabi oscillations',
        'Quantum gates as rotations'
      ],
      references: [
        'Nielsen & Chuang, Quantum Computation (2010)'
      ]
    }
  },

  // Spin & Magnetism
  {
    id: 'llg-solver',
    title: 'Landau-Lifshitz-Gilbert Dynamics',
    description: 'Classical magnetization dynamics simulator: Gilbert damping mechanisms, precessional motion analysis, and magnetic switching under external field perturbations.',
    category: 'Spin & Magnetism',
    difficulty: 'Intermediate',
    tags: ['LLG', 'Magnetization', 'Precession'],
    equation: '\\frac{d\\mathbf{m}}{dt} = -\\gamma \\mathbf{m} \\times \\mathbf{H} - \\alpha\\, \\mathbf{m} \\times (\\mathbf{m} \\times \\mathbf{H})',
    isImplemented: true,
    route: '/modules/llg-dynamics',
    theory: {
      overview: 'Classical description of magnetization dynamics including Gilbert damping.',
      mathematics: [
        'LLG equation derivation',
        'Numerical integration schemes',
        'Energy landscape analysis',
        'Switching dynamics'
      ],
      references: [
        'Gilbert, IEEE Trans. Magn. 40, 3443 (2004)'
      ]
    }
  },

  // Field Theory
  {
    id: 'laplace-eigenmodes',
    title: 'Laplace Eigenmodes',
    description: 'Spectral analysis of differential operators: eigenmode decomposition in complex geometries with finite element discretization and boundary condition implementation.',
    category: 'Field Theory',
    difficulty: 'Intermediate',
    tags: ['Eigenmodes', 'PDE', 'Finite Difference'],
    equation: '\\nabla^2 \\phi_n = -\\lambda_n \\phi_n',
    isImplemented: true,
    route: '/modules/laplace-eigenmodes',
    theory: {
      overview: 'Solution of the Laplace eigenvalue problem using finite difference methods.',
      mathematics: [
        'Finite difference discretization',
        'Sparse matrix eigenvalue problems',
        'Boundary condition implementation',
        'Mode visualization'
      ],
      references: [
        'Strang, Computational Science and Engineering (2007)'
      ]
    }
  },

  // Materials & Crystals
  {
    id: 'crystal-visualizer',
    title: 'Crystal Structure Visualizer',
    description: 'Interactive crystallographic analysis: 3D lattice construction, symmetry operation visualization, and space group exploration with real-time manipulation.',
    category: 'Materials & Crystals',
    difficulty: 'Beginner',
    tags: ['Lattices', 'Symmetry', '3D'],
    equation: '\\mathbf{r} = n_1 \\mathbf{a}_1 + n_2 \\mathbf{a}_2 + n_3 \\mathbf{a}_3',
    isImplemented: true,
    route: '/modules/crystal-visualizer',
    theory: {
      overview: 'Fundamental concepts of crystal lattices and their mathematical description.',
      mathematics: [
        'Lattice vectors',
        'Unit cell construction',
        'Symmetry operations',
        'Space group theory'
      ],
      references: [
        'Ashcroft & Mermin, Solid State Physics (1976)'
      ]
    }
  },

  {
    id: 'quantum-tunneling',
    title: 'Quantum Tunneling Observatory',
    description: 'Witness particles penetrating classically forbidden barriers. Interactive transmission probability analysis with real-time wave packet visualization.',
    category: 'Quantum Dynamics',
    difficulty: 'Intermediate',
    tags: ['TDSE', 'Tunneling', 'Wave Packets'],
    equation: 'T \\approx \\exp(-2\\kappa a),\\quad \\kappa = \\sqrt{\\frac{2m(V - E)}{\\hbar^2}}',
    isImplemented: true,
    route: '/modules/quantum-tunneling',
    theory: {
      overview: 'Quantum tunneling phenomenon where particles penetrate classically forbidden energy barriers.',
      mathematics: [
        'Time-dependent Schrödinger equation',
        'WKB approximation',
        'Transmission coefficients',
        'Wave packet dynamics'
      ],
      references: [
        'Griffiths, Introduction to Quantum Mechanics (2017)',
        'Shankar, Principles of Quantum Mechanics (1994)'
      ]
    }
  },

  // Machine Learning
  {
    id: 'pinn-schrodinger',
    title: 'Neural Quantum Solver',
    description: 'AI meets quantum mechanics. Physics-informed neural networks learn to solve the Schrödinger equation with automatic differentiation and constraint enforcement.',
    category: 'Scientific ML',
    difficulty: 'Research',
    tags: ['PINN', 'Neural Networks', 'PDE'],
    equation: '\\mathrm{Loss} = \\mathrm{MSE}_{\\mathrm{PDE}} + \\mathrm{MSE}_{\\mathrm{BC}} + \\mathrm{MSE}_{\\mathrm{IC}}',
    isImplemented: true,
    route: '/modules/pinn-schrodinger',
    theory: {
      overview: 'Using neural networks constrained by physical laws to solve quantum mechanical problems.',
      mathematics: [
        'Automatic differentiation',
        'Physics loss functions',
        'Boundary condition enforcement',
        'Neural network optimization'
      ],
      references: [
        'Raissi et al., J. Comput. Phys. 378, 686 (2019)'
      ]
    }
  },
  {
    id: 'quantum-field-theory',
    title: 'Quantum Field Theory Simulator',
    description: 'Advanced field quantization laboratory: particle creation/annihilation operators, vacuum fluctuation analysis, and Feynman diagram visualization in interactive format.',
    category: 'Field Theory',
    difficulty: 'Research',
    tags: ['QFT', 'Field Quantization', 'Vacuum', 'Particle Physics'],
    equation: '\\phi(x,t) = \\sum_k \\left[ a_k e^{i(kx - \\omega_k t)} + a_k^\\dagger e^{-i(kx - \\omega_k t)} \\right]',
    isImplemented: true,
    route: '/modules/quantum-field-theory',
    theory: {
      overview: 'Quantum field theory principles with field quantization, vacuum state properties, and particle creation/annihilation operators.',
      mathematics: [
        'Field quantization methods',
        'Creation and annihilation operators',
        'Vacuum state fluctuations',
        'Feynman diagram basics'
      ],
      references: [
        'Peskin & Schroeder, An Introduction to QFT (1995)',
        'Weinberg, The Quantum Theory of Fields (1995)'
      ]
    }
  },
  {
    id: 'ml-showcase',
    title: 'ML Showcase',
    description: 'Artificial intelligence-enhanced physics research: pattern recognition algorithms, predictive analytics, and anomaly detection for experimental data interpretation.',
    category: 'Scientific ML',
    difficulty: 'Advanced',
    tags: ['ML', 'Pattern Recognition', 'Analytics'],
    equation: 'y = f(X;\\, \\theta) + \\varepsilon',
    isImplemented: true,
    route: '/modules/ml-showcase',
    theory: {
      overview: 'Machine learning techniques applied to physics simulation data for pattern recognition and predictive analytics.',
      mathematics: [
        'Statistical pattern recognition',
        'Correlation analysis',
        'Time series prediction',
        'Anomaly detection'
      ],
      references: [
        'Carleo et al., Rev. Mod. Phys. 91, 045002 (2019)'
      ]
    }
  },
  {
    id: 'neural-operator',
    title: 'Neural Operator PDE Surrogate',
    description: 'Operator-learning surrogates (e.g., FNO) that learn solution operators of PDEs for fast inference and real-time simulation.',
    category: 'Scientific ML',
    difficulty: 'Advanced',
    tags: ['Neural Operator', 'FNO', 'PDE Surrogate'],
    equation: '\\hat{u} = \\mathcal{G}_\\theta(f)',
    isImplemented: true,
    route: '/modules/ml-showcase',
    theory: {
      overview: 'Neural operators approximate mappings between function spaces, enabling fast surrogates for PDE solution operators.',
      mathematics: [
        'Operator learning objective',
        'Fourier Neural Operator layers',
        'Spectral convolution',
        'Generalization across discretizations'
      ],
      references: [
        'Li et al., Neural Operator: Learning Maps Between Function Spaces (2021)'
      ]
    }
  },

  // Statistical Physics Modules
  {
    id: 'ising-model',
    title: 'Ising Phase Transition Lab',
    description: 'Watch magnetic phase transitions emerge from microscopic interactions. Monte Carlo simulation with real-time critical temperature analysis and correlation tracking.',
    category: 'Statistical Physics',
    difficulty: 'Intermediate',
    tags: ['Monte Carlo', 'Phase Transitions', 'Magnetism', 'Critical Temperature'],
    equation: 'H = -J \\sum_{\\langle i,j \\rangle} S_i S_j - h \\sum_i S_i',
    isImplemented: true,
    route: '/modules/ising-model',
    theory: {
      overview: 'The Ising model describes magnetic interactions between neighboring spins and exhibits phase transitions.',
      mathematics: [
        'Hamiltonian construction',
        'Metropolis algorithm',
        'Critical temperature calculation',
        'Order parameter analysis'
      ],
      references: [
        'Ising (1925)', 'Onsager (1944)', 'Metropolis et al. (1953)'
      ]
    }
  },
  {
    id: 'boltzmann-distribution',
    title: 'Boltzmann Distribution',
    description: 'Statistical thermodynamics laboratory: energy level population analysis, partition function calculations, and thermodynamic property derivation for quantum systems.',
    category: 'Statistical Physics',
    difficulty: 'Intermediate',
    tags: ['Statistical Mechanics', 'Energy Levels', 'Partition Function', 'Thermodynamics'],
    equation: 'P_i = \\frac{g_i e^{-E_i/k_B T}}{Z}',
    isImplemented: true,
    route: '/modules/boltzmann-distribution',
    theory: {
      overview: 'The Boltzmann distribution describes the probability of finding a system in different energy states.',
      mathematics: [
        'Partition function calculation',
        'Average energy derivation',
        'Heat capacity relations',
        'Entropy calculations'
      ],
      references: [
        'Boltzmann (1877)', 'Gibbs (1902)', 'Planck (1900)'
      ]
    }
  },
  {
    id: 'microstates-entropy',
    title: 'Microstates & Entropy',
    description: 'Fundamental statistical mechanics: microstate enumeration, entropy maximization principles, and information-theoretic foundations of thermodynamics with interactive exploration.',
    category: 'Statistical Physics',
    difficulty: 'Beginner',
    tags: ['Entropy', 'Microstates', 'Information Theory', 'Thermodynamics'],
    equation: 'S = k_B \\ln \\Omega',
    isImplemented: true,
    route: '/modules/microstates-entropy',
    theory: {
      overview: 'Entropy measures the number of microscopic arrangements consistent with macroscopic observations.',
      mathematics: [
        'Microstate counting',
        'Multinomial coefficients',
        'Information entropy',
        'Maximum entropy principle'
      ],
      references: [
        'Boltzmann (1877)', 'Shannon (1948)', 'Jaynes (1957)'
      ]
    }
  },
  {
    id: 'canonical-ensemble',
    title: 'Canonical Ensemble',
    description: 'Equilibrium statistical mechanics: canonical partition function analysis, free energy calculations, and thermal fluctuation studies in constant-temperature systems.',
    category: 'Statistical Physics',
    difficulty: 'Advanced',
    tags: ['Canonical Ensemble', 'Partition Function', 'Free Energy', 'Fluctuations'],
    equation: 'Z = \\sum_i e^{-\\beta E_i}',
    isImplemented: true,
    route: '/modules/canonical-ensemble',
    theory: {
      overview: 'The canonical ensemble describes systems in thermal equilibrium with a heat reservoir at fixed temperature.',
      mathematics: [
        'Partition function properties',
        'Thermodynamic relations',
        'Energy fluctuations',
        'Free energy calculations'
      ],
      references: [
        'Gibbs (1902)', 'Boltzmann (1877)', 'Planck (1900)'
      ]
    }
  },
  {
    id: 'brownian-motion',
    title: 'Brownian Motion',
    description: 'Stochastic dynamics simulation: Langevin equation integration, diffusion process analysis, and fluctuation-dissipation theorem demonstration in thermal equilibrium.',
    category: 'Statistical Physics',
    difficulty: 'Intermediate',
    tags: ['Brownian Motion', 'Langevin Dynamics', 'Diffusion', 'Random Walk'],
    equation: 'm\\frac{dv}{dt} = -\\gamma v + F_{\\text{random}}(t)',
    isImplemented: true,
    route: '/modules/brownian-motion',
    theory: {
      overview: 'Brownian motion describes the random movement of particles suspended in a fluid due to molecular collisions.',
      mathematics: [
        'Langevin equation',
        'Einstein relation',
        'Mean square displacement',
        'Fluctuation-dissipation theorem'
      ],
      references: [
        'Brown (1827)', 'Einstein (1905)', 'Perrin (1908)'
      ]
    }
  },
  {
    id: 'symbolic-regression',
    title: 'Symbolic Regression Discovery',
    description: 'Discover governing equations from data using physics-aware symbolic regression (e.g., AI Feynman, PySR).',
    category: 'Scientific ML',
    difficulty: 'Advanced',
    tags: ['Symbolic Regression', 'AI Feynman', 'PySR'],
    equation: '\\min_{\\mathcal{E}} \\; \\mathcal{L}(\\mathcal{E}(X), y) + \\lambda |\\mathcal{E}|',
    isImplemented: true,
    route: '/modules/ml-showcase',
    theory: {
      overview: 'Symbolic regression searches interpretable mathematical forms that fit data while balancing accuracy and complexity.',
      mathematics: [
        'Genetic programming / enumerative search',
        'Complexity-regularized objective',
        'Dimensional analysis priors',
        'Physics constraints and invariances'
      ],
      references: [
        'Udrescu & Tegmark, AI Feynman (2020)',
        'Cranmer et al., PySR (2020)'
      ]
    }
  }
];

export const moduleCategories = [
  'Band Structure',
  'Quantum Dynamics',
  'Materials & Crystals',
  'Spin & Magnetism',
  'Statistical Physics',
  'Field Theory',
  'Scientific ML'
];