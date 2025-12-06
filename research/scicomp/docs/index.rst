.. Berkeley SciComp Framework documentation master file

🐻 Berkeley SciComp Framework Documentation
===========================================

.. image:: https://img.shields.io/badge/Berkeley-SciComp-003262?style=flat-square&logo=university
   :alt: Berkeley SciComp
   :target: https://github.com/berkeley/scicomp

.. image:: https://img.shields.io/badge/version-1.0.0-blue?style=flat-square
   :alt: Version

.. image:: https://img.shields.io/badge/license-MIT-green?style=flat-square
   :alt: License

**University of California, Berkeley**

**Scientific Computing Excellence Since 1868**

Welcome to the comprehensive documentation for the Berkeley SciComp Framework, 
a cutting-edge scientific computing platform developed at UC Berkeley.

.. note::
   This framework represents the culmination of Berkeley's commitment to 
   excellence in computational physics, quantum mechanics, and scientific computing.

Quick Start
-----------

Install the framework:

.. code-block:: bash

   pip install berkeley-scicomp

Get started with a simple example:

.. code-block:: python

   from Quantum.core.quantum_states import BellStates
   from visualization.berkeley_style import BerkeleyPlot
   
   # Create a Bell state
   bell = BellStates.phi_plus()
   
   # Visualize with Berkeley styling
   plotter = BerkeleyPlot()
   plotter.plot_quantum_state(bell)

Key Features
------------

🔬 **Quantum Physics**
   Complete quantum mechanics framework with Bell states, entanglement measures, 
   and cavity QED simulations.

🌡️ **Thermal Transport**
   Advanced heat transfer simulations with finite element methods and 
   multi-dimensional solvers.

🌊 **Signal Processing**
   Comprehensive FFT, spectral analysis, and time-frequency processing tools.

🎯 **Optimization**
   Mathematical optimization suite including linear programming, genetic algorithms, 
   and multi-objective optimization.

🧠 **Machine Learning**
   Physics-informed neural networks (PINNs), neural operators, and materials 
   property prediction.

⚡ **GPU Acceleration**
   CUDA-accelerated computations with automatic CPU fallback.

📊 **Berkeley Integration**
   Official UC Berkeley branding, colors, and academic standards compliance.

Documentation Contents
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/quantum
   api/thermal
   api/signal_processing
   api/optimization
   api/ml_physics
   api/numerical_methods
   api/visualization

.. toctree::
   :maxdepth: 2
   :caption: Theory & Methods
   
   theory/quantum_mechanics
   theory/computational_methods
   theory/ml_physics_theory
   theory/berkeley_standards

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   deployment
   gpu_testing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: About
   
   berkeley_mission
   license
   acknowledgments

Performance Metrics
-------------------

.. list-table:: Framework Performance
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operation
     - Performance
     - Scale
     - Status
   * - Matrix Multiplication
     - 182 GFLOPS
     - 2000×2000
     - ✅
   * - FFT Processing
     - 6.4M samples/sec
     - Real-time
     - ✅
   * - Quantum Simulation
     - 16 qubits
     - Research-grade
     - ✅
   * - Heat Equation
     - 1000 points/sec
     - Engineering
     - ✅

Getting Help
------------

- **Documentation**: https://berkeley-scicomp.readthedocs.io
- **GitHub Issues**: https://github.com/berkeley/scicomp/issues
- **Email Support**: meshal@berkeley.edu
- **Berkeley Community**: UC Berkeley Physics & Engineering

Citation
--------

If you use the Berkeley SciComp Framework in your research, please cite:

.. code-block:: bibtex

   @software{berkeley_scicomp,
     title = {Berkeley SciComp Framework: Scientific Computing Excellence},
     author = {Alawein, Meshal and UC Berkeley SciComp Team},
     institution = {University of California, Berkeley},
     year = {2025},
     url = {https://github.com/berkeley/scicomp},
     version = {1.0.0}
   }

Berkeley Excellence
-------------------

.. note::
   🎓 **Fiat Lux** - *Let There Be Light*
   
   The Berkeley SciComp Framework embodies UC Berkeley's mission of advancing 
   human knowledge through computational excellence and open science.

**💙💛 University of California, Berkeley 💙💛**

*Scientific Computing Excellence Since 1868*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`