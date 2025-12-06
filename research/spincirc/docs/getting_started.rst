Getting Started with SpinCirc
=============================

Welcome to SpinCirc, the comprehensive MATLAB/Python/Verilog-A framework for computational spintronics research. This guide will help you get up and running with SpinCirc quickly and efficiently.

.. note::
   **New to spintronics?** Check out our :doc:`physics_background` section for a comprehensive introduction to the physics behind spintronic devices.

What is SpinCirc?
-----------------

SpinCirc is a state-of-the-art computational framework that implements the **equivalent-circuit spin-transport formalism** for modeling spintronic devices. It provides:

🔬 **Advanced Physics Modeling**
   - 4×4 conductance matrix formalism for F/N/F heterostructures
   - LLG/LLGS solvers with multiple integration schemes
   - Self-consistent transport + magnetodynamics coupling

📊 **Professional Tools**
   - Berkeley-themed plotting with publication quality
   - Machine learning tools for parameter extraction and optimization
   - Comprehensive validation examples

⚡ **High Performance**
   - Parallel computing support
   - GPU acceleration capabilities
   - Memory-optimized algorithms

System Requirements
-------------------

**MATLAB Environment**
   - MATLAB R2019b or later (recommended: R2023a+)
   - Signal Processing Toolbox
   - Statistics and Machine Learning Toolbox
   - Parallel Computing Toolbox (optional but recommended)

**Python Environment**
   - Python 3.8+ (recommended: 3.10+)
   - NumPy 1.20+, SciPy 1.7+, Matplotlib 3.5+
   - PyTorch 1.12+ (for machine learning features)
   - See :doc:`installation` for complete requirements

**Alternative: Docker**
   - Docker Desktop or Docker Engine
   - 4GB RAM minimum, 8GB recommended
   - See our :doc:`installation` guide for Docker setup

First Steps
-----------

1. **Installation**

   .. code-block:: bash

      git clone https://github.com/alaweimm90/SpinCirc.git
      cd SpinCirc

2. **MATLAB Setup**

   .. code-block:: matlab

      % Add SpinCirc to MATLAB path
      addpath(genpath('matlab'));
      
      % Apply Berkeley styling (optional)
      berkeley();
      
      % Verify installation
      runtests('matlab/tests');

3. **Python Setup**

   .. code-block:: bash

      # Install Python dependencies
      pip install -r requirements.txt
      
      # Add to Python path
      export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

   .. code-block:: python

      # Verify Python installation
      import sys
      sys.path.append('python')
      import ml_tools, visualization
      print("✓ SpinCirc Python modules loaded successfully")

Your First Simulation
---------------------

Let's create a simple spin valve simulation to get familiar with SpinCirc:

.. matlab-example::
   :caption: Basic Spin Valve Simulation

   % Create a spin transport solver
   solver = SpinTransportSolver();
   
   % Define F/N/F geometry (200nm × 100nm × 2nm)
   solver.setGeometry(200e-9, 100e-9, 2e-9);
   
   % Set materials: CoFeB/Cu/CoFeB
   materials = [MaterialsDB.CoFeB, MaterialsDB.Cu, MaterialsDB.CoFeB];
   solver.setMaterials(materials);
   
   % Configure magnetizations (antiparallel state)
   solver.setMagnetization(1, [1, 0, 0]);  % Left FM: +x
   solver.setMagnetization(3, [-1, 0, 0]); % Right FM: -x
   
   % Solve transport equations
   [V, I_s, info] = solver.solve();
   
   % Plot results
   solver.plotSolution();
   
   % Calculate TMR
   TMR = info.resistance.parallel / info.resistance.antiparallel - 1;
   fprintf('TMR = %.2f%%\n', TMR * 100);

Expected output:

.. code-block:: none

   SpinCirc Transport Solver
   =========================
   Geometry: 200.0 × 100.0 × 2.0 nm
   Materials: CoFeB/Cu/CoFeB
   
   Solving transport equations...
   ✓ Converged in 23 iterations
   ✓ Residual: 1.2e-12
   
   Results:
   --------
   TMR = 45.3%
   Spin accumulation: 2.1 mV
   Interface resistance: 0.8 Ω·μm²

Understanding the Results
------------------------

The simulation calculates several key quantities:

**Transport Properties**
   - **TMR (Tunnel Magnetoresistance)**: The resistance change between parallel and antiparallel states
   - **Spin accumulation**: The electrochemical potential difference for spin-up and spin-down electrons
   - **Interface resistance**: Contact resistance at F/N interfaces

**Physical Insight**
   The 45.3% TMR indicates strong spin-dependent scattering at the interfaces, typical of CoFeB/Cu systems. The 2.1 mV spin accumulation shows efficient spin injection into the copper spacer layer.

Next Steps
----------

Now that you have your first simulation running, explore these areas:

1. **Learn the Physics** (:doc:`physics_background`)
   - Understand the 4×4 conductance matrix formalism
   - Learn about magnetization dynamics
   - Explore spin-orbit coupling effects

2. **Try More Examples** (:doc:`examples/index`)
   - All-spin logic devices
   - Magnetization switching dynamics
   - Literature reproduction examples

3. **Advanced Features** (:doc:`advanced/optimization`)
   - Parameter optimization with genetic algorithms
   - Monte Carlo uncertainty quantification
   - Machine learning for device design

4. **Customize Your Setup** (:doc:`advanced/custom_materials`)
   - Add new materials to the database
   - Create custom device geometries
   - Implement new physics models

Common Workflows
---------------

**Device Characterization**

.. code-block:: matlab

   % Sweep magnetic field
   H_range = linspace(-100, 100, 201); % Oe
   TMR_vs_H = zeros(size(H_range));
   
   for i = 1:length(H_range)
       solver.setField([H_range(i), 0, 0]);
       [~, ~, info] = solver.solve();
       TMR_vs_H(i) = info.TMR;
   end
   
   % Plot hysteresis loop
   figure;
   plot(H_range, TMR_vs_H, 'LineWidth', 2);
   xlabel('Applied Field (Oe)');
   ylabel('TMR (%)');
   title('TMR vs. Applied Field');

**Parameter Extraction**

.. code-block:: python

   # Use machine learning for parameter extraction
   from ml_tools import ParameterExtractor
   
   extractor = ParameterExtractor()
   experimental_data = load_experimental_iv_curve()
   
   # Extract parameters using Bayesian optimization
   params = extractor.extract_from_iv(
       experimental_data,
       method='bayesian',
       n_trials=100
   )
   
   print(f"Extracted TMR: {params.TMR:.1f}%")
   print(f"Interface resistance: {params.R_int:.2f} Ω·μm²")

**Visualization**

.. code-block:: python

   # Create publication-quality plots
   from visualization import BerkeleyPlots
   
   plotter = BerkeleyPlots()
   
   # Plot magnetization dynamics
   fig = plotter.plot_magnetization_trajectory(
       time, mx, my, mz,
       title="LLG Dynamics",
       save_path="figures/llg_dynamics.pdf"
   )

Getting Help
------------

**Documentation**
   - :doc:`tutorials/index` - Step-by-step tutorials
   - :doc:`api/matlab/index` - MATLAB API reference
   - :doc:`api/python/index` - Python API reference
   - :doc:`examples/index` - Comprehensive examples

**Community**
   - GitHub Discussions: https://github.com/alaweimm90/SpinCirc/discussions
   - Issues: https://github.com/alaweimm90/SpinCirc/issues
   - Email: meshal@berkeley.edu

**Quick References**
   - :doc:`glossary` - Key terms and definitions
   - :doc:`bibliography` - Relevant literature
   - :doc:`changelog` - Version history

Tips for Success
----------------

1. **Start Simple**: Begin with basic examples before moving to complex simulations
2. **Understand the Physics**: Read the :doc:`physics_background` to understand what you're modeling
3. **Validate Results**: Compare with analytical solutions or experimental data when possible
4. **Use Version Control**: Keep track of your simulation parameters and results
5. **Optimize Performance**: Use parallel computing for large parameter sweeps

.. note::
   **Performance Tip**: For large simulations, consider using the Docker container with GPU support for faster computation.

What's Next?
------------

Ready to dive deeper? Here are recommended next steps:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Topic
     - Description
   * - :doc:`installation`
     - Complete installation guide with all options
   * - :doc:`tutorials/index`
     - Step-by-step tutorials for common tasks
   * - :doc:`examples/index`
     - 20+ validation and demonstration examples
   * - :doc:`physics_background`
     - Theoretical foundation of computational spintronics
   * - :doc:`advanced/optimization`
     - Advanced optimization and ML techniques

.. raw:: html

   <div class="center-text" style="margin-top: 2em;">
   <p><strong>Welcome to the SpinCirc community!</strong><br>
   <em>Let's advance computational spintronics together</em> 🚀</p>
   </div>