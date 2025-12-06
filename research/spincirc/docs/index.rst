SpinCirc Documentation
======================

.. image:: _static/images/spincirc_logo.png
   :width: 400px
   :align: center
   :alt: SpinCirc Logo

**SpinCirc** is a comprehensive MATLAB/Python/Verilog-A framework for computational spintronics research. It implements the equivalent-circuit spin-transport formalism with advanced magnetodynamics solvers, providing cutting-edge tools for spintronic device modeling and simulation.

.. note::
   🎆 **New in Version 1.0**: Complete 4×4 conductance matrix implementation, Berkeley plotting theme, 20+ validation examples, and professional CI/CD pipeline.

Key Features
------------

🔬 **Advanced Physics Modeling**
   - 4×4 conductance matrix formalism for F/N/F heterostructures
   - LLG/LLGS solvers with multiple integration schemes
   - Self-consistent transport + magnetodynamics coupling
   - Temperature-dependent material properties

🎯 **Comprehensive Device Library**
   - All-spin logic (ASL) devices with optimization
   - Spin valves (NLSV, GMR, TMR) with field sweeps
   - Multiferroic devices with voltage-controlled switching
   - SOT devices with spin-orbit torque characterization

📊 **Professional Visualization**
   - Berkeley-themed plotting with publication quality
   - 3D magnetization trajectory visualization
   - Advanced parameter sweep and optimization tools

⚡ **High Performance Computing**
   - Parallel computing integration
   - GPU acceleration support
   - Memory-optimized algorithms
   - Adaptive meshing capabilities

Quick Start
-----------

**Installation**

.. code-block:: bash

   git clone https://github.com/alaweimm90/SpinCirc.git
   cd SpinCirc

**MATLAB Setup**

.. code-block:: matlab

   addpath(genpath('matlab'));
   berkeley();  % Apply Berkeley styling
   runtests('matlab/tests');  % Verify installation

**Basic Usage**

.. code-block:: matlab

   % Create transport solver
   solver = SpinTransportSolver();
   solver.setGeometry(200e-9, 100e-9, 2e-9);
   
   % Define F/N/F structure
   materials = [MaterialsDB.CoFeB, MaterialsDB.Cu, MaterialsDB.CoFeB];
   solver.setMaterials(materials);
   
   % Solve transport
   [V, I_s, info] = solver.solve();
   solver.plotSolution();

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   tutorials/index
   examples/index
   physics_background

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/matlab/index
   api/python/index
   api/verilog/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/custom_materials
   advanced/parallel_computing
   advanced/optimization
   advanced/monte_carlo

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   development/coding_standards
   development/testing
   development/documentation

.. toctree::
   :maxdepth: 1
   :caption: Reference

   changelog
   bibliography
   glossary
   license

Validation Examples
-------------------

SpinCirc includes 20+ comprehensive validation examples that demonstrate various aspects of computational spintronics:

**Basic Transport**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Example
     - Description
   * - ``transport_1D_diffusion``
     - 1D spin diffusion with analytical validation
   * - ``transport_hanle_precession``
     - Hanle precession under transverse magnetic fields
   * - ``transport_interface_resistance``
     - F/N interface characterization and scaling

**Magnetodynamics**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Example
     - Description
   * - ``llg_damped_precession``
     - LLG equation with Gilbert damping validation
   * - ``llg_hysteresis_loop``
     - Quasi-static hysteresis loop characterization
   * - ``llgs_current_switching``
     - STT-driven magnetization switching dynamics

**Literature Reproduction**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Example
     - Reference
   * - ``reproduce_fig8_ASL``
     - Manipatruni et al., Nature Physics 2015
   * - ``reproduce_fig9_NLSV``
     - Johnson & Silsbee, Phys. Rev. Lett. 1985
   * - ``reproduce_fig10_multistate``
     - Ikeda et al., Nature Materials 2010

Berkeley Plotting Theme
-----------------------

SpinCirc features a professional Berkeley-themed plotting system with the official UC Berkeley color palette:

.. code-block:: matlab

   berkeley();  % Apply Berkeley styling
   
   figure;
   plot(t, mx, 'LineWidth', 2);  % Berkeley Blue
   hold on;
   plot(t, my, 'LineWidth', 2);  % California Gold
   xlabel('Time (ns)');
   ylabel('Magnetization');
   title('LLG Dynamics');
   
   % Save publication-quality figure
   saveBerkeleyFigure(gcf, 'llg_dynamics', 'Format', 'pdf');

.. image:: _static/images/berkeley_colors.png
   :width: 600px
   :align: center
   :alt: Berkeley Color Palette

Physics Background
------------------

SpinCirc implements state-of-the-art computational spintronics based on the equivalent-circuit formalism:

.. physics-note:: Transport Equations

   The 4×4 conductance matrix relates charge and spin currents to electrochemical potentials:

   .. math::

      \begin{pmatrix}
      I_c \\
      I_{s,x} \\
      I_{s,y} \\
      I_{s,z}
      \end{pmatrix} = 
      \begin{pmatrix}
      G_{cc} & G_{c,sx} & G_{c,sy} & G_{c,sz} \\
      G_{sx,c} & G_{sx,sx} & G_{sx,sy} & G_{sx,sz} \\
      G_{sy,c} & G_{sy,sx} & G_{sy,sy} & G_{sy,sz} \\
      G_{sz,c} & G_{sz,sx} & G_{sz,sy} & G_{sz,sz}
      \end{pmatrix}
      \begin{pmatrix}
      \mu_c \\
      \mu_{s,x} \\
      \mu_{s,y} \\
      \mu_{s,z}
      \end{pmatrix}

   This formalism captures the complete coupling between charge and spin degrees of freedom in spintronic devices.

.. physics-note:: Magnetization Dynamics

   The Landau-Lifshitz-Gilbert equation describes magnetization dynamics with damping:

   .. math::

      \frac{d\mathbf{m}}{dt} = -\frac{\gamma}{1+\alpha^2} \left[ \mathbf{m} \times \mathbf{H}_{\text{eff}} + \alpha \mathbf{m} \times (\mathbf{m} \times \mathbf{H}_{\text{eff}}) \right]

   where $\gamma$ is the gyromagnetic ratio, $\alpha$ is the Gilbert damping parameter, and $\mathbf{H}_{\text{eff}}$ includes all effective fields.

Citation
--------

If you use SpinCirc in your research, please cite:

.. code-block:: bibtex

   @article{alawein2018circuit,
     title={Circuit Models for Spintronic Devices Subject to Electric and Magnetic Fields},
     author={Alawein, Meshal and Fariborzi, Hamid},
     journal={IEEE Journal on Exploratory Solid-State Computational Devices and Circuits},
     volume={4},
     number={2},
     pages={76--85},
     year={2018},
     publisher={IEEE}
   }

   @software{spincirc2025,
     title={SpinCirc: Advanced Spin Transport Circuit Framework},
     author={Alawein, Meshal},
     year={2025},
     url={https://github.com/alaweimm90/SpinCirc},
     version={1.0.0}
   }

Support and Community
---------------------

- **Documentation**: https://docs.spincirc.org
- **GitHub Issues**: https://github.com/alaweimm90/SpinCirc/issues
- **Discussions**: https://github.com/alaweimm90/SpinCirc/discussions
- **Email**: meshal@berkeley.edu

License
-------

SpinCirc is released under the MIT License. See :doc:`license` for details.

Acknowledgments
---------------

- **University of California, Berkeley** - Department of Electrical Engineering and Computer Sciences
- **IEEE Journal on Exploratory Solid-State Computational Devices and Circuits** - Original publication venue
- **Spintronic Research Community** - Invaluable feedback and validation

.. raw:: html

   <div class="center-text">
   <p><strong>Built with ❤️ at UC Berkeley</strong><br>
   <em>Advancing the frontiers of computational spintronics</em></p>
   </div>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`