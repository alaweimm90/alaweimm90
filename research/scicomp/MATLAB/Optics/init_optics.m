function init_optics()
%INIT_OPTICS Initialize MATLAB Optics Package
%   This function initializes the Berkeley SciComp MATLAB Optics package
%   by adding necessary paths and displaying package information.
%
%   Usage:
%       init_optics()
%
%   Author: Berkeley SciComp Team
%   Date: 2024
fprintf('Berkeley SciComp - MATLAB Optics Package\n');
fprintf('========================================\n\n');
% Get package root directory
package_root = fileparts(mfilename('fullpath'));
% Add core modules to path
core_path = fullfile(package_root, 'core');
if exist(core_path, 'dir')
    addpath(core_path);
    fprintf('Added core modules to path: %s\n', core_path);
else
    warning('Core modules directory not found: %s', core_path);
end
% Add examples to path
examples_path = fullfile(package_root, 'examples');
if exist(examples_path, 'dir')
    addpath(examples_path);
    fprintf('Added examples to path: %s\n', examples_path);
else
    warning('Examples directory not found: %s', examples_path);
end
% Add tests to path (if available)
tests_path = fullfile(package_root, 'tests');
if exist(tests_path, 'dir')
    addpath(tests_path);
    fprintf('Added tests to path: %s\n', tests_path);
end
fprintf('\nPackage initialized successfully!\n\n');
% Display available classes and functions
fprintf('Available Classes:\n');
fprintf('==================\n');
fprintf('• WaveOptics        - Wave optics calculations and analysis\n');
fprintf('• GaussianBeam      - Gaussian beam propagation and parameters\n');
fprintf('• RayOptics         - Ray tracing and geometric optics systems\n');
fprintf('• OpticalSurface    - Base class for optical surfaces\n');
fprintf('• ThinLens          - Thin lens optical element\n');
fprintf('• Ray               - Optical ray for ray tracing\n');
fprintf('• OpticalMaterials  - Material properties and dispersion models\n');
fprintf('• OpticsVisualization - Berkeley-themed visualization tools\n');
fprintf('\nKey Features:\n');
fprintf('=============\n');
fprintf('• Wave Optics: Plane waves, spherical waves, Gaussian beams\n');
fprintf('• Diffraction: Single slit, double slit, circular aperture\n');
fprintf('• Interference: Young''s double slit, Michelson patterns\n');
fprintf('• Ray Tracing: Geometric optics, lens systems, ray propagation\n');
fprintf('• Materials: Sellmeier, Cauchy dispersion models, material database\n');
fprintf('• Visualization: Berkeley-themed plots, ray diagrams, beam profiles\n');
fprintf('\nQuick Start:\n');
fprintf('============\n');
fprintf('Run the basic demo:           basic_optics_demo\n');
fprintf('Wave optics demo:             WaveOptics.demo()\n');
fprintf('Gaussian beam demo:           GaussianBeam.demo()\n');
fprintf('Ray tracing demo:             RayOptics.demo()\n');
fprintf('Material analysis demo:       OpticalMaterials.demo()\n');
fprintf('Visualization demo:           OpticsVisualization.demo()\n');
fprintf('\nPhysical Constants:\n');
fprintf('===================\n');
fprintf('Speed of light:     %.8e m/s\n', 2.99792458e8);
fprintf('Planck constant:    %.8e J⋅s\n', 6.62607015e-34);
fprintf('\nCommon Wavelengths:\n');
fprintf('===================\n');
wavelengths = containers.Map(...
    {'UV-A', 'Violet', 'Blue', 'Green', 'Yellow', 'Red', 'NIR', 'Telecom'}, ...
    {365, 400, 450, 532, 589, 633, 800, 1550});
wl_names = keys(wavelengths);
for i = 1:length(wl_names)
    name = wl_names{i};
    wl = wavelengths(name);
    fprintf('%-10s: %4d nm\n', name, wl);
end
fprintf('\nFor detailed documentation, use ''help <ClassName>'' or ''doc <ClassName>''\n');
fprintf('Example: help GaussianBeam\n\n');
end