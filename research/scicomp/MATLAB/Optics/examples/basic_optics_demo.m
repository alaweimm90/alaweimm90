%% Basic Optics Demo - MATLAB Implementation
% This script demonstrates fundamental optics concepts using the Berkeley
% SciComp MATLAB Optics toolbox including geometric optics, wave optics,
% and material properties.
%
% Author: Berkeley SciComp Team
% Date: 2024
%% Initialize
clear; close all; clc;
% Add path to core modules
addpath('../core');
fprintf('BERKELEY SCICOMP - MATLAB OPTICS DEMO\n');
fprintf('=====================================\n\n');
%% Example 1: Snell's Law and Refraction
fprintf('Example 1: Snell''s Law and Refraction\n');
fprintf('======================================\n');
% Parameters
wavelength = 589.3e-9;  % Sodium D-line
incident_angle = 30;    % degrees
% Materials database
materials = OpticalMaterials();
n_air = materials.refractiveIndex('air', wavelength);
n_glass = materials.refractiveIndex('BK7', wavelength);
n_water = materials.refractiveIndex('H2O', wavelength);
fprintf('Wavelength: %.1f nm\n', wavelength*1e9);
fprintf('Incident angle: %.0f°\n', incident_angle);
fprintf('\nRefractive indices:\n');
fprintf('Air:   n = %.6f\n', n_air);
fprintf('BK7:   n = %.6f\n', n_glass);
fprintf('Water: n = %.6f\n', n_water);
% Calculate refraction angles using Snell's law
theta_i_rad = deg2rad(incident_angle);
% Air to glass
sin_theta_r_glass = n_air * sin(theta_i_rad) / n_glass;
theta_r_glass = rad2deg(asin(sin_theta_r_glass));
% Air to water
sin_theta_r_water = n_air * sin(theta_i_rad) / n_water;
theta_r_water = rad2deg(asin(sin_theta_r_water));
fprintf('\nRefraction angles:\n');
fprintf('Air → BK7:   %.2f°\n', theta_r_glass);
fprintf('Air → Water: %.2f°\n', theta_r_water);
% Critical angles
theta_c_glass = rad2deg(asin(n_air / n_glass));
theta_c_water = rad2deg(asin(n_air / n_water));
fprintf('\nCritical angles (for total internal reflection):\n');
fprintf('BK7 → Air:   %.2f°\n', theta_c_glass);
fprintf('Water → Air: %.2f°\n', theta_c_water);
% Visualization
viz = OpticsVisualization();
figure('Position', [100, 100, 1000, 600]);
% Interface
hold on;
plot([-1.5, 1.5], [0, 0], 'k-', 'LineWidth', 2, 'DisplayName', 'Interface');
% Incident ray
x_incident = [-1, 0];
y_incident = [tan(theta_i_rad), 0];
plot(x_incident, y_incident, 'r-', 'LineWidth', 3, ...
    'DisplayName', sprintf('Incident (%.0f°)', incident_angle));
% Refracted ray
x_refracted = [0, 1];
y_refracted = [0, -tan(deg2rad(theta_r_glass))];
plot(x_refracted, y_refracted, 'b-', 'LineWidth', 3, ...
    'DisplayName', sprintf('Refracted (%.1f°)', theta_r_glass));
% Normal line
plot([0, 0], [-1, 1.5], 'k--', 'LineWidth', 1, 'DisplayName', 'Normal');
% Materials
text(-0.7, 0.8, 'Air', 'FontSize', 12, 'FontWeight', 'bold');
text(-0.7, -0.5, 'BK7 Glass', 'FontSize', 12, 'FontWeight', 'bold');
xlim([-1.5, 1.5]);
ylim([-1, 1.5]);
xlabel('Position');
ylabel('Position');
title('Snell''s Law: Refraction at Air-Glass Interface', ...
    'FontSize', 14, 'Color', viz.berkeley_blue, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
axis equal;
hold off;
input('\nPress Enter to continue to Example 2...');
%% Example 2: Thin Lens Ray Tracing
fprintf('\n\nExample 2: Thin Lens Ray Tracing\n');
fprintf('================================\n');
% Lens parameters
focal_length = 0.1;     % 100 mm
lens_diameter = 0.025;  % 25 mm
object_distance = 0.15; % 150 mm
fprintf('Lens focal length: %.0f mm\n', focal_length*1000);
fprintf('Object distance: %.0f mm\n', object_distance*1000);
% Calculate image distance using thin lens equation
image_distance = 1 / (1/focal_length - 1/object_distance);
magnification = -image_distance / object_distance;
fprintf('Image distance: %.1f mm\n', image_distance*1000);
fprintf('Magnification: %.2fx\n', magnification);
% Setup ray tracing system
ray_system = RayOptics('Wavelength', 589e-9);
lens = ThinLens(0, focal_length, lens_diameter);
ray_system.addSurface(lens);
% Create ray bundle
object_height = 0.005;  % 5 mm
ray_heights = linspace(-object_height, object_height, 11);
wavelength = 589e-9;
rays = {};
for i = 1:length(ray_heights)
    ray_start = [ray_heights(i), 0, -object_distance];
    ray_direction = [0, 0, 1];  % Parallel to optical axis
    rays{end+1} = Ray(ray_start, ray_direction, wavelength);
end
% Plot ray diagram
fprintf('Plotting ray diagram...\n');
ray_system.plotRayDiagram(rays, 'Title', 'Thin Lens Ray Tracing');
input('\nPress Enter to continue to Example 3...');
%% Example 3: Gaussian Beam Propagation
fprintf('\n\nExample 3: Gaussian Beam Propagation\n');
fprintf('====================================\n');
% Beam parameters
wavelength = 633e-9;  % HeNe laser
waist_radius = 1e-3;  % 1 mm
power = 1e-3;         % 1 mW
beam = GaussianBeam(wavelength, waist_radius, 'Power', power);
fprintf('Wavelength: %.0f nm\n', wavelength*1e9);
fprintf('Beam waist: %.1f mm\n', waist_radius*1000);
fprintf('Rayleigh range: %.2f mm\n', beam.rayleigh_range*1000);
fprintf('Divergence angle: %.2f mrad\n', beam.divergence_angle*1000);
fprintf('Power: %.1f mW\n', power*1000);
% Plot beam propagation
z_range = [-5e-3, 5e-3];  % ±5 mm
beam.plotPropagation(z_range);
% Plot intensity profile at waist
beam.plotIntensityProfile(0);
input('\nPress Enter to continue to Example 4...');
%% Example 4: Young's Double Slit Interference
fprintf('\n\nExample 4: Young''s Double Slit Interference\n');
fprintf('===========================================\n');
% Parameters
wavelength = 550e-9;      % Green light
slit_separation = 100e-6; % 100 μm
screen_distance = 1.0;    % 1 m
fprintf('Wavelength: %.0f nm\n', wavelength*1e9);
fprintf('Slit separation: %.0f μm\n', slit_separation*1e6);
fprintf('Screen distance: %.1f m\n', screen_distance);
% Calculate interference pattern
interference = WaveOptics.analyzeInterference(wavelength, slit_separation, screen_distance);
fprintf('Fringe spacing: %.3f mm\n', interference.fringe_spacing*1000);
fprintf('Visibility: %.3f\n', interference.visibility);
% Theoretical fringe spacing
fringe_spacing_theory = wavelength * screen_distance / slit_separation;
fprintf('Theoretical fringe spacing: %.3f mm\n', fringe_spacing_theory*1000);
% Plot interference pattern
pattern_info = struct();
pattern_info.wavelength = wavelength;
pattern_info.fringe_spacing = interference.fringe_spacing;
pattern_info.visibility = interference.visibility;
viz.plotInterferencePattern(interference.position, interference.intensity_normalized, ...
    'Title', 'Young''s Double Slit Interference', 'PatternInfo', pattern_info, 'ShowZoom', true);
input('\nPress Enter to continue to Example 5...');
%% Example 5: Single Slit Diffraction
fprintf('\n\nExample 5: Single Slit Diffraction\n');
fprintf('==================================\n');
% Parameters
wavelength = 633e-9;  % HeNe laser
slit_width = 50e-6;   % 50 μm
screen_distance = 2.0; % 2 m
fprintf('Wavelength: %.0f nm\n', wavelength*1e9);
fprintf('Slit width: %.0f μm\n', slit_width*1e6);
fprintf('Screen distance: %.1f m\n', screen_distance);
% Calculate diffraction pattern
diffraction = WaveOptics.calculateDiffraction('single_slit', slit_width, ...
    wavelength, screen_distance, 0.02);
% Theoretical first minimum position
first_minimum = wavelength * screen_distance / slit_width;
fprintf('First minimum position: ±%.2f mm\n', first_minimum*1000);
% Plot diffraction pattern
aperture_info = struct();
aperture_info.aperture_type = 'single_slit';
aperture_info.aperture_size = slit_width;
aperture_info.wavelength = wavelength;
viz.plotDiffractionPattern(diffraction.position, diffraction.intensity_normalized, ...
    'Title', 'Single Slit Diffraction', 'ApertureInfo', aperture_info);
input('\nPress Enter to continue to Material Analysis...');
%% Example 6: Material Dispersion Analysis
fprintf('\n\nExample 6: Material Dispersion Analysis\n');
fprintf('======================================\n');
% Analyze BK7 glass
fprintf('BK7 Glass Analysis:\n');
wl_range = [400e-9, 800e-9];  % Visible range
materials.plotDispersion('BK7', wl_range);
% Compare materials
fprintf('Comparing materials...\n');
compare_materials = {'BK7', 'SiO2', 'H2O'};
materials.compareMaterials(compare_materials, wl_range);
% Calculate properties at specific wavelengths
test_wavelengths = [486.1e-9, 589.3e-9, 656.3e-9];  % F, d, C lines
line_names = {'F-line', 'd-line', 'C-line'};
fprintf('\nBK7 properties at standard wavelengths:\n');
for i = 1:length(test_wavelengths)
    wl = test_wavelengths(i);
    n = materials.refractiveIndex('BK7', wl);
    ng = materials.groupIndex('BK7', wl);
    gvd = materials.groupVelocityDispersion('BK7', wl);
    fprintf('%s (%.1f nm): n = %.6f, ng = %.6f, GVD = %.2f ps²/km\n', ...
        line_names{i}, wl*1e9, n, ng, gvd);
end
%% Example 7: Optical System Design
fprintf('\n\nExample 7: Optical System Design\n');
fprintf('================================\n');
% Create telescope system
elements = {};
elements{1} = struct('type', 'lens', 'position', 0, 'focal_length', 0.2, 'diameter', 50e-3);
elements{2} = struct('type', 'aperture', 'position', 0.15, 'diameter', 10e-3);
elements{3} = struct('type', 'lens', 'position', 0.18, 'focal_length', 0.05, 'diameter', 25e-3);
fprintf('Creating telescope system diagram...\n');
viz.createOpticalSystemDiagram(elements, 'Title', 'Keplerian Telescope');
%% Summary
fprintf('\n\n');
fprintf('=' * ones(1, 60)); fprintf('\n');
fprintf('ALL EXAMPLES COMPLETED SUCCESSFULLY!\n');
fprintf('=' * ones(1, 60)); fprintf('\n');
fprintf('\nKey Learning Points:\n');
fprintf('• Snell''s law governs refraction at interfaces\n');
fprintf('• Thin lenses follow predictable ray tracing rules\n');
fprintf('• Gaussian beams have characteristic propagation properties\n');
fprintf('• Interference creates predictable fringe patterns\n');
fprintf('• Diffraction limits resolution in optical systems\n');
fprintf('• Material dispersion affects optical system design\n');
fprintf('\nFor more advanced functionality, explore the individual class demos:\n');
fprintf('• WaveOptics.demo()\n');
fprintf('• GaussianBeam.demo()\n');
fprintf('• RayOptics.demo()\n');
fprintf('• ThinLens.demo()\n');
fprintf('• Ray.demo()\n');
fprintf('• OpticalMaterials.demo()\n');
fprintf('• OpticsVisualization.demo()\n');
fprintf('\nDemo completed successfully!\n');