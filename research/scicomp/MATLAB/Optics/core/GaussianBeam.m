classdef GaussianBeam < handle
    %GAUSSIANBEAM Gaussian beam propagation and analysis
    %   Comprehensive Gaussian beam functionality including propagation,
    %   beam parameter calculations, and intensity profiles
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        wavelength_vacuum    % Wavelength in vacuum (meters)
        waist_radius        % Beam waist radius (meters)
        waist_position      % Z position of beam waist (meters)
        medium_index        % Refractive index of medium
        power               % Beam power (watts)
        % Derived properties
        wavelength          % Wavelength in medium (meters)
        k                   % Wave number (rad/m)
        rayleigh_range      % Rayleigh range (meters)
        divergence_angle    % Far-field divergence half-angle (rad)
        % Berkeley color scheme
        berkeley_blue = [0, 50, 98]/255
        california_gold = [253, 181, 21]/255
        berkeley_light_blue = [59, 126, 161]/255
    end
    properties (Constant)
        % Physical constants
        SPEED_OF_LIGHT = 2.99792458e8  % m/s
        PLANCK_CONSTANT = 6.62607015e-34  % J⋅s
    end
    methods
        function obj = GaussianBeam(wavelength, waist_radius, varargin)
            %GAUSSIANBEAM Constructor
            %   beam = GaussianBeam(wavelength, waist_radius, ...)
            %
            %   Parameters:
            %       'WaistPosition' - Z position of waist (meters, default: 0)
            %       'MediumIndex' - Refractive index (default: 1.0)
            %       'Power' - Beam power (watts, default: 1e-3)
            p = inputParser;
            addRequired(p, 'wavelength', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'waist_radius', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'WaistPosition', 0, @isnumeric);
            addParameter(p, 'MediumIndex', 1.0, @(x) isnumeric(x) && x >= 1);
            addParameter(p, 'Power', 1e-3, @(x) isnumeric(x) && x > 0);
            parse(p, wavelength, waist_radius, varargin{:});
            obj.wavelength_vacuum = p.Results.wavelength;
            obj.waist_radius = p.Results.waist_radius;
            obj.waist_position = p.Results.WaistPosition;
            obj.medium_index = p.Results.MediumIndex;
            obj.power = p.Results.Power;
            % Calculate derived properties
            obj.wavelength = obj.wavelength_vacuum / obj.medium_index;
            obj.k = 2 * pi / obj.wavelength;
            obj.rayleigh_range = pi * obj.waist_radius^2 / obj.wavelength;
            obj.divergence_angle = obj.wavelength / (pi * obj.waist_radius);
            obj.validateInput(wavelength, waist_radius);
        end
        function validateInput(obj, varargin)
            %VALIDATEINPUT Validate input parameters
            for i = 1:length(varargin)
                arg = varargin{i};
                if isnumeric(arg)
                    if ~all(isfinite(arg(:)))
                        error('GaussianBeam:InvalidInput', 'All parameters must be finite');
                    end
                    if any(arg(:) <= 0)
                        error('GaussianBeam:InvalidInput', 'Wavelength and waist radius must be positive');
                    end
                end
            end
        end
        function w = beamRadius(obj, z)
            %BEAMRADIUS Calculate beam radius at position z
            %   w = beamRadius(z)
            %
            %   Input:
            %       z - Axial position (meters)
            %
            %   Output:
            %       w - Beam radius (meters)
            z_rel = z - obj.waist_position;
            w = obj.waist_radius * sqrt(1 + (z_rel / obj.rayleigh_range)^2);
        end
        function R = radiusOfCurvature(obj, z)
            %RADIUSOFCURVATURE Calculate radius of curvature at position z
            %   R = radiusOfCurvature(z)
            %
            %   Input:
            %       z - Axial position (meters)
            %
            %   Output:
            %       R - Radius of curvature (meters)
            z_rel = z - obj.waist_position;
            if abs(z_rel) < 1e-12
                R = Inf;
            else
                R = z_rel * (1 + (obj.rayleigh_range / z_rel)^2);
            end
        end
        function phi = gouyPhase(obj, z)
            %GOUYPHASE Calculate Gouy phase at position z
            %   phi = gouyPhase(z)
            %
            %   Input:
            %       z - Axial position (meters)
            %
            %   Output:
            %       phi - Gouy phase (radians)
            z_rel = z - obj.waist_position;
            phi = atan(z_rel / obj.rayleigh_range);
        end
        function field = fieldProfile(obj, x, y, z)
            %FIELDPROFILE Calculate complex field amplitude profile
            %   field = fieldProfile(x, y, z)
            %
            %   Inputs:
            %       x, y - Transverse coordinate arrays (meters)
            %       z - Axial position (meters)
            %
            %   Output:
            %       field - Complex field amplitude matrix
            [X, Y] = meshgrid(x, y);
            r_squared = X.^2 + Y.^2;
            % Beam parameters at z
            w_z = obj.beamRadius(z);
            R_z = obj.radiusOfCurvature(z);
            phi_z = obj.gouyPhase(z);
            % Field amplitude
            amplitude = obj.waist_radius / w_z * sqrt(2 * obj.power / (pi * obj.waist_radius^2));
            % Phase terms
            phase_curvature = zeros(size(r_squared));
            if ~isinf(R_z)
                phase_curvature = obj.k * r_squared / (2 * R_z);
            end
            % Complete field
            field = amplitude * exp(-r_squared / w_z^2) .* ...
                   exp(-1i * (obj.k * (z - obj.waist_position) - phi_z + phase_curvature));
        end
        function intensity = intensityProfile(obj, x, y, z)
            %INTENSITYPROFILE Calculate intensity profile
            %   intensity = intensityProfile(x, y, z)
            %
            %   Inputs:
            %       x, y - Transverse coordinate arrays (meters)
            %       z - Axial position (meters)
            %
            %   Output:
            %       intensity - Intensity matrix (W/m²)
            field = obj.fieldProfile(x, y, z);
            intensity = abs(field).^2;
        end
        function [q, q_param] = complexBeamParameter(obj, z)
            %COMPLEXBEAMPARAMETER Calculate complex beam parameter
            %   [q, q_param] = complexBeamParameter(z)
            %
            %   Input:
            %       z - Axial position (meters)
            %
            %   Outputs:
            %       q - Complex beam parameter (meters)
            %       q_param - Structure with beam parameters
            z_rel = z - obj.waist_position;
            q = z_rel + 1i * obj.rayleigh_range;
            if nargout > 1
                q_param = struct();
                q_param.beam_radius = obj.beamRadius(z);
                q_param.radius_of_curvature = obj.radiusOfCurvature(z);
                q_param.gouy_phase = obj.gouyPhase(z);
                q_param.rayleigh_range = obj.rayleigh_range;
            end
        end
        function propagated_beam = propagateDistance(obj, distance)
            %PROPAGATEDISTANCE Propagate beam by specified distance
            %   propagated_beam = propagateDistance(distance)
            %
            %   Input:
            %       distance - Propagation distance (meters)
            %
            %   Output:
            %       propagated_beam - New GaussianBeam object
            new_waist_position = obj.waist_position + distance;
            propagated_beam = GaussianBeam(obj.wavelength_vacuum, obj.waist_radius, ...
                'WaistPosition', new_waist_position, ...
                'MediumIndex', obj.medium_index, ...
                'Power', obj.power);
        end
        function result = propagateThroughLens(obj, focal_length, lens_position)
            %PROPAGATETHROUGHLENS Propagate beam through thin lens
            %   result = propagateThroughLens(focal_length, lens_position)
            %
            %   Inputs:
            %       focal_length - Lens focal length (meters)
            %       lens_position - Z position of lens (meters)
            %
            %   Output:
            %       result - Structure with new beam parameters
            % Complex beam parameter at lens
            z_at_lens = lens_position - obj.waist_position;
            q_in = z_at_lens + 1i * obj.rayleigh_range;
            % ABCD matrix for thin lens
            A = 1; B = 0; C = -1/focal_length; D = 1;
            % Transform beam parameter
            q_out = (A * q_in + B) / (C * q_in + D);
            % Extract new beam parameters
            z_out_rel = real(q_out);
            zr_out = imag(q_out);
            new_waist_position = lens_position + z_out_rel;
            new_waist_radius = sqrt(zr_out * obj.wavelength / pi);
            result = struct();
            result.waist_position = new_waist_position;
            result.waist_radius = new_waist_radius;
            result.rayleigh_range = zr_out;
            result.beam = GaussianBeam(obj.wavelength_vacuum, new_waist_radius, ...
                'WaistPosition', new_waist_position, ...
                'MediumIndex', obj.medium_index, ...
                'Power', obj.power);
        end
        function plotPropagation(obj, z_range, varargin)
            %PLOTPROPAGATION Plot beam propagation
            %   plotPropagation(z_range, ...)
            %
            %   Parameters:
            %       'NumPoints' - Number of z points (default: 200)
            %       'ShowWaist' - Show waist position (default: true)
            %       'ShowRayleigh' - Show Rayleigh range (default: true)
            p = inputParser;
            addRequired(p, 'z_range', @(x) isnumeric(x) && length(x) == 2);
            addParameter(p, 'NumPoints', 200, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'ShowWaist', true, @islogical);
            addParameter(p, 'ShowRayleigh', true, @islogical);
            parse(p, z_range, varargin{:});
            z_positions = linspace(z_range(1), z_range(2), p.Results.NumPoints);
            beam_radii = arrayfun(@(z) obj.beamRadius(z), z_positions);
            figure('Position', [100, 100, 1000, 600]);
            % Plot beam envelope
            plot(z_positions * 1000, beam_radii * 1000, ...
                'Color', obj.berkeley_blue, 'LineWidth', 3, 'DisplayName', 'Beam Radius');
            hold on;
            plot(z_positions * 1000, -beam_radii * 1000, ...
                'Color', obj.berkeley_blue, 'LineWidth', 3, 'HandleVisibility', 'off');
            % Show waist
            if p.Results.ShowWaist
                plot(obj.waist_position * 1000, obj.waist_radius * 1000, ...
                    'o', 'Color', obj.california_gold, 'MarkerSize', 10, ...
                    'MarkerFaceColor', obj.california_gold, 'DisplayName', 'Beam Waist');
                plot(obj.waist_position * 1000, -obj.waist_radius * 1000, ...
                    'o', 'Color', obj.california_gold, 'MarkerSize', 10, ...
                    'MarkerFaceColor', obj.california_gold, 'HandleVisibility', 'off');
            end
            % Show Rayleigh range
            if p.Results.ShowRayleigh
                zr_positions = [obj.waist_position - obj.rayleigh_range, ...
                               obj.waist_position + obj.rayleigh_range] * 1000;
                for zr_pos = zr_positions
                    plot([zr_pos, zr_pos], [-max(beam_radii), max(beam_radii)] * 1000, ...
                        '--', 'Color', obj.berkeley_light_blue, 'LineWidth', 2, ...
                        'DisplayName', 'Rayleigh Range');
                end
            end
            % Optical axis
            plot(z_positions([1, end]) * 1000, [0, 0], 'k:', 'LineWidth', 1, ...
                'DisplayName', 'Optical Axis');
            xlabel('Position (mm)');
            ylabel('Beam Radius (mm)');
            title(sprintf('Gaussian Beam Propagation (λ = %.0f nm, w₀ = %.1f mm)', ...
                obj.wavelength_vacuum * 1e9, obj.waist_radius * 1000), ...
                'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            grid('alpha', 0.3);
            % Add beam parameters as text
            param_text = sprintf(['Wavelength: %.0f nm\n' ...
                                'Waist radius: %.2f mm\n' ...
                                'Rayleigh range: %.2f mm\n' ...
                                'Divergence: %.2f mrad\n' ...
                                'Power: %.1f mW'], ...
                obj.wavelength_vacuum * 1e9, obj.waist_radius * 1000, ...
                obj.rayleigh_range * 1000, obj.divergence_angle * 1000, ...
                obj.power * 1000);
            text(0.02, 0.98, param_text, 'Units', 'normalized', ...
                'VerticalAlignment', 'top', 'FontSize', 10, ...
                'BackgroundColor', 'white', 'EdgeColor', 'black');
            hold off;
        end
        function plotIntensityProfile(obj, z, varargin)
            %PLOTINTENSITYPROFILE Plot 2D intensity profile at position z
            %   plotIntensityProfile(z, ...)
            %
            %   Parameters:
            %       'Range' - Spatial range in mm (default: 5*beam_radius)
            %       'NumPoints' - Number of grid points (default: 100)
            p = inputParser;
            addRequired(p, 'z', @isnumeric);
            addParameter(p, 'Range', [], @isnumeric);
            addParameter(p, 'NumPoints', 100, @(x) isnumeric(x) && x > 0);
            parse(p, z, varargin{:});
            % Determine plotting range
            w_z = obj.beamRadius(z);
            if isempty(p.Results.Range)
                plot_range = 5 * w_z;
            else
                plot_range = p.Results.Range * 1e-3;  % Convert mm to m
            end
            % Create coordinate arrays
            x = linspace(-plot_range, plot_range, p.Results.NumPoints);
            y = linspace(-plot_range, plot_range, p.Results.NumPoints);
            % Calculate intensity
            intensity = obj.intensityProfile(x, y, z);
            % Plot
            figure('Position', [100, 100, 1200, 500]);
            % 2D intensity map
            subplot(1, 2, 1);
            [X, Y] = meshgrid(x * 1000, y * 1000);
            contourf(X, Y, intensity, 50, 'LineStyle', 'none');
            colormap('Blues');
            colorbar('Label', 'Intensity (W/m²)');
            xlabel('X Position (mm)');
            ylabel('Y Position (mm)');
            title(sprintf('Intensity Profile at z = %.1f mm', z * 1000), ...
                'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            axis equal;
            % Cross-section
            subplot(1, 2, 2);
            center_idx = ceil(length(y) / 2);
            x_profile = intensity(center_idx, :);
            y_profile = intensity(:, center_idx);
            plot(x * 1000, x_profile, 'Color', obj.berkeley_blue, ...
                'LineWidth', 2, 'DisplayName', 'Horizontal');
            hold on;
            plot(y * 1000, y_profile, 'Color', obj.california_gold, ...
                'LineWidth', 2, 'DisplayName', 'Vertical');
            xlabel('Position (mm)');
            ylabel('Intensity (W/m²)');
            title('Beam Cross-sections', 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            grid('alpha', 0.3);
            hold off;
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstrate Gaussian beam functionality
            fprintf('Gaussian Beam Demo\n');
            fprintf('==================\n\n');
            % Create Gaussian beam
            wavelength = 633e-9;  % HeNe laser
            waist_radius = 1e-3;  % 1 mm
            power = 1e-3;         % 1 mW
            beam = GaussianBeam(wavelength, waist_radius, 'Power', power);
            fprintf('Beam parameters:\n');
            fprintf('Wavelength: %.0f nm\n', wavelength * 1e9);
            fprintf('Waist radius: %.1f mm\n', waist_radius * 1000);
            fprintf('Rayleigh range: %.2f mm\n', beam.rayleigh_range * 1000);
            fprintf('Divergence angle: %.2f mrad\n', beam.divergence_angle * 1000);
            fprintf('Power: %.1f mW\n\n', power * 1000);
            % Plot propagation
            fprintf('1. Beam Propagation\n');
            z_range = [-5e-3, 5e-3];  % ±5 mm
            beam.plotPropagation(z_range);
            % Plot intensity profile at waist
            fprintf('2. Intensity Profile at Waist\n');
            beam.plotIntensityProfile(0);
            % Propagate through lens
            fprintf('3. Propagation through Lens\n');
            focal_length = 0.1;      % 100 mm focal length
            lens_position = 2e-3;    % 2 mm from waist
            result = beam.propagateThroughLens(focal_length, lens_position);
            fprintf('Lens parameters:\n');
            fprintf('Focal length: %.0f mm\n', focal_length * 1000);
            fprintf('Lens position: %.1f mm\n', lens_position * 1000);
            fprintf('\nNew beam parameters:\n');
            fprintf('New waist position: %.2f mm\n', result.waist_position * 1000);
            fprintf('New waist radius: %.3f mm\n', result.waist_radius * 1000);
            fprintf('New Rayleigh range: %.2f mm\n', result.rayleigh_range * 1000);
            % Plot focused beam
            focused_beam = result.beam;
            z_range_focused = [result.waist_position - 2e-3, result.waist_position + 2e-3];
            focused_beam.plotPropagation(z_range_focused);
            fprintf('\nDemo completed!\n');
        end
    end
end