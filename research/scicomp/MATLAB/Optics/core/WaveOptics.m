classdef WaveOptics < handle
    %WAVEOPTICS Wave optics calculations and propagation
    %   Comprehensive wave optics functionality including plane waves,
    %   spherical waves, Gaussian beams, diffraction, and interference
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        wavelength_vacuum    % Wavelength in vacuum (meters)
        medium_index        % Refractive index of medium
        wavelength          % Wavelength in medium (meters)
        k                   % Wave number (rad/m)
        frequency           % Frequency (Hz)
        % Berkeley color scheme
        berkeley_blue = [0, 50, 98]/255
        california_gold = [253, 181, 21]/255
        berkeley_light_blue = [59, 126, 161]/255
    end
    properties (Constant)
        % Physical constants
        SPEED_OF_LIGHT = 2.99792458e8  % m/s
        PLANCK_CONSTANT = 6.62607015e-34  % J⋅s
        % Common wavelengths (meters)
        WAVELENGTHS = containers.Map(...
            {'UV_A', 'Violet', 'Blue', 'Green', 'Yellow', 'Red', 'NIR', 'Telecom_C'}, ...
            {365e-9, 400e-9, 450e-9, 532e-9, 589e-9, 633e-9, 800e-9, 1550e-9})
    end
    methods
        function obj = WaveOptics(wavelength, medium_index)
            %WAVEOPTICS Constructor
            %   obj = WaveOptics(wavelength, medium_index)
            %
            %   Inputs:
            %       wavelength - Wavelength in vacuum (meters)
            %       medium_index - Refractive index (default: 1.0)
            if nargin < 2
                medium_index = 1.0;
            end
            obj.wavelength_vacuum = wavelength;
            obj.medium_index = medium_index;
            obj.wavelength = wavelength / medium_index;
            obj.k = 2 * pi / obj.wavelength;
            obj.frequency = obj.SPEED_OF_LIGHT / wavelength;
        end
        function validateInput(obj, varargin)
            %VALIDATEINPUT Validate input parameters
            for i = 1:length(varargin)
                arg = varargin{i};
                if isnumeric(arg)
                    if ~all(isfinite(arg(:)))
                        error('WaveOptics:InvalidInput', 'All parameters must be finite');
                    end
                end
            end
        end
    end
    methods (Static)
        function beam = createGaussianBeam(wavelength, waist_radius, varargin)
            %CREATEGAUSSIANBEAM Create Gaussian beam object
            %   beam = createGaussianBeam(wavelength, waist_radius, ...)
            %
            %   Parameters:
            %       'Power' - Beam power (watts, default: 1e-3)
            %       'WaistPosition' - Z position of waist (meters, default: 0)
            %       'MediumIndex' - Refractive index (default: 1.0)
            p = inputParser;
            addRequired(p, 'wavelength', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'waist_radius', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Power', 1e-3, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'WaistPosition', 0, @isnumeric);
            addParameter(p, 'MediumIndex', 1.0, @(x) isnumeric(x) && x >= 1);
            parse(p, wavelength, waist_radius, varargin{:});
            beam = GaussianBeam(p.Results.wavelength, p.Results.waist_radius, ...
                p.Results.WaistPosition, p.Results.MediumIndex, p.Results.Power);
        end
        function result = calculateDiffraction(aperture_type, aperture_size, ...
                wavelength, screen_distance, screen_size, varargin)
            %CALCULATEDIFFRACTION Calculate diffraction patterns
            %   result = calculateDiffraction(aperture_type, aperture_size, ...
            %                                wavelength, screen_distance, screen_size)
            %
            %   Inputs:
            %       aperture_type - 'single_slit', 'double_slit', 'circular'
            %       aperture_size - Characteristic size (meters)
            %       wavelength - Wavelength (meters)
            %       screen_distance - Distance to screen (meters)
            %       screen_size - Size of observation screen (meters)
            %
            %   Optional Parameters:
            %       'NumPoints' - Number of calculation points (default: 1000)
            p = inputParser;
            addRequired(p, 'aperture_type', @ischar);
            addRequired(p, 'aperture_size', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'wavelength', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'screen_distance', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'screen_size', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'NumPoints', 1000, @(x) isnumeric(x) && x > 0);
            parse(p, aperture_type, aperture_size, wavelength, screen_distance, ...
                screen_size, varargin{:});
            x = linspace(-screen_size/2, screen_size/2, p.Results.NumPoints);
            switch lower(aperture_type)
                case 'single_slit'
                    intensity = WaveOptics.singleSlitDiffraction(x, aperture_size, ...
                        wavelength, screen_distance);
                case 'double_slit'
                    intensity = WaveOptics.doubleSlitDiffraction(x, aperture_size, ...
                        wavelength, screen_distance);
                case 'circular'
                    intensity = WaveOptics.circularAperture(x, aperture_size, ...
                        wavelength, screen_distance);
                otherwise
                    error('WaveOptics:InvalidAperture', 'Unknown aperture type: %s', aperture_type);
            end
            result = struct();
            result.position = x;
            result.intensity = intensity;
            result.intensity_normalized = intensity / max(intensity);
            result.aperture_type = aperture_type;
            result.aperture_size = aperture_size;
            result.wavelength = wavelength;
            result.screen_distance = screen_distance;
        end
        function intensity = singleSlitDiffraction(x, slit_width, wavelength, distance)
            %SINGLESLITDIFFRACTION Single slit diffraction pattern
            theta = x / distance;  % Small angle approximation
            beta = pi * slit_width * sin(theta) / wavelength;
            % Avoid division by zero
            beta(abs(beta) < 1e-10) = 1e-10;
            intensity = (sin(beta) ./ beta).^2;
        end
        function intensity = doubleSlitDiffraction(x, slit_separation, wavelength, distance)
            %DOUBLESLITDIFFRACTION Double slit diffraction pattern
            slit_width = slit_separation / 10;  % Assume thin slits
            theta = x / distance;
            beta = pi * slit_width * sin(theta) / wavelength;
            alpha = pi * slit_separation * sin(theta) / wavelength;
            beta(abs(beta) < 1e-10) = 1e-10;
            envelope = (sin(beta) ./ beta).^2;
            interference = cos(alpha).^2;
            intensity = envelope .* interference;
        end
        function intensity = circularAperture(x, aperture_radius, wavelength, distance)
            %CIRCULARAPERTURE Circular aperture (Airy disk) diffraction
            theta = x / distance;
            u = pi * aperture_radius * sin(theta) / wavelength;
            % First-order Bessel function
            u(abs(u) < 1e-10) = 1e-10;
            intensity = (2 * besselj(1, u) ./ u).^2;
        end
        function result = analyzeInterference(wavelength, source_separation, ...
                screen_distance, varargin)
            %ANALYZEINTERFERENCE Analyze interference patterns
            %   result = analyzeInterference(wavelength, source_separation, ...
            %                               screen_distance, ...)
            %
            %   Optional Parameters:
            %       'PatternType' - 'double_slit', 'young', 'michelson' (default: 'double_slit')
            %       'CoherenceLength' - Coherence length for partial coherence (meters)
            %       'NumPoints' - Number of calculation points (default: 1000)
            p = inputParser;
            addRequired(p, 'wavelength', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'source_separation', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'screen_distance', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'PatternType', 'double_slit', @ischar);
            addParameter(p, 'CoherenceLength', [], @(x) isnumeric(x) && x > 0);
            addParameter(p, 'NumPoints', 1000, @(x) isnumeric(x) && x > 0);
            parse(p, wavelength, source_separation, screen_distance, varargin{:});
            % Screen size
            screen_size = 10 * wavelength * screen_distance / source_separation;
            x = linspace(-screen_size/2, screen_size/2, p.Results.NumPoints);
            switch lower(p.Results.PatternType)
                case {'double_slit', 'young'}
                    theta = x / screen_distance;
                    path_difference = source_separation * sin(theta);
                    phase_difference = 2 * pi * path_difference / wavelength;
                    intensity = 4 * cos(phase_difference / 2).^2;
                    % Add partial coherence effects if specified
                    if ~isempty(p.Results.CoherenceLength)
                        coherence_factor = exp(-abs(path_difference) / p.Results.CoherenceLength);
                        visibility = coherence_factor;
                        intensity = 2 * (1 + visibility .* cos(phase_difference));
                    end
                case 'michelson'
                    phase_difference = 4 * pi * source_separation / wavelength;
                    intensity = 2 * (1 + cos(phase_difference));
                otherwise
                    error('WaveOptics:InvalidPattern', 'Unknown interference type: %s', p.Results.PatternType);
            end
            % Calculate fringe parameters
            fringe_spacing = wavelength * screen_distance / source_separation;
            % Visibility calculation
            I_max = max(intensity);
            I_min = min(intensity);
            visibility = (I_max - I_min) / (I_max + I_min);
            result = struct();
            result.position = x;
            result.intensity = intensity;
            result.intensity_normalized = intensity / max(intensity);
            result.fringe_spacing = fringe_spacing;
            result.visibility = visibility;
            result.pattern_type = p.Results.PatternType;
            result.wavelength = wavelength;
            result.source_separation = source_separation;
            result.screen_distance = screen_distance;
        end
        function fn = fresnelNumber(aperture_radius, wavelength, distance)
            %FRESNELNUMBER Calculate Fresnel number
            fn = aperture_radius^2 / (wavelength * distance);
        end
        function zr = rayleighRange(wavelength, waist_radius, medium_index)
            %RAYLEIGHRANGE Calculate Rayleigh range for Gaussian beam
            if nargin < 3
                medium_index = 1.0;
            end
            wavelength_medium = wavelength / medium_index;
            zr = pi * waist_radius^2 / wavelength_medium;
        end
        function demo()
            %DEMO Demonstrate wave optics functionality
            fprintf('Wave Optics Demo\n');
            fprintf('================\n\n');
            % Parameters
            wavelength = 633e-9;  % HeNe laser
            % 1. Gaussian beam propagation
            fprintf('1. Gaussian Beam Propagation\n');
            beam = WaveOptics.createGaussianBeam(wavelength, 1e-3, 'Power', 1e-3);
            z_positions = linspace(-5e-3, 5e-3, 100);
            beam_radii = arrayfun(@(z) beam.beamRadius(z), z_positions);
            figure('Position', [100, 100, 1200, 400]);
            subplot(1, 3, 1);
            plot(z_positions*1000, beam_radii*1000, 'Color', [0, 50, 98]/255, 'LineWidth', 2);
            hold on;
            plot(z_positions*1000, ones(size(z_positions))*beam.waist_radius*1000, ...
                '--', 'Color', [253, 181, 21]/255, 'LineWidth', 2);
            xlabel('Position (mm)');
            ylabel('Beam Radius (mm)');
            title('Gaussian Beam Propagation');
            legend('Beam Radius', 'Waist Radius');
            grid on;
            fprintf('Rayleigh range: %.2f mm\n', beam.rayleigh_range*1000);
            fprintf('Divergence angle: %.2f mrad\n', beam.divergence_angle*1000);
            % 2. Single slit diffraction
            fprintf('\n2. Single Slit Diffraction\n');
            diffraction = WaveOptics.calculateDiffraction(...
                'single_slit', 50e-6, wavelength, 1.0, 0.01);
            subplot(1, 3, 2);
            plot(diffraction.position*1000, diffraction.intensity_normalized, ...
                'Color', [0, 50, 98]/255, 'LineWidth', 2);
            xlabel('Position (mm)');
            ylabel('Normalized Intensity');
            title('Single Slit Diffraction');
            grid on;
            % 3. Double slit interference
            fprintf('3. Double Slit Interference\n');
            interference = WaveOptics.analyzeInterference(wavelength, 100e-6, 1.0);
            subplot(1, 3, 3);
            plot(interference.position*1000, interference.intensity_normalized, ...
                'Color', [253, 181, 21]/255, 'LineWidth', 2);
            xlabel('Position (mm)');
            ylabel('Normalized Intensity');
            title('Double Slit Interference');
            grid on;
            fprintf('Fringe spacing: %.3f mm\n', interference.fringe_spacing*1000);
            fprintf('Visibility: %.3f\n', interference.visibility);
            fprintf('\nDemo completed!\n');
        end
    end
end