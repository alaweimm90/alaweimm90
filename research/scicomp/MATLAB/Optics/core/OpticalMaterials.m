classdef OpticalMaterials < handle
    %OPTICALMATERIALS Optical materials database and dispersion models
    %   Comprehensive optical material properties including refractive index
    %   models, dispersion analysis, and material databases
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        materials_database  % Container for material properties
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
            {365e-9, 400e-9, 450e-9, 532e-9, 589e-9, 633e-9, 800e-9, 1550e-9});
    end
    methods
        function obj = OpticalMaterials()
            %OPTICALMATERIALS Constructor
            obj.createMaterialDatabase();
        end
        function createMaterialDatabase(obj)
            %CREATEMATERIALDATABASE Create database of optical materials
            obj.materials_database = containers.Map();
            % BK7 Glass (Schott)
            bk7_params = struct();
            bk7_params.name = 'BK7';
            bk7_params.dispersion_type = 'sellmeier';
            bk7_params.B = [1.03961212, 0.231792344, 1.01046945];
            bk7_params.C = [6.00069867e-3, 2.00179144e-2, 103.560653];
            bk7_params.wavelength_range = [310e-9, 2500e-9];
            bk7_params.abbe_number = 64.17;
            bk7_params.density = 2.51;  % g/cm³
            obj.materials_database('BK7') = bk7_params;
            % Fused Silica
            silica_params = struct();
            silica_params.name = 'Fused Silica';
            silica_params.dispersion_type = 'sellmeier';
            silica_params.B = [0.6961663, 0.4079426, 0.8974794];
            silica_params.C = [4.67914826e-3, 1.35120631e-2, 97.9340025];
            silica_params.wavelength_range = [210e-9, 3700e-9];
            silica_params.abbe_number = 67.8;
            silica_params.density = 2.20;
            obj.materials_database('SiO2') = silica_params;
            % Silicon
            si_params = struct();
            si_params.name = 'Silicon';
            si_params.dispersion_type = 'sellmeier';
            si_params.B = [10.6684293, 0.0030434748, 1.54133408];
            si_params.C = [0.301516485, 1.13475115, 1104.0];
            si_params.wavelength_range = [1200e-9, 14000e-9];
            si_params.bandgap = 1.12;  % eV
            si_params.density = 2.33;
            obj.materials_database('Si') = si_params;
            % Sapphire (Al2O3)
            sapphire_params = struct();
            sapphire_params.name = 'Sapphire';
            sapphire_params.dispersion_type = 'sellmeier';
            sapphire_params.B = [1.4313493, 0.65054713, 5.3414021];
            sapphire_params.C = [5.2799261e-3, 1.42382647e-2, 325.017834];
            sapphire_params.wavelength_range = [150e-9, 5500e-9];
            sapphire_params.abbe_number = 72.2;
            sapphire_params.density = 3.98;
            obj.materials_database('Al2O3') = sapphire_params;
            % Water
            water_params = struct();
            water_params.name = 'Water';
            water_params.dispersion_type = 'sellmeier';
            water_params.B = [5.684027565e-1, 1.726177391e-1, 2.086189578e-2];
            water_params.C = [5.101829712e-3, 1.821153936e-2, 2.620722293e-2];
            water_params.wavelength_range = [200e-9, 200000e-9];
            water_params.density = 1.0;
            obj.materials_database('H2O') = water_params;
            % Air (at 15°C, 760 Torr, 0.03% CO2)
            air_params = struct();
            air_params.name = 'Air';
            air_params.dispersion_type = 'cauchy';
            air_params.A = 1.000293;
            air_params.B = 0;
            air_params.C = 0;
            air_params.wavelength_range = [200e-9, 2000000e-9];
            air_params.density = 1.225e-3;
            obj.materials_database('air') = air_params;
        end
        function n = refractiveIndex(obj, material, wavelength, temperature)
            %REFRACTIVEINDEX Calculate refractive index
            %   n = refractiveIndex(material, wavelength, temperature)
            %
            %   Inputs:
            %       material - Material name or parameters structure
            %       wavelength - Wavelength (meters)
            %       temperature - Temperature (Kelvin, optional, default: 293.15)
            if nargin < 4
                temperature = 293.15;  % Room temperature
            end
            if ischar(material)
                if obj.materials_database.isKey(material)
                    params = obj.materials_database(material);
                else
                    % Default values for common materials
                    defaults = containers.Map(...
                        {'air', 'vacuum', 'water', 'glass', 'diamond'}, ...
                        {1.000293, 1.0, 1.333, 1.5, 2.4});
                    if defaults.isKey(lower(material))
                        n = defaults(lower(material));
                        return;
                    else
                        error('OpticalMaterials:UnknownMaterial', 'Unknown material: %s', material);
                    end
                end
            else
                params = material;
            end
            % Check wavelength range
            if isfield(params, 'wavelength_range')
                if wavelength < params.wavelength_range(1) || wavelength > params.wavelength_range(2)
                    warning('OpticalMaterials:WavelengthOutOfRange', ...
                        'Wavelength %.1f nm outside valid range for %s', ...
                        wavelength*1e9, params.name);
                end
            end
            % Calculate refractive index based on dispersion model
            switch params.dispersion_type
                case 'sellmeier'
                    n = obj.sellmeierIndex(wavelength, params.B, params.C);
                case 'cauchy'
                    n = obj.cauchyIndex(wavelength, params.A, params.B, params.C);
                case 'constant'
                    n = params.value;
                otherwise
                    error('OpticalMaterials:UnknownDispersion', ...
                        'Unknown dispersion model: %s', params.dispersion_type);
            end
        end
        function n = sellmeierIndex(obj, wavelength, B, C)
            %SELLMEIERINDEX Calculate refractive index using Sellmeier equation
            %   n = sellmeierIndex(wavelength, B, C)
            %
            %   Sellmeier equation: n²(λ) = 1 + Σ(Bᵢλ²/(λ² - Cᵢ))
            lambda_um = wavelength * 1e6;  % Convert to micrometers
            lambda_sq = lambda_um^2;
            n_squared = 1;
            for i = 1:length(B)
                n_squared = n_squared + B(i) * lambda_sq / (lambda_sq - C(i));
            end
            n = sqrt(n_squared);
        end
        function n = cauchyIndex(obj, wavelength, A, B, C)
            %CAUCHYINDEX Calculate refractive index using Cauchy equation
            %   n = cauchyIndex(wavelength, A, B, C)
            %
            %   Cauchy equation: n(λ) = A + B/λ² + C/λ⁴
            if nargin < 4, B = 0; end
            if nargin < 5, C = 0; end
            lambda_um = wavelength * 1e6;  % Convert to micrometers
            n = A + B / lambda_um^2 + C / lambda_um^4;
        end
        function ng = groupIndex(obj, material, wavelength)
            %GROUPINDEX Calculate group index ng = n - λ(dn/dλ)
            %   ng = groupIndex(material, wavelength)
            % Numerical derivative
            dlambda = wavelength * 1e-6;  % Small wavelength increment
            n_plus = obj.refractiveIndex(material, wavelength + dlambda);
            n_minus = obj.refractiveIndex(material, wavelength - dlambda);
            dn_dlambda = (n_plus - n_minus) / (2 * dlambda);
            n = obj.refractiveIndex(material, wavelength);
            ng = n - wavelength * dn_dlambda;
        end
        function gvd = groupVelocityDispersion(obj, material, wavelength)
            %GROUPVELOCITYDISPERSION Calculate group velocity dispersion
            %   gvd = groupVelocityDispersion(material, wavelength)
            %
            %   Output:
            %       gvd - Group velocity dispersion in ps²/km
            % Numerical second derivative
            dlambda = wavelength * 1e-6;
            ng_plus = obj.groupIndex(material, wavelength + dlambda);
            ng_center = obj.groupIndex(material, wavelength);
            ng_minus = obj.groupIndex(material, wavelength - dlambda);
            d2ng_dlambda2 = (ng_plus - 2*ng_center + ng_minus) / dlambda^2;
            % Convert to standard units
            lambda_um = wavelength * 1e6;
            gvd = -(lambda_um^3 / (2 * pi * obj.SPEED_OF_LIGHT)) * d2ng_dlambda2 * 1e21;  % ps²/km
        end
        function D = chromaticDispersion(obj, material, wavelength)
            %CHROMATICDISPERSION Calculate chromatic dispersion parameter
            %   D = chromaticDispersion(material, wavelength)
            %
            %   Output:
            %       D - Dispersion parameter in ps/(nm⋅km)
            gvd = obj.groupVelocityDispersion(material, wavelength);
            lambda_nm = wavelength * 1e9;
            D = -2 * pi * obj.SPEED_OF_LIGHT * gvd / lambda_nm^2 * 1e-6;
        end
        function abbe = abbeNumber(obj, material)
            %ABBENUMBER Calculate Abbe number
            %   abbe = abbeNumber(material)
            % Standard wavelengths for Abbe number
            lambda_d = 589.3e-9;  % Sodium D-line
            lambda_f = 486.1e-9;  % Hydrogen F-line
            lambda_c = 656.3e-9;  % Hydrogen C-line
            n_d = obj.refractiveIndex(material, lambda_d);
            n_f = obj.refractiveIndex(material, lambda_f);
            n_c = obj.refractiveIndex(material, lambda_c);
            abbe = (n_d - 1) / (n_f - n_c);
        end
        function alpha = absorptionCoefficient(obj, material, wavelength)
            %ABSORPTIONCOEFFICIENT Calculate absorption coefficient
            %   alpha = absorptionCoefficient(material, wavelength)
            %
            %   Output:
            %       alpha - Absorption coefficient (1/m)
            % Simplified absorption models
            switch lower(material)
                case 'bk7'
                    alpha = 0.001;  % Very low absorption in visible
                case 'sio2'
                    if wavelength > 200e-9
                        alpha = 0.0001;
                    else
                        alpha = 100;  % UV cutoff
                    end
                case 'si'
                    if wavelength < 1100e-9
                        alpha = 1e6;  % Bandgap absorption
                    else
                        alpha = 0.01;
                    end
                case 'h2o'
                    alpha = 0.01 * (wavelength * 1e6)^2;  % Increases with wavelength
                case 'air'
                    alpha = 1e-6;  % Minimal absorption
                otherwise
                    alpha = 0.001;  % Default low absorption
            end
        end
        function analysis = dispersionAnalysis(obj, material, wavelength_range, varargin)
            %DISPERSIONANALYSIS Perform comprehensive dispersion analysis
            %   analysis = dispersionAnalysis(material, wavelength_range, ...)
            %
            %   Parameters:
            %       'NumPoints' - Number of wavelength points (default: 1000)
            p = inputParser;
            addRequired(p, 'material', @(x) ischar(x) || isstruct(x));
            addRequired(p, 'wavelength_range', @(x) isnumeric(x) && length(x) == 2);
            addParameter(p, 'NumPoints', 1000, @(x) isnumeric(x) && x > 0);
            parse(p, material, wavelength_range, varargin{:});
            wavelengths = linspace(wavelength_range(1), wavelength_range(2), p.Results.NumPoints);
            % Calculate dispersion properties
            n = arrayfun(@(w) obj.refractiveIndex(material, w), wavelengths);
            ng = arrayfun(@(w) obj.groupIndex(material, w), wavelengths);
            gvd = arrayfun(@(w) obj.groupVelocityDispersion(material, w), wavelengths);
            % Find zero dispersion wavelengths
            zero_disp_idx = find(diff(sign(gvd)));
            zero_disp_wavelengths = wavelengths(zero_disp_idx);
            % Calculate Abbe number if in visible range
            abbe_number = [];
            if wavelength_range(1) <= 656.3e-9 && wavelength_range(2) >= 486.1e-9
                abbe_number = obj.abbeNumber(material);
            end
            % Package results
            analysis = struct();
            analysis.wavelengths = wavelengths;
            analysis.refractive_index = n;
            analysis.group_index = ng;
            analysis.gvd = gvd;
            analysis.zero_dispersion_wavelengths = zero_disp_wavelengths;
            analysis.abbe_number = abbe_number;
            analysis.wavelength_range = wavelength_range;
        end
        function plotDispersion(obj, material, wavelength_range, varargin)
            %PLOTDISPERSION Plot material dispersion curves
            %   plotDispersion(material, wavelength_range, ...)
            %
            %   Parameters:
            %       'NumPoints' - Number of points (default: 1000)
            %       'ShowGVD' - Show GVD plot (default: true)
            p = inputParser;
            addRequired(p, 'material', @(x) ischar(x) || isstruct(x));
            addRequired(p, 'wavelength_range', @(x) isnumeric(x) && length(x) == 2);
            addParameter(p, 'NumPoints', 1000, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'ShowGVD', true, @islogical);
            parse(p, material, wavelength_range, varargin{:});
            % Get dispersion analysis
            analysis = obj.dispersionAnalysis(material, wavelength_range, ...
                'NumPoints', p.Results.NumPoints);
            wavelengths_nm = analysis.wavelengths * 1e9;
            if p.Results.ShowGVD
                figure('Position', [100, 100, 1200, 500]);
                % Refractive index plot
                subplot(1, 2, 1);
                plot(wavelengths_nm, analysis.refractive_index, ...
                    'Color', obj.berkeley_blue, 'LineWidth', 2, 'DisplayName', 'n');
                hold on;
                plot(wavelengths_nm, analysis.group_index, ...
                    'Color', obj.california_gold, 'LineWidth', 2, 'DisplayName', 'n_g');
                xlabel('Wavelength (nm)');
                ylabel('Refractive Index');
                if ischar(material)
                    title(sprintf('%s - Refractive Index', material), ...
                        'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                else
                    title('Refractive Index', ...
                        'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                end
                legend('Location', 'best');
                grid on;
                grid('alpha', 0.3);
                % GVD plot
                subplot(1, 2, 2);
                plot(wavelengths_nm, analysis.gvd, ...
                    'Color', obj.berkeley_blue, 'LineWidth', 2);
                xlabel('Wavelength (nm)');
                ylabel('GVD (ps²/km)');
                if ischar(material)
                    title(sprintf('%s - Group Velocity Dispersion', material), ...
                        'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                else
                    title('Group Velocity Dispersion', ...
                        'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                end
                grid on;
                grid('alpha', 0.3);
                % Mark zero dispersion wavelengths
                if ~isempty(analysis.zero_dispersion_wavelengths)
                    hold on;
                    for zdw = analysis.zero_dispersion_wavelengths
                        plot([zdw*1e9, zdw*1e9], ylim, '--', ...
                            'Color', obj.california_gold, 'LineWidth', 2);
                    end
                    hold off;
                end
            else
                figure('Position', [100, 100, 800, 500]);
                % Refractive index only
                plot(wavelengths_nm, analysis.refractive_index, ...
                    'Color', obj.berkeley_blue, 'LineWidth', 2, 'DisplayName', 'n');
                hold on;
                plot(wavelengths_nm, analysis.group_index, ...
                    'Color', obj.california_gold, 'LineWidth', 2, 'DisplayName', 'n_g');
                xlabel('Wavelength (nm)');
                ylabel('Refractive Index');
                if ischar(material)
                    title(sprintf('%s - Refractive Index', material), ...
                        'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                else
                    title('Refractive Index', ...
                        'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                end
                legend('Location', 'best');
                grid on;
                grid('alpha', 0.3);
            end
            % Add Abbe number if available
            if ~isempty(analysis.abbe_number)
                text(0.02, 0.98, sprintf('Abbe number: %.1f', analysis.abbe_number), ...
                    'Units', 'normalized', 'VerticalAlignment', 'top', ...
                    'BackgroundColor', 'white', 'EdgeColor', 'black');
            end
            hold off;
        end
        function compareMaterials(obj, materials, wavelength_range, varargin)
            %COMPAREMATERIALS Compare dispersion of multiple materials
            %   compareMaterials(materials, wavelength_range, ...)
            %
            %   Parameters:
            %       'Property' - Property to compare ('n', 'ng', 'gvd', default: 'n')
            p = inputParser;
            addRequired(p, 'materials', @iscell);
            addRequired(p, 'wavelength_range', @(x) isnumeric(x) && length(x) == 2);
            addParameter(p, 'Property', 'n', @(x) any(strcmp(x, {'n', 'ng', 'gvd'})));
            parse(p, materials, wavelength_range, varargin{:});
            wavelengths = linspace(wavelength_range(1), wavelength_range(2), 1000);
            wavelengths_nm = wavelengths * 1e9;
            figure('Position', [100, 100, 1000, 600]);
            colors = [obj.berkeley_blue; obj.california_gold; obj.berkeley_light_blue; ...
                     [0.8, 0.2, 0.2]; [0.2, 0.8, 0.2]; [0.8, 0.2, 0.8]];
            for i = 1:length(materials)
                material = materials{i};
                color = colors(mod(i-1, size(colors, 1)) + 1, :);
                switch p.Results.Property
                    case 'n'
                        values = arrayfun(@(w) obj.refractiveIndex(material, w), wavelengths);
                        ylabel_str = 'Refractive Index';
                        title_str = 'Refractive Index Comparison';
                    case 'ng'
                        values = arrayfun(@(w) obj.groupIndex(material, w), wavelengths);
                        ylabel_str = 'Group Index';
                        title_str = 'Group Index Comparison';
                    case 'gvd'
                        values = arrayfun(@(w) obj.groupVelocityDispersion(material, w), wavelengths);
                        ylabel_str = 'GVD (ps²/km)';
                        title_str = 'Group Velocity Dispersion Comparison';
                end
                plot(wavelengths_nm, values, 'Color', color, 'LineWidth', 2, ...
                    'DisplayName', material);
                hold on;
            end
            xlabel('Wavelength (nm)');
            ylabel(ylabel_str);
            title(title_str, 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            grid('alpha', 0.3);
            hold off;
        end
        function listMaterials(obj)
            %LISTMATERIALS List available materials
            fprintf('Available Materials:\n');
            fprintf('===================\n');
            material_names = keys(obj.materials_database);
            for i = 1:length(material_names)
                name = material_names{i};
                params = obj.materials_database(name);
                fprintf('%d. %s (%s)\n', i, params.name, name);
                fprintf('   Dispersion: %s\n', params.dispersion_type);
                if isfield(params, 'wavelength_range')
                    fprintf('   Range: %.0f - %.0f nm\n', ...
                        params.wavelength_range(1)*1e9, params.wavelength_range(2)*1e9);
                end
                if isfield(params, 'abbe_number')
                    fprintf('   Abbe number: %.1f\n', params.abbe_number);
                end
                fprintf('\n');
            end
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstrate optical materials functionality
            fprintf('Optical Materials Demo\n');
            fprintf('=====================\n\n');
            % Create materials database
            materials = OpticalMaterials();
            % List available materials
            materials.listMaterials();
            % Calculate refractive indices at common wavelengths
            fprintf('Refractive indices at 589.3 nm (sodium D-line):\n');
            fprintf('================================================\n');
            wavelength = 589.3e-9;
            material_names = {'BK7', 'SiO2', 'air', 'H2O'};
            for mat = material_names
                n = materials.refractiveIndex(mat{1}, wavelength);
                fprintf('%s: n = %.6f\n', mat{1}, n);
            end
            % Analyze BK7 dispersion
            fprintf('\nBK7 Glass Analysis:\n');
            fprintf('==================\n');
            wl_range = [400e-9, 800e-9];  % Visible range
            analysis = materials.dispersionAnalysis('BK7', wl_range);
            fprintf('Wavelength range: %.0f - %.0f nm\n', wl_range(1)*1e9, wl_range(2)*1e9);
            if ~isempty(analysis.abbe_number)
                fprintf('Abbe number: %.1f\n', analysis.abbe_number);
            end
            % Find refractive index at specific wavelengths
            test_wavelengths = [486.1e-9, 589.3e-9, 656.3e-9];  % F, d, C lines
            line_names = {'F-line', 'd-line', 'C-line'};
            for i = 1:length(test_wavelengths)
                wl = test_wavelengths(i);
                n = materials.refractiveIndex('BK7', wl);
                ng = materials.groupIndex('BK7', wl);
                gvd = materials.groupVelocityDispersion('BK7', wl);
                fprintf('%s (%.1f nm): n = %.6f, ng = %.6f, GVD = %.2f ps²/km\n', ...
                    line_names{i}, wl*1e9, n, ng, gvd);
            end
            % Plot dispersion curves
            fprintf('\nPlotting BK7 dispersion...\n');
            materials.plotDispersion('BK7', [400e-9, 1600e-9]);
            % Compare materials
            fprintf('Comparing materials...\n');
            compare_materials = {'BK7', 'SiO2', 'H2O'};
            materials.compareMaterials(compare_materials, [400e-9, 800e-9]);
            fprintf('\nDemo completed!\n');
        end
    end
end