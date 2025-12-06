classdef OpticsVisualization < handle
    %OPTICSVISUALIZATION Berkeley-themed visualization tools for optics
    %   Provides comprehensive visualization functionality for optical systems,
    %   beam profiles, ray diagrams, and interference patterns
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        % Berkeley color scheme
        berkeley_blue = [0, 50, 98]/255
        california_gold = [253, 181, 21]/255
        berkeley_light_blue = [59, 126, 161]/255
        berkeley_dark_gold = [196, 130, 14]/255
        berkeley_secondary_blue = [0, 176, 218]/255
        % Color palette
        colors
    end
    methods
        function obj = OpticsVisualization()
            %OPTICSVISUALIZATION Constructor
            obj.colors = [obj.berkeley_blue; obj.california_gold; ...
                         obj.berkeley_light_blue; obj.berkeley_dark_gold; ...
                         obj.berkeley_secondary_blue];
            obj.setupBerkeleyStyle();
        end
        function setupBerkeleyStyle(obj)
            %SETUPBERKELEYSTYLE Setup Berkeley visual styling for plots
            % Set default figure properties
            set(groot, 'DefaultFigurePosition', [100, 100, 1000, 600]);
            set(groot, 'DefaultAxesColorOrder', obj.colors);
            set(groot, 'DefaultAxesGridAlpha', 0.3);
            set(groot, 'DefaultAxesFontSize', 11);
            set(groot, 'DefaultAxesLabelFontSize', 12);
            set(groot, 'DefaultAxesTitleFontSize', 14);
            set(groot, 'DefaultLegendFontSize', 10);
            set(groot, 'DefaultLineLineWidth', 2);
            set(groot, 'DefaultAxesLineWidth', 1.2);
        end
        function fig = plotBeamProfile(obj, x, y, z, intensity, varargin)
            %PLOTBEAMPROFILE Plot 2D beam intensity profile
            %   fig = plotBeamProfile(x, y, z, intensity, ...)
            %
            %   Parameters:
            %       'Title' - Plot title (default: 'Beam Profile')
            %       'Units' - Spatial units ('m', 'mm', 'um', default: 'mm')
            %       'ShowCrossSections' - Show cross-sections (default: true)
            p = inputParser;
            addRequired(p, 'x', @isnumeric);
            addRequired(p, 'y', @isnumeric);
            addRequired(p, 'z', @isnumeric);
            addRequired(p, 'intensity', @isnumeric);
            addParameter(p, 'Title', 'Beam Profile', @ischar);
            addParameter(p, 'Units', 'mm', @ischar);
            addParameter(p, 'ShowCrossSections', true, @islogical);
            parse(p, x, y, z, intensity, varargin{:});
            % Unit conversion
            switch lower(p.Results.Units)
                case 'mm'
                    scale = 1000;
                    unit_str = 'mm';
                case 'um'
                    scale = 1e6;
                    unit_str = 'μm';
                case 'm'
                    scale = 1;
                    unit_str = 'm';
                otherwise
                    scale = 1000;
                    unit_str = 'mm';
            end
            x_scaled = x * scale;
            y_scaled = y * scale;
            if p.Results.ShowCrossSections
                fig = figure('Position', [100, 100, 1200, 500]);
                % 2D intensity map
                subplot(1, 2, 1);
                [X, Y] = meshgrid(x_scaled, y_scaled);
                contourf(X, Y, intensity, 50, 'LineStyle', 'none');
                colormap('Blues');
                colorbar('Label', 'Intensity (W/m²)', 'FontSize', 12);
                xlabel(sprintf('X Position (%s)', unit_str));
                ylabel(sprintf('Y Position (%s)', unit_str));
                title(sprintf('%s at z = %.2f %s', p.Results.Title, z*scale, unit_str), ...
                    'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                axis equal;
                grid on;
                % Cross-sections
                subplot(1, 2, 2);
                center_y = ceil(length(y) / 2);
                center_x = ceil(length(x) / 2);
                x_profile = intensity(center_y, :);
                y_profile = intensity(:, center_x);
                plot(x_scaled, x_profile, 'Color', obj.berkeley_blue, ...
                    'LineWidth', 2, 'DisplayName', 'Horizontal');
                hold on;
                plot(y_scaled, y_profile, 'Color', obj.california_gold, ...
                    'LineWidth', 2, 'DisplayName', 'Vertical');
                xlabel(sprintf('Position (%s)', unit_str));
                ylabel('Intensity (W/m²)');
                title('Beam Cross-sections', 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                legend('Location', 'best');
                grid on;
            else
                fig = figure('Position', [100, 100, 800, 600]);
                [X, Y] = meshgrid(x_scaled, y_scaled);
                contourf(X, Y, intensity, 50, 'LineStyle', 'none');
                colormap('Blues');
                colorbar('Label', 'Intensity (W/m²)', 'FontSize', 12);
                xlabel(sprintf('X Position (%s)', unit_str));
                ylabel(sprintf('Y Position (%s)', unit_str));
                title(sprintf('%s at z = %.2f %s', p.Results.Title, z*scale, unit_str), ...
                    'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                axis equal;
                grid on;
            end
            hold off;
        end
        function fig = plotRayDiagram(obj, rays, surfaces, varargin)
            %PLOTRAYDIAGRAM Plot ray diagram with optical elements
            %   fig = plotRayDiagram(rays, surfaces, ...)
            %
            %   Parameters:
            %       'Title' - Plot title (default: 'Ray Diagram')
            %       'Units' - Spatial units ('m', 'mm', default: 'mm')
            %       'ZRange' - Z-axis range (default: auto)
            %       'YRange' - Y-axis range (default: auto)
            p = inputParser;
            addRequired(p, 'rays', @iscell);
            addRequired(p, 'surfaces', @iscell);
            addParameter(p, 'Title', 'Ray Diagram', @ischar);
            addParameter(p, 'Units', 'mm', @ischar);
            addParameter(p, 'ZRange', [], @isnumeric);
            addParameter(p, 'YRange', [], @isnumeric);
            parse(p, rays, surfaces, varargin{:});
            % Unit conversion
            switch lower(p.Results.Units)
                case 'mm'
                    scale = 1000;
                    unit_str = 'mm';
                case 'm'
                    scale = 1;
                    unit_str = 'm';
                otherwise
                    scale = 1000;
                    unit_str = 'mm';
            end
            fig = figure('Position', [100, 100, 1200, 600]);
            % Plot rays
            all_z = [];
            all_y = [];
            for i = 1:length(rays)
                ray_data = rays{i};
                if iscell(ray_data)
                    % Ray path (multiple ray segments)
                    z_path = [];
                    y_path = [];
                    for j = 1:length(ray_data)
                        ray = ray_data{j};
                        z_path = [z_path, ray.position(3)];
                        y_path = [y_path, ray.position(1)];  % Use x as height
                    end
                else
                    % Single ray
                    z_path = ray_data.position(3);
                    y_path = ray_data.position(1);
                end
                % Plot ray path
                color_idx = mod(i-1, size(obj.colors, 1)) + 1;
                plot(z_path * scale, y_path * scale, '-', ...
                    'Color', obj.colors(color_idx, :), 'LineWidth', 1.5);
                hold on;
                all_z = [all_z, z_path];
                all_y = [all_y, y_path];
            end
            % Plot optical surfaces
            if ~isempty(surfaces)
                for i = 1:length(surfaces)
                    surface = surfaces{i};
                    obj.plotSurface(surface, scale);
                end
            end
            % Set axis ranges
            if ~isempty(p.Results.ZRange)
                if length(p.Results.ZRange) == 2
                    xlim(p.Results.ZRange * scale);
                end
            elseif ~isempty(all_z)
                z_margin = (max(all_z) - min(all_z)) * 0.1;
                xlim([min(all_z) - z_margin, max(all_z) + z_margin] * scale);
            end
            if ~isempty(p.Results.YRange)
                if length(p.Results.YRange) == 2
                    ylim(p.Results.YRange * scale);
                end
            elseif ~isempty(all_y)
                y_margin = max(abs(all_y)) * 0.1;
                ylim([-max(abs(all_y)) - y_margin, max(abs(all_y)) + y_margin] * scale);
            end
            % Optical axis
            if ~isempty(all_z)
                plot([min(all_z), max(all_z)] * scale, [0, 0], 'k:', 'LineWidth', 1);
            end
            % Formatting
            xlabel(sprintf('Optical Axis (%s)', unit_str));
            ylabel(sprintf('Height (%s)', unit_str));
            title(p.Results.Title, 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            grid on;
            axis equal;
            hold off;
        end
        function plotSurface(obj, surface, scale)
            %PLOTSURFACE Plot optical surface element
            if ~isempty(surface) && isfield(surface, 'position')
                z_pos = surface.position * scale;
                if isfield(surface, 'focal_length')
                    % Thin lens
                    height = 12.5 * scale / 1000;  % Default height
                    if isfield(surface, 'diameter')
                        height = surface.diameter * scale / 2;
                    end
                    plot([z_pos, z_pos], [-height, height], ...
                        'Color', obj.california_gold, 'LineWidth', 4);
                    % Focal points
                    if surface.focal_length > 0
                        f_pos = z_pos + surface.focal_length * scale;
                        plot(f_pos, 0, 'o', 'Color', obj.berkeley_blue, 'MarkerSize', 6);
                    end
                elseif isfield(surface, 'radius')
                    % Curved surface
                    height = 12.5 * scale / 1000;
                    if isfield(surface, 'diameter')
                        height = surface.diameter * scale / 2;
                    end
                    plot([z_pos, z_pos], [-height, height], ...
                        'Color', obj.berkeley_blue, 'LineWidth', 3);
                elseif isfield(surface, 'type') && strcmp(surface.type, 'mirror')
                    % Mirror
                    height = 12.5 * scale / 1000;
                    if isfield(surface, 'diameter')
                        height = surface.diameter * scale / 2;
                    end
                    plot([z_pos, z_pos], [-height, height], ...
                        'Color', [0.5, 0.5, 0.5], 'LineWidth', 6);
                end
            end
        end
        function fig = plotInterferencePattern(obj, x, intensity, varargin)
            %PLOTINTERFERENCEPATTERN Plot interference pattern
            %   fig = plotInterferencePattern(x, intensity, ...)
            %
            %   Parameters:
            %       'Title' - Plot title (default: 'Interference Pattern')
            %       'Units' - Spatial units ('m', 'mm', default: 'mm')
            %       'PatternInfo' - Pattern information structure
            %       'ShowZoom' - Show zoomed central region (default: false)
            p = inputParser;
            addRequired(p, 'x', @isnumeric);
            addRequired(p, 'intensity', @isnumeric);
            addParameter(p, 'Title', 'Interference Pattern', @ischar);
            addParameter(p, 'Units', 'mm', @ischar);
            addParameter(p, 'PatternInfo', struct(), @isstruct);
            addParameter(p, 'ShowZoom', false, @islogical);
            parse(p, x, intensity, varargin{:});
            % Unit conversion
            switch lower(p.Results.Units)
                case 'mm'
                    scale = 1000;
                    unit_str = 'mm';
                case 'um'
                    scale = 1e6;
                    unit_str = 'μm';
                case 'm'
                    scale = 1;
                    unit_str = 'm';
                otherwise
                    scale = 1000;
                    unit_str = 'mm';
            end
            x_scaled = x * scale;
            if p.Results.ShowZoom
                fig = figure('Position', [100, 100, 1200, 500]);
                % Full pattern
                subplot(1, 2, 1);
                plot(x_scaled, intensity, 'Color', obj.berkeley_blue, 'LineWidth', 2);
                fill_x = [x_scaled, fliplr(x_scaled)];
                fill_y = [intensity, zeros(size(intensity))];
                fill(fill_x, fill_y, obj.berkeley_blue, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                xlabel(sprintf('Position (%s)', unit_str));
                ylabel('Intensity');
                title(p.Results.Title, 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                grid on;
                % Zoomed central region
                subplot(1, 2, 2);
                central_range = max(abs(x_scaled)) * 0.3;  % 30% of full range
                central_mask = abs(x_scaled) <= central_range;
                plot(x_scaled(central_mask), intensity(central_mask), ...
                    'Color', obj.berkeley_blue, 'LineWidth', 2);
                fill_x_zoom = [x_scaled(central_mask), fliplr(x_scaled(central_mask))];
                fill_y_zoom = [intensity(central_mask), zeros(sum(central_mask), 1)'];
                fill(fill_x_zoom, fill_y_zoom, obj.berkeley_blue, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                xlabel(sprintf('Position (%s)', unit_str));
                ylabel('Intensity');
                title('Central Region (Zoomed)', 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                grid on;
            else
                fig = figure('Position', [100, 100, 1000, 600]);
                plot(x_scaled, intensity, 'Color', obj.berkeley_blue, 'LineWidth', 2);
                fill_x = [x_scaled, fliplr(x_scaled)];
                fill_y = [intensity, zeros(size(intensity))];
                fill(fill_x, fill_y, obj.berkeley_blue, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                xlabel(sprintf('Position (%s)', unit_str));
                ylabel('Intensity');
                title(p.Results.Title, 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                grid on;
            end
            % Add pattern information if provided
            if ~isempty(fieldnames(p.Results.PatternInfo))
                info = p.Results.PatternInfo;
                info_text = {};
                if isfield(info, 'wavelength')
                    info_text{end+1} = sprintf('λ = %.1f nm', info.wavelength * 1e9);
                end
                if isfield(info, 'fringe_spacing')
                    info_text{end+1} = sprintf('Δx = %.3f %s', info.fringe_spacing * scale, unit_str);
                end
                if isfield(info, 'visibility')
                    info_text{end+1} = sprintf('V = %.3f', info.visibility);
                end
                if ~isempty(info_text)
                    text(0.02, 0.98, strjoin(info_text, '\n'), ...
                        'Units', 'normalized', 'VerticalAlignment', 'top', ...
                        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
                end
            end
            hold off;
        end
        function fig = plotDiffractionPattern(obj, x, intensity, varargin)
            %PLOTDIFFRACTIONPATTERN Plot diffraction pattern
            %   fig = plotDiffractionPattern(x, intensity, ...)
            %
            %   Parameters:
            %       'Title' - Plot title (default: 'Diffraction Pattern')
            %       'Units' - Spatial units ('m', 'mm', default: 'mm')
            %       'ApertureInfo' - Aperture information structure
            %       'ShowLogScale' - Show log scale plot (default: true)
            p = inputParser;
            addRequired(p, 'x', @isnumeric);
            addRequired(p, 'intensity', @isnumeric);
            addParameter(p, 'Title', 'Diffraction Pattern', @ischar);
            addParameter(p, 'Units', 'mm', @ischar);
            addParameter(p, 'ApertureInfo', struct(), @isstruct);
            addParameter(p, 'ShowLogScale', true, @islogical);
            parse(p, x, intensity, varargin{:});
            % Unit conversion
            switch lower(p.Results.Units)
                case 'mm'
                    scale = 1000;
                    unit_str = 'mm';
                case 'um'
                    scale = 1e6;
                    unit_str = 'μm';
                case 'm'
                    scale = 1;
                    unit_str = 'm';
                otherwise
                    scale = 1000;
                    unit_str = 'mm';
            end
            x_scaled = x * scale;
            if p.Results.ShowLogScale
                fig = figure('Position', [100, 100, 1000, 800]);
                % Linear scale
                subplot(2, 1, 1);
                plot(x_scaled, intensity, 'Color', obj.berkeley_blue, 'LineWidth', 2);
                fill_x = [x_scaled, fliplr(x_scaled)];
                fill_y = [intensity, zeros(size(intensity))];
                fill(fill_x, fill_y, obj.berkeley_blue, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                ylabel('Intensity');
                title(p.Results.Title, 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                grid on;
                % Log scale
                subplot(2, 1, 2);
                semilogy(x_scaled, intensity, 'Color', obj.california_gold, 'LineWidth', 2);
                xlabel(sprintf('Position (%s)', unit_str));
                ylabel('Intensity (log scale)');
                title('Diffraction Pattern (Log Scale)', 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                grid on;
            else
                fig = figure('Position', [100, 100, 1000, 600]);
                plot(x_scaled, intensity, 'Color', obj.berkeley_blue, 'LineWidth', 2);
                fill_x = [x_scaled, fliplr(x_scaled)];
                fill_y = [intensity, zeros(size(intensity))];
                fill(fill_x, fill_y, obj.berkeley_blue, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                xlabel(sprintf('Position (%s)', unit_str));
                ylabel('Intensity');
                title(p.Results.Title, 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
                grid on;
            end
            % Add aperture information if provided
            if ~isempty(fieldnames(p.Results.ApertureInfo))
                info = p.Results.ApertureInfo;
                info_text = {};
                if isfield(info, 'aperture_type')
                    info_text{end+1} = sprintf('Aperture: %s', info.aperture_type);
                end
                if isfield(info, 'aperture_size')
                    size_scaled = info.aperture_size * scale;
                    info_text{end+1} = sprintf('Size: %.3f %s', size_scaled, unit_str);
                end
                if isfield(info, 'wavelength')
                    info_text{end+1} = sprintf('λ = %.1f nm', info.wavelength * 1e9);
                end
                if ~isempty(info_text)
                    text(0.02, 0.98, strjoin(info_text, '\n'), ...
                        'Units', 'normalized', 'VerticalAlignment', 'top', ...
                        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
                end
            end
            hold off;
        end
        function fig = createOpticalSystemDiagram(obj, elements, varargin)
            %CREATEOPTICALSYSTEMDIAGRAM Create comprehensive optical system diagram
            %   fig = createOpticalSystemDiagram(elements, ...)
            %
            %   Parameters:
            %       'Title' - Diagram title (default: 'Optical System')
            %       'Units' - Spatial units ('m', 'mm', default: 'mm')
            p = inputParser;
            addRequired(p, 'elements', @iscell);
            addParameter(p, 'Title', 'Optical System', @ischar);
            addParameter(p, 'Units', 'mm', @ischar);
            parse(p, elements, varargin{:});
            % Unit conversion
            switch lower(p.Results.Units)
                case 'mm'
                    scale = 1000;
                    unit_str = 'mm';
                case 'm'
                    scale = 1;
                    unit_str = 'm';
                otherwise
                    scale = 1000;
                    unit_str = 'mm';
            end
            fig = figure('Position', [100, 100, 1400, 800]);
            % Find z-range
            z_positions = [];
            for i = 1:length(elements)
                element = elements{i};
                if isfield(element, 'position')
                    z_positions = [z_positions, element.position];
                end
            end
            if ~isempty(z_positions)
                z_min = min(z_positions) - 50e-3;
                z_max = max(z_positions) + 50e-3;
            else
                z_min = -0.1;
                z_max = 0.1;
            end
            % Draw optical axis
            plot([z_min, z_max] * scale, [0, 0], 'k--', 'LineWidth', 1, 'Alpha', 0.5);
            hold on;
            % Draw elements
            for i = 1:length(elements)
                element = elements{i};
                obj.drawSystemElement(element, i, scale);
            end
            % Formatting
            xlim([z_min, z_max] * scale);
            xlabel(sprintf('Optical Axis (%s)', unit_str));
            ylabel(sprintf('Height (%s)', unit_str));
            title(p.Results.Title, 'FontSize', 16, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            grid on;
            axis equal;
            hold off;
        end
        function drawSystemElement(obj, element, index, scale)
            %DRAWSYSTEMELEMENT Draw individual optical element
            elem_type = '';
            if isfield(element, 'type')
                elem_type = element.type;
            end
            position = 0;
            if isfield(element, 'position')
                position = element.position * scale;
            end
            color = obj.colors(mod(index-1, size(obj.colors, 1)) + 1, :);
            switch lower(elem_type)
                case 'lens'
                    % Draw lens
                    focal_length = 0.1;
                    if isfield(element, 'focal_length')
                        focal_length = element.focal_length;
                    end
                    diameter = 25e-3 * scale;
                    if isfield(element, 'diameter')
                        diameter = element.diameter * scale / 2;
                    else
                        diameter = diameter / 2;
                    end
                    % Lens symbol
                    plot([position, position], [-diameter, diameter], ...
                        'Color', color, 'LineWidth', 4);
                    % Focal points
                    if focal_length > 0
                        f_pos = position + focal_length * scale;
                        plot(f_pos, 0, 'o', 'Color', color, 'MarkerSize', 6);
                        plot(position - focal_length * scale, 0, 'o', ...
                            'Color', color, 'MarkerSize', 6, 'MarkerFaceColor', 'none');
                    end
                    % Label
                    text(position, diameter + 5*scale/1000, sprintf('L%d', index), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                        'Color', color, 'FontWeight', 'bold');
                case 'mirror'
                    % Draw mirror
                    diameter = 25e-3 * scale / 2;
                    if isfield(element, 'diameter')
                        diameter = element.diameter * scale / 2;
                    end
                    plot([position, position], [-diameter, diameter], ...
                        'Color', color, 'LineWidth', 6);
                    % Label
                    text(position, diameter + 5*scale/1000, sprintf('M%d', index), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                        'Color', color, 'FontWeight', 'bold');
                case 'aperture'
                    % Draw aperture
                    diameter = 10e-3 * scale / 2;
                    if isfield(element, 'diameter')
                        diameter = element.diameter * scale / 2;
                    end
                    plot([position, position], [-50*scale/1000, -diameter], ...
                        'Color', color, 'LineWidth', 3);
                    plot([position, position], [diameter, 50*scale/1000], ...
                        'Color', color, 'LineWidth', 3);
                    % Label
                    text(position, 30*scale/1000, sprintf('A%d', index), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                        'Color', color, 'FontWeight', 'bold');
            end
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstrate visualization capabilities
            fprintf('Optics Visualization Demo\n');
            fprintf('========================\n\n');
            viz = OpticsVisualization();
            % 1. Gaussian beam profile
            fprintf('1. Gaussian Beam Profile\n');
            x = linspace(-5e-3, 5e-3, 100);
            y = linspace(-5e-3, 5e-3, 100);
            [X, Y] = meshgrid(x, y);
            % Gaussian intensity profile
            w0 = 1e-3;  % Beam waist
            intensity = exp(-2 * (X.^2 + Y.^2) / w0^2);
            viz.plotBeamProfile(x, y, 0, intensity, 'Title', 'Gaussian Beam');
            % 2. Interference pattern
            fprintf('2. Interference Pattern\n');
            x_screen = linspace(-5e-3, 5e-3, 1000);
            wavelength = 633e-9;
            slit_separation = 100e-6;
            % Double slit interference
            phase_diff = 2 * pi * slit_separation * x_screen / (wavelength * 1.0);
            intensity_interference = 4 * cos(phase_diff / 2).^2;
            pattern_info = struct();
            pattern_info.wavelength = wavelength;
            pattern_info.fringe_spacing = wavelength * 1.0 / slit_separation;
            pattern_info.visibility = 1.0;
            viz.plotInterferencePattern(x_screen, intensity_interference, ...
                'Title', 'Double Slit Interference', 'PatternInfo', pattern_info, ...
                'ShowZoom', true);
            % 3. Optical system diagram
            fprintf('3. Optical System Diagram\n');
            elements = {};
            elements{1} = struct('type', 'lens', 'position', 0, 'focal_length', 0.1, 'diameter', 25e-3);
            elements{2} = struct('type', 'aperture', 'position', 0.05, 'diameter', 5e-3);
            elements{3} = struct('type', 'lens', 'position', 0.15, 'focal_length', 0.05, 'diameter', 20e-3);
            viz.createOpticalSystemDiagram(elements, 'Title', 'Telescope System');
            fprintf('\nVisualization demo completed!\n');
        end
    end
end