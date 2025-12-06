classdef RayOptics < handle
    %RAYOPTICS Ray tracing and geometric optics system
    %   Comprehensive ray tracing functionality including ray propagation,
    %   optical surfaces, and system analysis
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        surfaces           % Cell array of optical surfaces
        medium_index       % Background medium refractive index
        wavelength         % Design wavelength (meters)
        % Berkeley color scheme
        berkeley_blue = [0, 50, 98]/255
        california_gold = [253, 181, 21]/255
        berkeley_light_blue = [59, 126, 161]/255
    end
    properties (Constant)
        SPEED_OF_LIGHT = 2.99792458e8  % m/s
    end
    methods
        function obj = RayOptics(varargin)
            %RAYOPTICS Constructor
            %   system = RayOptics(...)
            %
            %   Parameters:
            %       'MediumIndex' - Background medium index (default: 1.0)
            %       'Wavelength' - Design wavelength (meters, default: 589e-9)
            p = inputParser;
            addParameter(p, 'MediumIndex', 1.0, @(x) isnumeric(x) && x >= 1);
            addParameter(p, 'Wavelength', 589e-9, @(x) isnumeric(x) && x > 0);
            parse(p, varargin{:});
            obj.surfaces = {};
            obj.medium_index = p.Results.MediumIndex;
            obj.wavelength = p.Results.Wavelength;
        end
        function addSurface(obj, surface)
            %ADDSURFACE Add optical surface to system
            %   addSurface(surface)
            %
            %   Input:
            %       surface - Optical surface object
            if ~isa(surface, 'OpticalSurface')
                error('RayOptics:InvalidSurface', 'Surface must be an OpticalSurface object');
            end
            obj.surfaces{end+1} = surface;
            % Sort surfaces by position
            positions = cellfun(@(s) s.position, obj.surfaces);
            [~, sort_idx] = sort(positions);
            obj.surfaces = obj.surfaces(sort_idx);
        end
        function result = traceRay(obj, ray)
            %TRACERAY Trace ray through optical system
            %   result = traceRay(ray)
            %
            %   Input:
            %       ray - Ray object
            %
            %   Output:
            %       result - Structure with ray trace results
            if ~isa(ray, 'Ray')
                error('RayOptics:InvalidRay', 'Input must be a Ray object');
            end
            current_ray = ray;
            ray_path = {current_ray};
            success = true;
            error_message = '';
            try
                for i = 1:length(obj.surfaces)
                    surface = obj.surfaces{i};
                    % Find intersection point
                    [intersection_point, hit] = surface.findIntersection(current_ray);
                    if ~hit
                        continue;  % Ray misses surface
                    end
                    % Create ray at intersection
                    intersection_ray = Ray(intersection_point, current_ray.direction, ...
                        current_ray.wavelength, current_ray.intensity);
                    ray_path{end+1} = intersection_ray;
                    % Refract/reflect ray
                    if isa(surface, 'ThinLens')
                        new_ray = surface.refractRay(current_ray, intersection_point, ...
                            current_ray.wavelength);
                    elseif isa(surface, 'SphericalSurface')
                        new_ray = surface.refractRay(current_ray, intersection_point, ...
                            current_ray.wavelength);
                    elseif isa(surface, 'Mirror')
                        new_ray = surface.reflectRay(current_ray, intersection_point);
                    else
                        error('RayOptics:UnknownSurface', 'Unknown surface type');
                    end
                    current_ray = new_ray;
                    ray_path{end+1} = current_ray;
                end
            catch ME
                success = false;
                error_message = ME.message;
            end
            result = struct();
            result.rays = ray_path;
            result.success = success;
            result.error_message = error_message;
            result.final_ray = current_ray;
        end
        function results = traceRayBundle(obj, rays)
            %TRACERAYBUNDLE Trace multiple rays through system
            %   results = traceRayBundle(rays)
            %
            %   Input:
            %       rays - Cell array of Ray objects
            %
            %   Output:
            %       results - Cell array of trace results
            results = cell(size(rays));
            for i = 1:length(rays)
                results{i} = obj.traceRay(rays{i});
            end
        end
        function spot_diagram = calculateSpotDiagram(obj, object_height, screen_distance, varargin)
            %CALCULATESPOTDIAGRAM Calculate spot diagram at image plane
            %   spot_diagram = calculateSpotDiagram(object_height, screen_distance, ...)
            %
            %   Parameters:
            %       'NumRays' - Number of rays (default: 100)
            %       'MaxAperture' - Maximum aperture radius (default: auto)
            p = inputParser;
            addRequired(p, 'object_height', @isnumeric);
            addRequired(p, 'screen_distance', @isnumeric);
            addParameter(p, 'NumRays', 100, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'MaxAperture', [], @isnumeric);
            parse(p, object_height, screen_distance, varargin{:});
            % Determine aperture size
            if isempty(p.Results.MaxAperture)
                if ~isempty(obj.surfaces)
                    aperture_sizes = cellfun(@(s) s.diameter/2, obj.surfaces, ...
                        'UniformOutput', false);
                    valid_apertures = [aperture_sizes{:}];
                    max_aperture = min(valid_apertures(valid_apertures > 0));
                else
                    max_aperture = 1e-2;  % 1 cm default
                end
            else
                max_aperture = p.Results.MaxAperture;
            end
            % Generate ray bundle
            num_rays = p.Results.NumRays;
            ray_angles = linspace(0, 2*pi, num_rays);
            ray_radii = sqrt(rand(1, num_rays)) * max_aperture;
            rays = cell(1, num_rays);
            for i = 1:num_rays
                % Ray starting position (at object)
                start_x = ray_radii(i) * cos(ray_angles(i));
                start_y = ray_radii(i) * sin(ray_angles(i));
                start_pos = [start_x, start_y, -abs(screen_distance)/2];
                % Ray direction (towards first surface)
                if ~isempty(obj.surfaces)
                    target_z = obj.surfaces{1}.position;
                else
                    target_z = 0;
                end
                direction = [0, 0, 1];  % Parallel rays
                direction = direction / norm(direction);
                rays{i} = Ray(start_pos, direction, obj.wavelength);
            end
            % Trace rays
            results = obj.traceRayBundle(rays);
            % Extract final ray positions
            hit_positions = [];
            for i = 1:length(results)
                if results{i}.success && ~isempty(results{i}.rays)
                    final_ray = results{i}.final_ray;
                    % Propagate to screen
                    t_to_screen = (screen_distance - final_ray.position(3)) / final_ray.direction(3);
                    if t_to_screen > 0
                        screen_pos = final_ray.position + t_to_screen * final_ray.direction;
                        hit_positions = [hit_positions; screen_pos(1:2)];
                    end
                end
            end
            spot_diagram = struct();
            spot_diagram.positions = hit_positions;
            spot_diagram.screen_distance = screen_distance;
            spot_diagram.object_height = object_height;
            spot_diagram.num_rays = size(hit_positions, 1);
            if ~isempty(hit_positions)
                spot_diagram.rms_radius = sqrt(mean(sum(hit_positions.^2, 2)));
                spot_diagram.centroid = mean(hit_positions, 1);
            else
                spot_diagram.rms_radius = 0;
                spot_diagram.centroid = [0, 0];
            end
        end
        function mtf_data = calculateMTF(obj, spatial_frequencies, varargin)
            %CALCULATEMTF Calculate Modulation Transfer Function
            %   mtf_data = calculateMTF(spatial_frequencies, ...)
            %
            %   Parameters:
            %       'ImageDistance' - Image plane distance (default: auto)
            %       'FieldHeight' - Field height for calculation (default: 0)
            p = inputParser;
            addRequired(p, 'spatial_frequencies', @(x) isnumeric(x) && all(x >= 0));
            addParameter(p, 'ImageDistance', [], @isnumeric);
            addParameter(p, 'FieldHeight', 0, @isnumeric);
            parse(p, spatial_frequencies, varargin{:});
            % Simplified MTF calculation using geometric ray tracing
            % This is a basic implementation - real MTF requires wave optics
            if isempty(p.Results.ImageDistance)
                image_distance = obj.calculateImageDistance(Inf);  % For collimated input
            else
                image_distance = p.Results.ImageDistance;
            end
            % Calculate spot diagram
            spot_data = obj.calculateSpotDiagram(p.Results.FieldHeight, image_distance);
            % Estimate MTF from spot size (simplified)
            rms_radius = spot_data.rms_radius;
            if rms_radius == 0
                mtf_values = ones(size(spatial_frequencies));
            else
                % Approximation: MTF ≈ exp(-(2πσν)²/2) for Gaussian spot
                sigma = rms_radius / sqrt(2);  % Convert RMS to standard deviation
                mtf_values = exp(-2 * (pi * sigma * spatial_frequencies).^2);
            end
            mtf_data = struct();
            mtf_data.spatial_frequencies = spatial_frequencies;
            mtf_data.mtf_values = mtf_values;
            mtf_data.field_height = p.Results.FieldHeight;
            mtf_data.image_distance = image_distance;
        end
        function image_distance = calculateImageDistance(obj, object_distance)
            %CALCULATEIMAGEDISTANCE Calculate image distance for given object distance
            %   image_distance = calculateImageDistance(object_distance)
            if isempty(obj.surfaces)
                image_distance = object_distance;
                return;
            end
            % Simple paraxial calculation for thin lens systems
            total_power = 0;
            for i = 1:length(obj.surfaces)
                surface = obj.surfaces{i};
                if isa(surface, 'ThinLens')
                    total_power = total_power + 1/surface.focal_length;
                end
            end
            if total_power == 0
                image_distance = object_distance;
            else
                focal_length = 1/total_power;
                if isinf(object_distance)
                    image_distance = focal_length;
                else
                    image_distance = 1 / (1/focal_length - 1/object_distance);
                end
            end
        end
        function plotRayDiagram(obj, rays, varargin)
            %PLOTRAYDIAGRAM Plot ray diagram
            %   plotRayDiagram(rays, ...)
            %
            %   Parameters:
            %       'ZRange' - Z-axis plotting range (default: auto)
            %       'YRange' - Y-axis plotting range (default: auto)
            %       'ShowSurfaces' - Show optical surfaces (default: true)
            p = inputParser;
            addRequired(p, 'rays', @iscell);
            addParameter(p, 'ZRange', [], @isnumeric);
            addParameter(p, 'YRange', [], @isnumeric);
            addParameter(p, 'ShowSurfaces', true, @islogical);
            parse(p, rays, varargin{:});
            figure('Position', [100, 100, 1200, 600]);
            % Trace rays and plot
            colors = [obj.berkeley_blue; obj.california_gold; obj.berkeley_light_blue; ...
                     [0.8, 0.2, 0.2]; [0.2, 0.8, 0.2]];
            all_z = [];
            all_y = [];
            for i = 1:length(rays)
                result = obj.traceRay(rays{i});
                if result.success
                    % Extract ray path
                    z_path = [];
                    y_path = [];
                    for j = 1:length(result.rays)
                        ray = result.rays{j};
                        z_path = [z_path, ray.position(3)];
                        y_path = [y_path, ray.position(1)];  % Use x as height
                    end
                    % Plot ray
                    color_idx = mod(i-1, size(colors, 1)) + 1;
                    plot(z_path * 1000, y_path * 1000, '-', ...
                        'Color', colors(color_idx, :), 'LineWidth', 1.5);
                    hold on;
                    all_z = [all_z, z_path];
                    all_y = [all_y, y_path];
                end
            end
            % Plot optical surfaces
            if p.Results.ShowSurfaces && ~isempty(obj.surfaces)
                for i = 1:length(obj.surfaces)
                    surface = obj.surfaces{i};
                    obj.plotSurface(surface);
                end
            end
            % Set axis ranges
            if isempty(p.Results.ZRange) && ~isempty(all_z)
                z_margin = (max(all_z) - min(all_z)) * 0.1;
                zlim([min(all_z) - z_margin, max(all_z) + z_margin] * 1000);
            elseif ~isempty(p.Results.ZRange)
                xlim(p.Results.ZRange * 1000);
            end
            if isempty(p.Results.YRange) && ~isempty(all_y)
                y_margin = max(abs(all_y)) * 0.1;
                ylim([-max(abs(all_y)) - y_margin, max(abs(all_y)) + y_margin] * 1000);
            elseif ~isempty(p.Results.YRange)
                ylim(p.Results.YRange * 1000);
            end
            % Formatting
            xlabel('Optical Axis (mm)');
            ylabel('Height (mm)');
            title('Ray Diagram', 'FontSize', 14, 'Color', obj.berkeley_blue, 'FontWeight', 'bold');
            grid on;
            grid('alpha', 0.3);
            % Optical axis
            if ~isempty(all_z)
                plot([min(all_z), max(all_z)] * 1000, [0, 0], 'k:', 'LineWidth', 1);
            end
            hold off;
        end
        function plotSurface(obj, surface)
            %PLOTSURFACE Plot optical surface
            z_pos = surface.position * 1000;  % Convert to mm
            if isa(surface, 'ThinLens')
                % Draw lens
                height = surface.diameter * 1000 / 2;
                plot([z_pos, z_pos], [-height, height], ...
                    'Color', obj.california_gold, 'LineWidth', 4);
                % Focal points
                if surface.focal_length > 0
                    f_pos = (z_pos + surface.focal_length * 1000);
                    plot(f_pos, 0, 'o', 'Color', obj.berkeley_blue, 'MarkerSize', 6);
                end
            elseif isa(surface, 'SphericalSurface')
                % Draw curved surface (simplified)
                height = surface.diameter * 1000 / 2;
                plot([z_pos, z_pos], [-height, height], ...
                    'Color', obj.berkeley_blue, 'LineWidth', 3);
            elseif isa(surface, 'Mirror')
                % Draw mirror
                height = surface.diameter * 1000 / 2;
                plot([z_pos, z_pos], [-height, height], ...
                    'Color', [0.5, 0.5, 0.5], 'LineWidth', 6);
            end
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstrate ray optics functionality
            fprintf('Ray Optics Demo\n');
            fprintf('===============\n\n');
            % Create optical system
            system = RayOptics('Wavelength', 589e-9);
            % Add thin lens
            lens = ThinLens(0, 0.1, 0.025);  % f=100mm, D=25mm at z=0
            system.addSurface(lens);
            fprintf('Optical system created:\n');
            fprintf('Lens: f = %.0f mm, D = %.0f mm\n\n', ...
                lens.focal_length * 1000, lens.diameter * 1000);
            % Create ray bundle
            object_distance = 0.15;  % 150 mm
            object_height = 0.005;   % 5 mm
            rays = {};
            ray_heights = linspace(-object_height, object_height, 11);
            for i = 1:length(ray_heights)
                start_pos = [ray_heights(i), 0, -object_distance];
                direction = [0, 0, 1];  % Parallel rays
                rays{end+1} = Ray(start_pos, direction, 589e-9);
            end
            fprintf('Ray bundle created: %d rays\n', length(rays));
            fprintf('Object distance: %.0f mm\n', object_distance * 1000);
            fprintf('Object height: ±%.1f mm\n\n', object_height * 1000);
            % Calculate image distance
            image_distance = system.calculateImageDistance(object_distance);
            magnification = -image_distance / object_distance;
            fprintf('Paraxial calculation:\n');
            fprintf('Image distance: %.1f mm\n', image_distance * 1000);
            fprintf('Magnification: %.2fx\n\n', magnification);
            % Plot ray diagram
            fprintf('Plotting ray diagram...\n');
            system.plotRayDiagram(rays);
            % Calculate spot diagram
            fprintf('Calculating spot diagram...\n');
            spot_data = system.calculateSpotDiagram(0, image_distance, 'NumRays', 100);
            fprintf('Spot diagram results:\n');
            fprintf('Number of rays hitting screen: %d\n', spot_data.num_rays);
            fprintf('RMS spot radius: %.3f mm\n', spot_data.rms_radius * 1000);
            fprintf('Centroid position: (%.3f, %.3f) mm\n', ...
                spot_data.centroid(1) * 1000, spot_data.centroid(2) * 1000);
            % Plot spot diagram
            if ~isempty(spot_data.positions)
                figure('Position', [150, 150, 600, 600]);
                scatter(spot_data.positions(:, 1) * 1000, spot_data.positions(:, 2) * 1000, ...
                    36, 'o', 'MarkerEdgeColor', system.berkeley_blue, ...
                    'MarkerFaceColor', system.berkeley_light_blue, 'MarkerFaceAlpha', 0.6);
                xlabel('X Position (mm)');
                ylabel('Y Position (mm)');
                title('Spot Diagram', 'FontSize', 14, 'Color', system.berkeley_blue, 'FontWeight', 'bold');
                grid on;
                axis equal;
                % Add RMS circle
                theta = linspace(0, 2*pi, 100);
                rms_x = spot_data.rms_radius * cos(theta) * 1000;
                rms_y = spot_data.rms_radius * sin(theta) * 1000;
                plot(rms_x + spot_data.centroid(1) * 1000, ...
                     rms_y + spot_data.centroid(2) * 1000, ...
                     '--', 'Color', system.california_gold, 'LineWidth', 2);
                legend('Ray hits', 'RMS circle', 'Location', 'best');
            end
            fprintf('\nDemo completed!\n');
        end
    end
end