classdef ThinLens < OpticalSurface
    %THINLENS Thin lens optical element
    %   Implements thin lens approximation for ray tracing
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        focal_length    % Focal length (meters, positive for converging)
    end
    methods
        function obj = ThinLens(position, focal_length, diameter, varargin)
            %THINLENS Constructor
            %   lens = ThinLens(position, focal_length, diameter, ...)
            %
            %   Inputs:
            %       position - Z position (meters)
            %       focal_length - Focal length (meters)
            %       diameter - Clear aperture diameter (meters)
            %
            %   Parameters:
            %       'MaterialBefore' - Material before lens (default: 'air')
            %       'MaterialAfter' - Material after lens (default: 'air')
            % Call parent constructor
            obj@OpticalSurface(position, diameter, varargin{:});
            % Validate and set focal length
            if ~isnumeric(focal_length) || focal_length == 0
                error('ThinLens:InvalidFocalLength', 'Focal length must be non-zero');
            end
            obj.focal_length = focal_length;
        end
        function [intersection_point, hit] = findIntersection(obj, ray)
            %FINDINTERSECTION Find intersection with thin lens plane
            %   [intersection_point, hit] = findIntersection(ray)
            % Lens is at z = position, infinite in x-y
            % Ray: r(t) = r0 + t*d
            % Intersection when r(t)_z = position
            if abs(ray.direction(3)) < 1e-12
                % Ray parallel to lens plane
                hit = false;
                intersection_point = [];
                return;
            end
            % Parameter for intersection
            t = (obj.position - ray.position(3)) / ray.direction(3);
            if t < 1e-12
                % Intersection behind ray start
                hit = false;
                intersection_point = [];
                return;
            end
            % Intersection point
            intersection_point = ray.position + t * ray.direction;
            % Check aperture
            hit = obj.checkAperture(intersection_point);
        end
        function normal = surfaceNormal(obj, point)
            %SURFACENORMAL Surface normal at point (always along z-axis for thin lens)
            %   normal = surfaceNormal(point)
            normal = [0, 0, 1];  % Always pointing in +z direction
        end
        function new_ray = refractRay(obj, ray, intersection_point, wavelength)
            %REFRACTRAY Apply thin lens refraction
            %   new_ray = refractRay(ray, intersection_point, wavelength)
            % For thin lens, use paraxial approximation
            % The ray direction changes according to lens power
            % Height from optical axis
            h = sqrt(intersection_point(1)^2 + intersection_point(2)^2);
            % Incident direction
            incident_dir = ray.direction;
            % Thin lens formula for direction change
            % The deflection angle is proportional to height and lens power
            lens_power = 1 / obj.focal_length;
            % Deflection in x and y directions
            if h > 1e-12
                % Unit vector in radial direction
                radial_unit = [intersection_point(1), intersection_point(2), 0] / h;
                % Deflection angle (small angle approximation)
                deflection_angle = -h * lens_power;
                % New direction (small angle approximation)
                new_direction = incident_dir + deflection_angle * radial_unit;
            else
                % On-axis ray, no deflection
                new_direction = incident_dir;
            end
            % Normalize direction
            new_direction = new_direction / norm(new_direction);
            % Create new ray
            new_ray = Ray(intersection_point, new_direction, wavelength, ray.intensity);
        end
        function abcd = getABCDMatrix(obj)
            %GETABCDMATRIX Get ABCD matrix for thin lens
            %   abcd = getABCDMatrix()
            abcd = [1, 0; -1/obj.focal_length, 1];
        end
        function power = getLensPower(obj)
            %GETLENSPOWER Get lens power in diopters
            %   power = getLensPower()
            power = 1 / obj.focal_length;  % Diopters (1/m)
        end
        function [front_focal, back_focal] = getFocalPoints(obj)
            %GETFOCALPOINTS Get front and back focal point positions
            %   [front_focal, back_focal] = getFocalPoints()
            front_focal = obj.position - obj.focal_length;
            back_focal = obj.position + obj.focal_length;
        end
        function image_pos = calculateImagePosition(obj, object_pos)
            %CALCULATEIMAGEPOSITION Calculate image position for given object position
            %   image_pos = calculateImagePosition(object_pos)
            % Object distance (measured from lens)
            object_distance = obj.position - object_pos;
            if abs(object_distance) < 1e-12
                % Object at lens
                image_pos = Inf;
                return;
            end
            % Thin lens equation: 1/f = 1/s + 1/s'
            % where s is object distance and s' is image distance
            if isinf(object_distance)
                % Object at infinity
                image_distance = obj.focal_length;
            else
                image_distance = 1 / (1/obj.focal_length - 1/object_distance);
            end
            image_pos = obj.position + image_distance;
        end
        function magnification = calculateMagnification(obj, object_pos)
            %CALCULATEMAGNIFICATION Calculate lateral magnification
            %   magnification = calculateMagnification(object_pos)
            object_distance = obj.position - object_pos;
            image_pos = obj.calculateImagePosition(object_pos);
            image_distance = image_pos - obj.position;
            if isinf(image_distance) || abs(object_distance) < 1e-12
                magnification = Inf;
            else
                magnification = -image_distance / object_distance;
            end
        end
        function plotLens(obj, varargin)
            %PLOTLENS Plot lens schematic
            %   plotLens(...)
            %
            %   Parameters:
            %       'Scale' - Plotting scale factor (default: 1000 for mm)
            %       'ShowFocalPoints' - Show focal points (default: true)
            p = inputParser;
            addParameter(p, 'Scale', 1000, @isnumeric);
            addParameter(p, 'ShowFocalPoints', true, @islogical);
            parse(p, varargin{:});
            scale = p.Results.Scale;
            % Lens position and size
            z_lens = obj.position * scale;
            lens_height = obj.diameter * scale / 2;
            % Draw lens symbol
            hold on;
            if obj.focal_length > 0
                % Converging lens (biconvex shape)
                lens_x = [z_lens-2, z_lens-1, z_lens, z_lens+1, z_lens+2];
                lens_y_top = [lens_height*0.8, lens_height, lens_height, lens_height, lens_height*0.8];
                lens_y_bottom = -lens_y_top;
                plot(lens_x, lens_y_top, 'b-', 'LineWidth', 3);
                plot(lens_x, lens_y_bottom, 'b-', 'LineWidth', 3);
                plot([lens_x(1), lens_x(1)], [lens_y_bottom(1), lens_y_top(1)], 'b-', 'LineWidth', 3);
                plot([lens_x(end), lens_x(end)], [lens_y_bottom(end), lens_y_top(end)], 'b-', 'LineWidth', 3);
            else
                % Diverging lens (biconcave shape)
                lens_x = [z_lens-2, z_lens-1, z_lens, z_lens+1, z_lens+2];
                lens_y_top = [lens_height, lens_height*0.8, lens_height*0.6, lens_height*0.8, lens_height];
                lens_y_bottom = -lens_y_top;
                plot(lens_x, lens_y_top, 'r-', 'LineWidth', 3);
                plot(lens_x, lens_y_bottom, 'r-', 'LineWidth', 3);
                plot([lens_x(1), lens_x(1)], [lens_y_bottom(1), lens_y_top(1)], 'r-', 'LineWidth', 3);
                plot([lens_x(end), lens_x(end)], [lens_y_bottom(end), lens_y_top(end)], 'r-', 'LineWidth', 3);
            end
            % Show focal points
            if p.Results.ShowFocalPoints
                [f_front, f_back] = obj.getFocalPoints();
                plot(f_front * scale, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
                plot(f_back * scale, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
                % Focal point labels
                text(f_front * scale, -lens_height*0.2, 'F', 'HorizontalAlignment', 'center');
                text(f_back * scale, -lens_height*0.2, 'F''', 'HorizontalAlignment', 'center');
            end
            % Optical axis
            axis_range = [z_lens - abs(obj.focal_length)*scale*1.5, ...
                         z_lens + abs(obj.focal_length)*scale*1.5];
            plot(axis_range, [0, 0], 'k:', 'LineWidth', 1);
            hold off;
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstrate thin lens functionality
            fprintf('Thin Lens Demo\n');
            fprintf('==============\n\n');
            % Create converging lens
            focal_length = 0.1;  % 100 mm
            diameter = 0.025;    % 25 mm
            position = 0;        % At origin
            lens = ThinLens(position, focal_length, diameter);
            fprintf('Lens parameters:\n');
            fprintf('Focal length: %.0f mm\n', focal_length * 1000);
            fprintf('Diameter: %.0f mm\n', diameter * 1000);
            fprintf('Position: %.0f mm\n', position * 1000);
            fprintf('Power: %.1f D\n\n', lens.getLensPower());
            % Test ray tracing
            fprintf('Ray tracing test:\n');
            % Create test ray (parallel to axis)
            ray_start = [0.005, 0, -0.05];  % 5 mm height, 50 mm before lens
            ray_direction = [0, 0, 1];      % Parallel to axis
            wavelength = 589e-9;
            test_ray = Ray(ray_start, ray_direction, wavelength);
            fprintf('Input ray: position = [%.1f, %.1f, %.1f] mm\n', ray_start * 1000);
            fprintf('           direction = [%.1f, %.1f, %.1f]\n', ray_direction);
            % Find intersection
            [intersection, hit] = lens.findIntersection(test_ray);
            if hit
                fprintf('Intersection: [%.2f, %.2f, %.2f] mm\n', intersection * 1000);
                % Refract ray
                new_ray = lens.refractRay(test_ray, intersection, wavelength);
                fprintf('Output ray: direction = [%.3f, %.3f, %.3f]\n', new_ray.direction);
                % Find where ray crosses optical axis
                if abs(new_ray.direction(1)) > 1e-12
                    t_focus = -new_ray.position(1) / new_ray.direction(1);
                    focus_z = new_ray.position(3) + t_focus * new_ray.direction(3);
                    fprintf('Focus position: z = %.1f mm\n', focus_z * 1000);
                    fprintf('Expected focus: z = %.1f mm\n', (position + focal_length) * 1000);
                end
            else
                fprintf('Ray missed lens aperture\n');
            end
            % Test image calculation
            fprintf('\nImage calculation test:\n');
            object_positions = [-0.15, -0.2, -Inf];  % 150 mm, 200 mm, infinity
            for obj_pos = object_positions
                if isinf(obj_pos)
                    fprintf('Object at infinity:\n');
                else
                    fprintf('Object at %.0f mm:\n', obj_pos * 1000);
                end
                img_pos = lens.calculateImagePosition(obj_pos);
                magnification = lens.calculateMagnification(obj_pos);
                if isinf(img_pos)
                    fprintf('  Image at infinity\n');
                else
                    fprintf('  Image position: %.1f mm\n', img_pos * 1000);
                end
                if isinf(magnification)
                    fprintf('  Magnification: infinite\n');
                else
                    fprintf('  Magnification: %.2fx\n', magnification);
                end
                fprintf('\n');
            end
            % Plot lens
            fprintf('Plotting lens schematic...\n');
            figure('Position', [100, 100, 800, 400]);
            lens.plotLens();
            xlabel('Position (mm)');
            ylabel('Height (mm)');
            title(sprintf('Thin Lens (f = %.0f mm)', focal_length * 1000));
            grid on;
            axis equal;
            fprintf('Demo completed!\n');
        end
    end
end